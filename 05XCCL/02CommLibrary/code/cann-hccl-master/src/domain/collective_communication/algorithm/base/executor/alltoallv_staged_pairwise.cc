/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alltoallv_staged_pairwise.h"
#include "externalinput_pub.h"
#include "log.h"

namespace hccl {
using namespace std;

AlltoAllVStagedPairwise::AlltoAllVStagedPairwise(const HcclDispatcher dispatcher, Stream &stream)
    : AlltoAllVStagedBase(dispatcher, stream)
{}

AlltoAllVStagedPairwise::~AlltoAllVStagedPairwise() {}

// 图模式Prepare入口
HcclResult AlltoAllVStagedPairwise::Prepare(DeviceMem &sendMem, DeviceMem &recvMem,
    StageAlltoAllVAddrInfo &sendAddrInfo, StageAlltoAllVAddrInfo &recvAddrInfo,
    bool isAlltoAllZCopyMode, const std::vector<Stream> &subStreams)
{
    DeviceMem scratchInputMem = DeviceMem();
    DeviceMem scratchOutputMem = DeviceMem();
    CHK_RET(Prepare(sendMem, recvMem, scratchInputMem, scratchOutputMem, sendAddrInfo, recvAddrInfo,
        isAlltoAllZCopyMode));
    return HCCL_SUCCESS;
}

// 单算子Prepare入口
HcclResult AlltoAllVStagedPairwise::Prepare(DeviceMem &sendMem, DeviceMem &recvMem, DeviceMem &scratchInputMem,
    DeviceMem &scratchOutputMem, StageAlltoAllVAddrInfo &sendAddrInfo, StageAlltoAllVAddrInfo &recvAddrInfo,
    bool isAlltoAllZCopyMode, const std::vector<Stream> &subStreams)
{
    isAlltoAllZCopyMode_ = isAlltoAllZCopyMode;
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
        !isAlltoAllZCopyMode_) {
        CHK_PRT_RET((scratchInputMem.size() != scratchOutputMem.size()),
            HCCL_ERROR(
                "[AlltoAllVStagedPairwise][Prepare]scratchInputMem and scratchOutputMem should be the same size, "
                "ScratchInputMem[%llu] ScratchOutputMem[%llu]", scratchInputMem.size(), scratchOutputMem.size()),
                HCCL_E_MEMORY);

        CHK_PRT_RET(scratchInputMem.size() == 0,
            HCCL_ERROR("[AlltoAllVStagedPairwise][Prepare] invilad scratchMemSize[%llu]", scratchInputMem.size()),
            HCCL_E_PARA);
        scratchInputMem_ = scratchInputMem;
        scratchOutputMem_ = scratchOutputMem;
        scratchMemSize_ = scratchInputMem.size();
    }
    sendMem_ = sendMem;
    recvMem_ = recvMem;
    sendAddrInfo_ = sendAddrInfo;
    recvAddrInfo_ = recvAddrInfo;

    HCCL_DEBUG("[AlltoAllVStagedPairwise][Prepare] finished");
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVStagedPairwise::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links)
{
    HCCL_INFO("[AlltoAllVStagedPairwise][RunAsync]: rank[%u] transportSize[%llu]", rank, links.size());
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(mainStream_.ptr());

    CHK_PRT_RET(rankSize == 0, HCCL_ERROR("[AlltoAllVStagedPairwise][Prepare] invilad rankSize[%u]", rankSize),
        HCCL_E_PARA);

    CHK_PRT_RET(rankSize != links.size(),
        HCCL_ERROR("[AlltoAllVStagedPairwise][RunAsync]: rankSize[%u] and transport size[%llu] do not match", rankSize,
        links.size()),
        HCCL_E_PARA);

    bool sizeEqual = (sendAddrInfo_.size() == recvAddrInfo_.size() && sendAddrInfo_.size() == rankSize);
    CHK_PRT_RET(!sizeEqual,
        HCCL_ERROR("[AlltoAllVStagedPairwise][RunAsync] invilad params: "\
        "sendAddrInfo size[%u] recvAddrInfo size[%u] rankSize[%u]",
        sendAddrInfo_.size(), recvAddrInfo_.size(), rankSize),
        HCCL_E_PARA);

    CHK_RET(LocalCopy(rank));

    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
        !isAlltoAllZCopyMode_) { // 单算子 && BCopy模式
        CHK_RET(RunBCopyAlltoAll(rank, rankSize, links));
    } else {
        CHK_RET(RunZCopyAlltoAll(rank, rankSize, links));
    }

    return HCCL_SUCCESS;
}

HcclResult AlltoAllVStagedPairwise::RunZCopyAlltoAll(const u32 rank, const u32 rankSize, const std::vector<LINK> &links)
{
    for (u32 i = 1; i < rankSize; i++) {
        u32 prevRank = (rank + rankSize - i) % rankSize;
        u32 nextRank = (rank + i) % rankSize;
        std::shared_ptr<Transport> prevTransport = links[prevRank];
        std::shared_ptr<Transport> nextTransport = links[nextRank];

        CHK_SMART_PTR_NULL(prevTransport);
        CHK_SMART_PTR_NULL(nextTransport);

        HCCL_DEBUG("[AlltoAllVStagedPairwise][RunZCopyAlltoAll]: prevRank[%u] nextRank[%u], step[%u]", prevRank,
            nextRank, i);

        CHK_RET(prevTransport->TxAck(mainStream_)); // transport sync record
        CHK_RET(nextTransport->RxAck(mainStream_)); // transport sync wait

        u32 sendDataNum = sendAddrInfo_[nextRank].size();
        vector<TxMemoryInfo> txMems(sendDataNum);
        u32 index = 0;
        for (auto &addrInfo : sendAddrInfo_[nextRank]) {
            txMems[index].dstMemType = UserMemType::OUTPUT_MEM;
            txMems[index].dstOffset = addrInfo.remoteOffset;
            txMems[index].src = static_cast<u8 *>(sendMem_.ptr()) + addrInfo.localOffset;
            txMems[index].len = addrInfo.localLength;
            index++;
        }

        u32 recvDataNum = recvAddrInfo_[prevRank].size();
        vector<RxMemoryInfo> rxMems(recvDataNum);
        index = 0;
        for (auto &addrInfo : recvAddrInfo_[prevRank]) {
            rxMems[index].srcMemType = UserMemType::INPUT_MEM;
            rxMems[index].srcOffset = addrInfo.remoteOffset;
            rxMems[index].dst = static_cast<u8 *>(recvMem_.ptr()) + addrInfo.localOffset;
            rxMems[index].len = addrInfo.localLength;
            index++;
        }
        CHK_RET(nextTransport->TxAsync(txMems, mainStream_)); // send payload + data notify
        CHK_RET(prevTransport->RxAsync(rxMems, mainStream_)); // wait data notify
        CHK_RET(ExecuteBarrier(prevTransport, nextTransport));
    }

    return HCCL_SUCCESS;
}

HcclResult AlltoAllVStagedPairwise::RunBCopyAlltoAll(const u32 rank, const u32 rankSize,
    const std::vector<LINK> &links)
{
    for (u32 i = 1; i < rankSize; ++i) {
        u32 prevRank = (rank + rankSize - i) % rankSize;
        u32 nextRank = (rank + i) % rankSize;
        std::shared_ptr<Transport> prevTransport = links[prevRank];
        std::shared_ptr<Transport> nextTransport = links[nextRank];

        CHK_SMART_PTR_NULL(prevTransport);
        CHK_SMART_PTR_NULL(nextTransport);

        HCCL_DEBUG("[AlltoAllVStagedPairwise][RunBCopyAlltoAll]: prevRank[%u] nextRank[%u], step[%u]", prevRank,
            nextRank, i);

        // 计算本轮收发次数
        u64 sendTimes = 0;
        u64 recvTimes = 0;

        CalcSendRecvTimes(sendTimes, recvTimes, prevRank, nextRank);
        HCCL_DEBUG("sendTimes[%llu] recvTimes[%llu]", sendTimes, recvTimes);

        std::vector<std::list<OneSendRecvAddrInfo>> sendPolicies(sendTimes);
        std::vector<std::list<OneSendRecvAddrInfo>> recvPolicies(recvTimes);
        LoadPolicies(nextRank, sendAddrInfo_, sendPolicies);
        CHK_RET(CheckPolicies(sendTimes, sendPolicies));

        LoadPolicies(prevRank, recvAddrInfo_, recvPolicies);
        CHK_RET(CheckPolicies(recvTimes, recvPolicies));

        u64 curSendTime = 0;
        u64 curRecvTime = 0;

        while (curSendTime < sendTimes || curRecvTime < recvTimes) {
            CHK_RET(SendRecv(curSendTime, sendPolicies, curRecvTime, recvPolicies, prevTransport, nextTransport));
            curSendTime = curSendTime < sendTimes ? curSendTime + 1 : curSendTime;
            curRecvTime = curRecvTime < recvTimes ? curRecvTime + 1 : curRecvTime;
        }
    }
    return HCCL_SUCCESS;
}

void AlltoAllVStagedPairwise::CalcSendRecvTimes(u64 &sendTimes, u64 &recvTimes, const u32 prevRank, const u32 nextRank)
{
    u64 sendBytes = 0;
    u64 recvBytes = 0;
    for (auto &addrInfo : sendAddrInfo_[nextRank]) {
        sendBytes += addrInfo.localLength;
    }
    for (auto &addrInfo : recvAddrInfo_[prevRank]) {
        recvBytes += addrInfo.localLength;
    }
    sendTimes = (sendBytes / scratchMemSize_) + ((sendBytes % scratchMemSize_) == 0 ? 0 : 1);
    recvTimes = (recvBytes / scratchMemSize_) + ((recvBytes % scratchMemSize_) == 0 ? 0 : 1);
}

void AlltoAllVStagedPairwise::LoadPolicies(const u32 rank, StageAlltoAllVAddrInfo &addrInfos,
    std::vector<std::list<OneSendRecvAddrInfo>> &policies)
{
    std::list<OneSendRecvAddrInfo> tempPolicies;
    u64 curSendTime = 0;
    u64 curCCLBufSize = scratchMemSize_;
    // 当CCLbuf剩余空间不够发送、接收一整个task时，对task做拆分
    OneSendRecvAddrInfo curLastInfo;
    for (auto &addrInfo : addrInfos[rank]) {
        // 若当前task收发数据量为0，直接看下一个
        if (addrInfo.localLength == 0) {
            continue;
        }
        u64 curBytes = addrInfo.localLength;
        if (curBytes <= curCCLBufSize) {
            tempPolicies.push_back(addrInfo);
            curCCLBufSize -= curBytes;
            if (curCCLBufSize == 0) {
                curCCLBufSize = scratchMemSize_;
                policies[curSendTime] = tempPolicies;
                ++curSendTime;
                tempPolicies.clear();
            }
        } else {
            OneSendRecvAddrInfo tmpInfo = addrInfo;
            u64 tmpBytes = curBytes;
            while (tmpBytes > curCCLBufSize) {
                SplitSendRecvAddrInfo(curLastInfo, tmpInfo, curCCLBufSize);
                tempPolicies.push_back(curLastInfo);
                curCCLBufSize = scratchMemSize_;
                policies[curSendTime] = tempPolicies;
                ++curSendTime;
                tempPolicies.clear();
                tmpBytes = tmpInfo.localLength;
            }
            if (tmpBytes != 0) {
                tempPolicies.push_back(tmpInfo);
                curCCLBufSize -= tmpBytes;
            }
        }
    }
    if (curCCLBufSize != scratchMemSize_) {
        policies[curSendTime] = tempPolicies;
        ++curSendTime;
    }
}

HcclResult AlltoAllVStagedPairwise::CheckPolicies(const u64 times,
    const std::vector<std::list<OneSendRecvAddrInfo>> &policies) const
{
    CHK_PRT_RET(times != policies.size(),
        HCCL_ERROR(
            "[AlltoAllVStagedPairwise][CheckPolicies] invilad params: times[%llu] policies size[%u]", times,
            policies.size()), HCCL_E_PARA);

    for (u32 i = 0; i < times; ++i) {
        u64 sum = 0;
        for (auto &addrInfo : policies[i]) {
            sum += addrInfo.localLength;
        }
        CHK_PRT_RET(sum > scratchMemSize_,
            HCCL_ERROR(
                "[AlltoAllVStagedPairwise][CheckPolicies] invilad params: curTime[%u] sum[%llu] scratchMemSize_[%u]", i,
                sum, scratchMemSize_), HCCL_E_PARA);
    }
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVStagedPairwise::SendRecv(const u64 curSendTime,
    const std::vector<std::list<OneSendRecvAddrInfo>> &sendPolicies, const u64 curRecvTime,
    const std::vector<std::list<OneSendRecvAddrInfo>> &recvPolicies, std::shared_ptr<Transport> prevTransport,
    std::shared_ptr<Transport> nextTransport)
{
    bool hasSend = curSendTime < sendPolicies.size();
    bool hasRecv = curRecvTime < recvPolicies.size();
    if (hasRecv) {
        CHK_RET(prevTransport->TxAck(mainStream_)); // transport sync record
    }
    if (hasSend) {
        CHK_RET(nextTransport->RxAck(mainStream_)); // transport sync wait
    }
    if (hasSend) {
        // 1、把对应内存块从sendbuf copy到CCLInputBuf
        u64 curCCLInputBufOffset = 0;
        for (auto &addrInfo : sendPolicies[curSendTime]) {
            DeviceMem dstMem = scratchInputMem_.range(curCCLInputBufOffset, addrInfo.localLength);
            DeviceMem srcMem = sendMem_.range(addrInfo.localOffset, addrInfo.localLength);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, mainStream_));
            curCCLInputBufOffset += addrInfo.localLength;
        }
        // 2、send CCLInputBuf to CCLOutPutBuf + record
        CHK_RET(nextTransport->TxAsync(UserMemType::OUTPUT_MEM, 0, scratchInputMem_.ptr(),
            curCCLInputBufOffset, mainStream_));
    }

    if (hasRecv) {
        // 3、recv CCLOutPutBuf from CCLInputBuf
        u64 recvBytes = 0;
        for (auto &addrInfo : recvPolicies[curRecvTime]) {
            recvBytes += addrInfo.localLength;
        }
        // wait
        CHK_RET(prevTransport->RxAsync(UserMemType::INPUT_MEM, 0, scratchOutputMem_.ptr(), recvBytes, mainStream_));
        // 4、把对应内存块从CCLOutputBuf copy到recvBuf
        u64 curCCLOutputBufOffset = 0;
        for (auto &addrInfo : recvPolicies[curRecvTime]) {
            DeviceMem srcMem = scratchOutputMem_.range(curCCLOutputBufOffset, addrInfo.localLength);
            DeviceMem dstMem = recvMem_.range(addrInfo.localOffset, addrInfo.localLength);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, mainStream_));
            curCCLOutputBufOffset += addrInfo.localLength;
        }
    }

    CHK_RET(ExecuteBarrier(hasSend, hasRecv, prevTransport, nextTransport));

    return HCCL_SUCCESS;
}

void AlltoAllVStagedPairwise::SplitSendRecvAddrInfo(OneSendRecvAddrInfo &curLastInfo, OneSendRecvAddrInfo &addrInfo,
    const u64 &curCCLBufSize) const
{
    // 单算子收发暂不使用remote offset、len，考虑演进性，remote的数据也进行更新
    curLastInfo = addrInfo;
    curLastInfo.localLength = curCCLBufSize;
    curLastInfo.remoteLength = curCCLBufSize;

    // 切分后剩余部分，可能大于CCLbuf size，需要循环处理
    addrInfo.localOffset += curCCLBufSize;
    addrInfo.localLength -= curCCLBufSize;
    addrInfo.remoteOffset += curCCLBufSize;
    addrInfo.remoteLength -= curCCLBufSize;
}

HcclResult AlltoAllVStagedPairwise::ExecuteBarrier(std::shared_ptr<Transport> preLink,
    std::shared_ptr<Transport> aftLink)
{
    // 同步与preLink保证数据收发已结束
    CHK_RET(preLink->TxAck(mainStream_)); // record

    CHK_RET(aftLink->RxAck(mainStream_)); // wait

    // 同步与aftLink保证数据收发已结束
    CHK_RET(aftLink->TxDataSignal(mainStream_)); // record

    CHK_RET(preLink->RxDataSignal(mainStream_)); // wait

    CHK_RET(preLink->RxWaitDone(mainStream_));
    CHK_RET(aftLink->TxWaitDone(mainStream_));

    return HCCL_SUCCESS;
}

HcclResult AlltoAllVStagedPairwise::ExecuteBarrier(bool hasSend, bool hasRecv,
    std::shared_ptr<Transport> preLink, std::shared_ptr<Transport> aftLink)
{
    // 同步与preLink保证数据收发已结束
    if (hasRecv) {
        CHK_RET(preLink->TxAck(mainStream_)); // record
    }
    if (hasSend) {
        CHK_RET(aftLink->RxAck(mainStream_)); // wait
    }

    // 同步与aftLink保证数据收发已结束
    if (hasSend) {
        CHK_RET(aftLink->TxDataSignal(mainStream_)); // record
    }
    if (hasRecv) {
        CHK_RET(preLink->RxDataSignal(mainStream_)); // wait
    }

    if (hasRecv) {
        CHK_RET(preLink->RxWaitDone(mainStream_));
    }

    if (hasSend) {
        CHK_RET(aftLink->TxWaitDone(mainStream_));
    }

    return HCCL_SUCCESS;
}
} // namespace hccl
