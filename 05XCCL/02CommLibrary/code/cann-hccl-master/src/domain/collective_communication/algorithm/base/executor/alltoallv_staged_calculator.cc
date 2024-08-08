/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alltoallv_staged_calculator.h"
#include "log.h"

namespace hccl {
using namespace std;

AlltoAllVStagedCalculator::AlltoAllVStagedCalculator() {}

AlltoAllVStagedCalculator::~AlltoAllVStagedCalculator() {}

// / STATIC MEMBER FUNCTIONS BEGINS
HcclResult AlltoAllVStagedCalculator::CheckSendRecvParams(
    const std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo)
{
    u32 rankSize = allMeshAggregationSendRecvInfo.size();
    for (u32 i = 0; i < rankSize; i++) {
        u32 sendsSize = allMeshAggregationSendRecvInfo[i].sendLength.size();
        u32 recvsSize = allMeshAggregationSendRecvInfo[i].recvLength.size();
        if (rankSize != sendsSize || rankSize != recvsSize) {
            HCCL_ERROR(
                "[AlltoAllV][CheckSendRecvParam] rankSize[%u], sendsSize[%u], recvsSize[%u] are not match Index[%u]",
                rankSize, sendsSize, recvsSize, i);
            return HCCL_E_PARA;
        }
        for (u32 j = 0; j < sendsSize; j++) {
            if (allMeshAggregationSendRecvInfo[i].sendLength[j] != allMeshAggregationSendRecvInfo[j].recvLength[i]) {
                HCCL_ERROR("SendLength[%u][%u]: %llu and recvLength[%u][%u]: %llu are not match", i, j,
                    allMeshAggregationSendRecvInfo[i].sendLength[j], j, i,
                    allMeshAggregationSendRecvInfo[j].recvLength[i]);
                return HCCL_E_PARA;
            }
        }
    }
    return HCCL_SUCCESS;
}

void AlltoAllVStagedCalculator::CalcWorkSpaceMemSize(const AlltoAllUserRankInfo &userRankInfo,
    const std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo, u64 &workspaceMemSize,
    u32 meshAggregationRankSize)
{
    for (const auto &oneMeshAggregationSendRecvInfo : allMeshAggregationSendRecvInfo) {
        for (const auto &sendLength : oneMeshAggregationSendRecvInfo.sendLength) {
            HCCL_DEBUG("[CalcWorkSpaceMemSize] sendLength[%llu]", sendLength);
        }
        for (const auto &sendOffset : oneMeshAggregationSendRecvInfo.sendOffset) {
            HCCL_DEBUG("[CalcWorkSpaceMemSize] sendOffset[%llu]", sendOffset);
        }
        for (const auto &recvLength : oneMeshAggregationSendRecvInfo.recvLength) {
            HCCL_DEBUG("[CalcWorkSpaceMemSize] recvLength[%llu]", recvLength);
        }
        for (const auto &recvOffset : oneMeshAggregationSendRecvInfo.recvOffset) {
            HCCL_DEBUG("[CalcWorkSpaceMemSize] recvOffset[%llu]", recvOffset);
        }
    }
    if (allMeshAggregationSendRecvInfo.size() % meshAggregationRankSize != 0 ||
        allMeshAggregationSendRecvInfo.size() == 0) {
        workspaceMemSize = 0;
        HCCL_ERROR("Invalid Send Recv Info Size[%u]", allMeshAggregationSendRecvInfo.size());
        return;
    }
    workspaceMemSize = 0;
    u32 meshAggregationIndex = userRankInfo.userRank / meshAggregationRankSize;
    u32 meshAggregationRankBegin = meshAggregationIndex * meshAggregationRankSize;
    for (u32 infoIndex = userRankInfo.userRank % meshAggregationRankSize; infoIndex < userRankInfo.userRankSize;
        infoIndex += meshAggregationRankSize) {
        for (u32 k = meshAggregationRankBegin; k < meshAggregationRankBegin + meshAggregationRankSize; k++) {
            workspaceMemSize += allMeshAggregationSendRecvInfo[k].sendLength[infoIndex];
        }
    }
    HCCL_INFO("[AlltoAllVStagedCalculator][CalcWorkSpaceMemSize] workspaceMemSize[%llu]", workspaceMemSize);
}

void AlltoAllVStagedCalculator::CalcIntraMeshAggregationSendInfo(const AlltoAllUserRankInfo &userRankInfo,
    const SendRecvInfo &mySendRecvInfo, const std::vector<SendRecvInfo> &myMeshAggregationSendRecvInfo,
    u32 rankInMeshAggregation, u32 infoIndex, OneSendRecvAddrInfo &curSendInfo, u32 meshAggregationRankSize,
    const bool &isSingleMesh)
{
    (void)userRankInfo;
    if (infoIndex >= mySendRecvInfo.sendOffset.size() || infoIndex >= mySendRecvInfo.sendLength.size()) {
        HCCL_ERROR("[CalcIntraMeshAggregationSendInfo] Invalid infoIndex[%u]", infoIndex);
        return;
    }
    curSendInfo.localOffset = mySendRecvInfo.sendOffset[infoIndex];
    curSendInfo.localLength = mySendRecvInfo.sendLength[infoIndex];
    u64 remoteOffset = 0;

    if (isSingleMesh) {
        remoteOffset = myMeshAggregationSendRecvInfo[infoIndex].recvOffset[userRankInfo.userRank];
    } else {
        for (u32 j = infoIndex % meshAggregationRankSize; j <= infoIndex; j += meshAggregationRankSize) {
            for (u32 k = 0; k < meshAggregationRankSize; k++) {
                if (j == infoIndex && k == rankInMeshAggregation) {
                    break;
                }
                if (k < myMeshAggregationSendRecvInfo.size() && j <
                    myMeshAggregationSendRecvInfo[k].sendLength.size()) {
                    remoteOffset += myMeshAggregationSendRecvInfo[k].sendLength[j];
                } else {
                    HCCL_ERROR("[AlltoAllVStagedCalculator] invalid MeshAggregationSendRecvInfo size[%u]",
                        myMeshAggregationSendRecvInfo.size());
                    return;
                }
            }
        }
    }

    curSendInfo.remoteOffset = remoteOffset;
    curSendInfo.remoteLength = curSendInfo.localLength;
    HCCL_DEBUG("[CalcIntraMeshAggregationSendInfo] localOffset[%llu], localLength[%llu], "\
        "remoteOffset[%llu], remoteLength[%llu]", curSendInfo.localOffset, curSendInfo.localLength,
        curSendInfo.remoteOffset, curSendInfo.remoteLength);
}

void AlltoAllVStagedCalculator::CalcIntraMeshAggregationRecvInfoInMeshAggregation(u32 rankIndex, u32 infoIndex,
    const std::vector<SendRecvInfo> &myMeshAggregationSendRecvInfo, u64 &localOffset, u32 &offsetCounter,
    u64 &localLength, u64 &remoteOffset, u32 meshAggregationRankSize)
{
    // 这里的判断在外部已经保证了，为了应对coverity sc
    if (myMeshAggregationSendRecvInfo.size() < meshAggregationRankSize) {
        HCCL_ERROR("[CalcIntraMeshAggregationSendInfo] Invalid myMeshAggregationSendRecvInfo[%u]",
            myMeshAggregationSendRecvInfo.size());
        return;
    }
    if (myMeshAggregationSendRecvInfo[0].sendLength.size() == 0 ||
        myMeshAggregationSendRecvInfo[0].sendOffset.size() == 0) {
        HCCL_ERROR("[CalcIntraMeshAggregationSendInfo] Invalid sendLength size[%u] or sendOffset size[%u]",
            myMeshAggregationSendRecvInfo[0].sendLength.size(), myMeshAggregationSendRecvInfo[0].sendOffset.size());
        return;
    }
    for (u32 k = 0; k < meshAggregationRankSize; k++) {
        if (infoIndex == 0) {
            localOffset = 0;
            localLength = myMeshAggregationSendRecvInfo[k].sendLength[rankIndex];
            remoteOffset = myMeshAggregationSendRecvInfo[k].sendOffset[rankIndex];
            break;
        }

        localOffset += myMeshAggregationSendRecvInfo[k].sendLength[rankIndex];
        offsetCounter++;
        if (offsetCounter == infoIndex) {
            if (k == meshAggregationRankSize - 1) {
                localLength = myMeshAggregationSendRecvInfo[0].sendLength[rankIndex + meshAggregationRankSize];
                remoteOffset = myMeshAggregationSendRecvInfo[0].sendOffset[rankIndex + meshAggregationRankSize];
            } else {
                localLength = myMeshAggregationSendRecvInfo[k + 1].sendLength[rankIndex];
                remoteOffset = myMeshAggregationSendRecvInfo[k + 1].sendOffset[rankIndex];
            }
            break;
        }
    }
}

void AlltoAllVStagedCalculator::CalcIntraMeshAggregationRecvInfo(const AlltoAllUserRankInfo &userRankInfo,
    const std::vector<SendRecvInfo> &myMeshAggregationSendRecvInfo, u32 infoIndex, OneSendRecvAddrInfo &curRecvInfo,
    u32 meshAggregationRankSize, const bool &isSingleMesh)
{
    u64 localOffset = 0;
    u32 offsetCounter = 0;
    u64 localLength = 0;
    u64 remoteLength = 0;
    u64 remoteOffset = 0;

    if (isSingleMesh) {
        localOffset = myMeshAggregationSendRecvInfo[userRankInfo.userRank].recvOffset[infoIndex];
        localLength = myMeshAggregationSendRecvInfo[userRankInfo.userRank].recvLength[infoIndex];
        remoteLength = myMeshAggregationSendRecvInfo[infoIndex].sendLength[userRankInfo.userRank];
        remoteOffset = myMeshAggregationSendRecvInfo[infoIndex].sendOffset[userRankInfo.userRank];
    } else {
        for (u32 j = userRankInfo.userRank % meshAggregationRankSize; j < userRankInfo.userRankSize;
            j += meshAggregationRankSize) {
            CalcIntraMeshAggregationRecvInfoInMeshAggregation(j, infoIndex, myMeshAggregationSendRecvInfo, localOffset,
                offsetCounter, localLength, remoteOffset, meshAggregationRankSize);
            if (offsetCounter == infoIndex || infoIndex == 0) {
                break;
            }
        }
        remoteLength = localLength;
    }
    curRecvInfo.localOffset = localOffset;
    curRecvInfo.localLength = localLength;

    curRecvInfo.remoteOffset = remoteOffset;
    curRecvInfo.remoteLength = remoteLength;
    HCCL_DEBUG("[CalcIntraMeshAggregationRecvInfo] localOffset[%llu], localLength[%llu], "\
        "remoteOffset[%llu], remoteLength[%llu]", localOffset, localLength, remoteOffset, remoteLength);
}

void AlltoAllVStagedCalculator::CalcIntraMeshAggregationAlltoAllMemInfo(const AlltoAllUserRankInfo &userRankInfo,
    const std::vector<SendRecvInfo> &allSendRecvInfo, map<u32, list<OneSendRecvAddrInfo>> &sendAddrInfosIntra,
    map<u32, list<OneSendRecvAddrInfo>> &recvAddrInfosIntra, u32 meshAggregationRankSize, const bool &isSingleMesh)
{
    sendAddrInfosIntra.clear();
    recvAddrInfosIntra.clear();
    if (allSendRecvInfo.size() != userRankInfo.userRankSize) {
        HCCL_ERROR("Invalid All send recv info size[%u], should be[%u]", allSendRecvInfo.size(),
            userRankInfo.userRankSize);
        return;
    }
    SendRecvInfo mySendRecvInfo = allSendRecvInfo[userRankInfo.userRank];
    u32 rankInMeshAggregation = userRankInfo.userRank % meshAggregationRankSize;
    u32 cluserIndex = userRankInfo.userRank / meshAggregationRankSize;
    auto itBegin = allSendRecvInfo.begin();
    auto itEnd = allSendRecvInfo.begin();
    std::advance(itBegin, cluserIndex * meshAggregationRankSize);
    std::advance(itEnd, (cluserIndex + 1) * meshAggregationRankSize);
    std::vector<SendRecvInfo> myMeshAggregationSendRecvInfo(itBegin, itEnd);

    for (u32 i = 0; i < userRankInfo.userRankSize; i++) {
        // sendInfo 的计算
        OneSendRecvAddrInfo curSendInfo;
        u32 remoteRecvRankInMeshAggregation = i % meshAggregationRankSize;

        CalcIntraMeshAggregationSendInfo(userRankInfo, mySendRecvInfo, myMeshAggregationSendRecvInfo,
            rankInMeshAggregation, i, curSendInfo, meshAggregationRankSize, isSingleMesh);
        sendAddrInfosIntra[remoteRecvRankInMeshAggregation].push_back(curSendInfo);

        // recvInfo 的计算
        OneSendRecvAddrInfo curRecvInfo;
        u32 remoteSendRankInmeshAggregation = i % meshAggregationRankSize;
        CalcIntraMeshAggregationRecvInfo(userRankInfo, myMeshAggregationSendRecvInfo, i,
            curRecvInfo, meshAggregationRankSize, isSingleMesh);
        recvAddrInfosIntra[remoteSendRankInmeshAggregation].push_back(curRecvInfo);
    }
}

void AlltoAllVStagedCalculator::CalcInterMeshAggregationRecvRemoteOffset(const AlltoAllUserRankInfo &userRankInfo,
    const std::vector<SendRecvInfo> &allSendRecvInfo, u32 index, u64 &remoteOffset, u32 meshAggregationRankSize)
{
    // 对于stage1 来说，相当于是从rand index 发送给 userRankInfo.userRank, 然后计算这种情况下的stage1 的接收偏移
    remoteOffset = 0;
    u32 anchoruserRank_ = index;
    u32 anchorIndex = userRankInfo.userRank;
    u32 beginIndex = anchorIndex % meshAggregationRankSize;
    u32 beginRank = anchoruserRank_ / meshAggregationRankSize * meshAggregationRankSize;
    bool getAnchor = false;
    for (index = beginIndex; index <= anchorIndex; index += meshAggregationRankSize) {
        for (u32 rank = beginRank; rank < beginRank + meshAggregationRankSize; rank++) {
            if (index == anchorIndex && rank == anchoruserRank_) {
                getAnchor = true;
                break;
            }
            remoteOffset += allSendRecvInfo[rank].sendLength[index];
        }
        if (getAnchor) {
            break;
        }
    }
}

void AlltoAllVStagedCalculator::CalcInterMeshAggregationAlltoAllMemInfo(
    const AlltoAllUserRankInfo &userRankInfo, const std::vector<SendRecvInfo> &allSendRecvInfo,
    map<u32, list<OneSendRecvAddrInfo>> &sendAddrInfosInter, map<u32, list<OneSendRecvAddrInfo>> &recvAddrInfosInter,
    u32 meshAggregationRankSize)
{
    sendAddrInfosInter.clear();
    recvAddrInfosInter.clear();

    u64 localOffsetMarker = 0;
    for (u32 toRank = 0; toRank < userRankInfo.userRankSize; toRank++) {
        u32 myRank = userRankInfo.userRank;
        u32 myMeshAggregationRankBegin = myRank / meshAggregationRankSize * meshAggregationRankSize;
        u32 myMeshAggregationRankEnd = myMeshAggregationRankBegin + meshAggregationRankSize;

        for (u32 myMeshAggregationRank = myMeshAggregationRankBegin; myMeshAggregationRank < myMeshAggregationRankEnd;
            myMeshAggregationRank++) {
            if (toRank % meshAggregationRankSize == myRank % meshAggregationRankSize) {
                OneSendRecvAddrInfo sendAddrInfo;
                sendAddrInfo.localLength = allSendRecvInfo[myMeshAggregationRank].sendLength[toRank];
                sendAddrInfo.localOffset = localOffsetMarker;
                localOffsetMarker += sendAddrInfo.localLength;
                sendAddrInfo.remoteOffset = allSendRecvInfo[toRank].recvOffset[myMeshAggregationRank];
                sendAddrInfo.remoteLength = allSendRecvInfo[toRank].recvLength[myMeshAggregationRank];
                u32 remoteRankInter = toRank / meshAggregationRankSize;
                sendAddrInfosInter[remoteRankInter].push_back(sendAddrInfo);
                HCCL_DEBUG("[CalcInterMeshAggregationAlltoAllMemInfo] sendAddrInfo localOffset[%llu], "\
                    "localLength[%llu], remoteOffset[%llu], remoteLength[%llu]", sendAddrInfo.localOffset,
                    sendAddrInfo.localLength, sendAddrInfo.remoteOffset, sendAddrInfo.remoteLength);
            }
        }
    }

    //  构造接收数据结构
    for (u32 index = 0; index < userRankInfo.userRankSize; index++) {
        OneSendRecvAddrInfo recvAddrInfo;
        u32 meshAggregationIndex = index / meshAggregationRankSize;

        recvAddrInfo.localOffset = allSendRecvInfo[userRankInfo.userRank].recvOffset[index];
        recvAddrInfo.localLength = allSendRecvInfo[userRankInfo.userRank].recvLength[index];
        // index 是 从那个rank 来的
        recvAddrInfo.remoteLength = allSendRecvInfo[index].sendLength[userRankInfo.userRank];
        u64 remoteOffset = 0;
        CalcInterMeshAggregationRecvRemoteOffset(userRankInfo, allSendRecvInfo, index, remoteOffset,
            meshAggregationRankSize);

        recvAddrInfo.remoteOffset = remoteOffset;
        recvAddrInfosInter[meshAggregationIndex].push_back(recvAddrInfo);
        HCCL_DEBUG("[CalcInterMeshAggregationAlltoAllMemInfo] recvAddrInfo localOffset[%llu], "\
            "localLength[%llu], remoteOffset[%llu], remoteLength[%llu]", recvAddrInfo.localOffset,
            recvAddrInfo.localLength, recvAddrInfo.remoteOffset, recvAddrInfo.remoteLength);
    }
}
// / STATIC MEMBER FUNCTIONS ENDS
} // namespace hccl
