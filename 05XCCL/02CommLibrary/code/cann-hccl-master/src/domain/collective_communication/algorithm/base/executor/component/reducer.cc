/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reducer.h"

namespace hccl {
Reducer::Reducer(const HcclDataType dataType, const HcclReduceOp reductionOp, const u64 reduceAttribute)
    : dataType_(dataType), reductionOp_(reductionOp), reduceAttribute_(reduceAttribute)
{
}

Reducer::~Reducer()
{
}

HcclResult Reducer::run(const HcclDispatcher dispatcher, const std::shared_ptr<Transport> &link,
    const u64 remoteMemOffset, DeviceMem &localSrc, DeviceMem &localDst, DeviceMem &remoteRcvTemp, Stream &stream,
    DstMemType resultMem, const UserMemType srcMemType) const
{
    CHK_PTR_NULL(localSrc.ptr());
    CHK_PTR_NULL(localDst.ptr());
    CHK_PTR_NULL(remoteRcvTemp.ptr());
    CHK_PTR_NULL(stream.ptr());

    HcclResult ret = HCCL_SUCCESS;

    u64 dataBytes = remoteRcvTemp.size();
    HCCL_DEBUG("localSrc[%p] localDst[%p] remoteRcvtmep[%p] offset[%llu]", localSrc.ptr(), localDst.ptr(),
        remoteRcvTemp.ptr(), remoteMemOffset);

    // server 内 reduce 并且 reduceAttribute_ 也支持，走该分支
    bool isSpInlineReduce = link->IsSpInlineReduce();
    if (link->IsSupportTransportWithReduce() && (RDMA_REDUCE_BITMASK & reduceAttribute_)) {
        // 数据接收端执行接收动作
        // RDMA的RxAsync不需要接收端内存信息
        CHK_RET(link->RxAsync(UserMemType::INPUT_MEM, remoteMemOffset, remoteRcvTemp.ptr(), dataBytes, stream));
        if (link->GetSupportDataReceivedAck()) {
            CHK_RET(link->DataReceivedAck(stream));
        }
        if (resultMem == DstMemType::RESULT_OUTPUT_MEM) {
            ret = HcclD2DMemcpyAsync(dispatcher, localDst, localSrc, stream);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Reducer][Run]memcpy_async localSrc[%p] localDst[%p] failed", localSrc.ptr(),
                localDst.ptr()),
                ret);
        }
    } else if (link->IsSupportTransportWithReduce() && link->GetLinkType() == LinkType::LINK_STANDARD_ROCE) {
        u64 dataCount = localDst.size() / SIZE_TABLE[dataType_];
        DeviceMem &reduceSrc = (localSrc == localDst) ? remoteRcvTemp : localSrc;
        CHK_RET(link->RxWithReduce(srcMemType, remoteMemOffset, remoteRcvTemp.ptr(), dataBytes,
            reduceSrc.ptr(), localDst.ptr(), dataCount, dataType_, reductionOp_, stream, reduceAttribute_));
    } else if (isSpInlineReduce && (INLINE_REDUCE_BITMASK & reduceAttribute_)) {
        //  runtime 的inline reduce 接口参数为数据的字节长度
        CHK_RET(link->RxDataSignal(stream));
        void *remoteMem = nullptr;
        CHK_RET(link->GetRemoteMem(srcMemType, &remoteMem));
        CHK_RET(HcclReduceAsync(dispatcher, static_cast<s8 *>(remoteMem) + remoteMemOffset,
            dataBytes / SIZE_TABLE[dataType_], dataType_, reductionOp_, stream, localSrc.ptr(), link->GetRemoteRank(),
            link->GetLinkType(), INLINE_REDUCE_BIT));

        if (localSrc != localDst) {
            ret = HcclD2DMemcpyAsync(dispatcher, localDst, localSrc, stream);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Reducer][Run]memcpy_async localSrc[%p] localDst[%p] failed", localSrc.ptr(),
                localDst.ptr()),
                ret);
        }
        if (link -> GetSupportDataReceivedAck()) {
            CHK_RET(link->TxAck(stream));
            CHK_RET(link->RxAck(stream));
            CHK_RET(link->TxDataSignal(stream));
            CHK_RET(link->RxDataSignal(stream));
        }
    } else {
        // 从上一个节点接收数据
        ret = link->RxAsync(UserMemType::INPUT_MEM, remoteMemOffset, remoteRcvTemp.ptr(), dataBytes, stream);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Reducer][Run]rx_sync remoteRcvTemp[%p] offset[%llu] size[%llu] "
            "failed",
            remoteRcvTemp.ptr(), remoteMemOffset, dataBytes),
            ret);

        if (link->GetSupportDataReceivedAck()) {
            ret = link->DataReceivedAck(stream);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Reducer][Run]rx_sync data received ack failed"), ret);
        }

        u64 dataCount = localDst.size() / SIZE_TABLE[dataType_];

        // 根据目的内存执行reduce
        DeviceMem reduceSrc = (localSrc == localDst) ? remoteRcvTemp : localSrc;
        ret = HcclReduceAsync(dispatcher, reduceSrc.ptr(), dataCount, dataType_, reductionOp_, stream, localDst.ptr(),
            INVALID_VALUE_RANKID, LinkType::LINK_ONCHIP, reduceAttribute_);

        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Reducer][Run]reduce_async remoteRcvTemp[%p] localSrc[%p] "
            "localDst[%p] failed",
            remoteRcvTemp.ptr(), localSrc.ptr(), localDst.ptr()),
            ret);
    }

    return ret;
}

HcclResult Reducer::run(const HcclDispatcher dispatcher, const std::shared_ptr<Transport> &link,
    const std::vector<ReducerMemoryInfo> &reducerMems, Stream &stream, DstMemType resultMem) const
{
    CHK_PTR_NULL(stream.ptr());

    LinkType linkType = link->GetLinkType();
    bool isSpInlineReduce = link->IsSpInlineReduce();
    bool isSpRdmaReduce = RDMA_REDUCE_BITMASK & reduceAttribute_;
    bool isSpTransportWithReduce = link->IsSupportTransportWithReduce();
    HcclResult ret = HCCL_SUCCESS;

    std::vector<RxMemoryInfo> rxMems;
    for (const ReducerMemoryInfo &reduceMem : reducerMems) {
        rxMems.emplace_back(RxMemoryInfo{ UserMemType::INPUT_MEM, reduceMem.remoteMemOffset,
            reduceMem.remoteRcvTemp.ptr(), reduceMem.remoteRcvTemp.size() });
    }

    std::vector<RxWithReduceMemoryInfo> rxWithReduceMems;
    for (ReducerMemoryInfo reduceMem : reducerMems) {
        u64 dataCount = reduceMem.localdst.size() / SIZE_TABLE[dataType_];
        DeviceMem reduceSrc = (reduceMem.localsrc == reduceMem.localdst) ? reduceMem.remoteRcvTemp : reduceMem.localsrc;

        rxWithReduceMems.emplace_back(RxWithReduceMemoryInfo{ UserMemType::INPUT_MEM, reduceMem.remoteMemOffset,
            reduceMem.remoteRcvTemp.ptr(), reduceMem.remoteRcvTemp.size(), reduceSrc.ptr(), reduceMem.localdst.ptr(),
            dataCount });
    }

    if (isSpTransportWithReduce && isSpRdmaReduce) {
        // 数据接收端执行接收动作
        // RDMA的RxAsync不需要接收端内存信息
        CHK_RET(link->RxAsync(rxMems, stream));
        if (link->GetSupportDataReceivedAck()) {
            CHK_RET(link->DataReceivedAck(stream));
        }

        if (resultMem == DstMemType::RESULT_OUTPUT_MEM) {
            for (ReducerMemoryInfo reduceMem : reducerMems) {
                ret = HcclD2DMemcpyAsync(dispatcher, reduceMem.localdst, reduceMem.localsrc, stream);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[Reducer][Run]memcpy_async localSrc[%p] localDst[%p] failed", reduceMem.localsrc.ptr(),
                    reduceMem.localdst.ptr()),
                    ret);
            }
        }
    } else if (isSpTransportWithReduce && (linkType == LinkType::LINK_STANDARD_ROCE)) {
        CHK_RET(link->RxWithReduce(rxWithReduceMems, dataType_, reductionOp_, stream, reduceAttribute_));
    } else if (isSpInlineReduce && (INLINE_REDUCE_BITMASK & reduceAttribute_)) {
        CHK_RET(link->RxDataSignal(stream));
        void *remoteMem = nullptr;
        CHK_RET(link->GetRemoteMem(UserMemType::INPUT_MEM, &remoteMem));
        for (ReducerMemoryInfo reduceMem : reducerMems) {
            const u64 dataBytes = reduceMem.remoteRcvTemp.size();
            CHK_RET(
                HcclReduceAsync(dispatcher, static_cast<s8 *>(remoteMem) + reduceMem.remoteMemOffset,
                dataBytes / SIZE_TABLE[dataType_], dataType_, reductionOp_, stream, reduceMem.localsrc.ptr(),
                link->GetRemoteRank(), link->GetLinkType(), INLINE_REDUCE_BIT));

            if (reduceMem.localsrc != reduceMem.localdst) {
                ret = HcclD2DMemcpyAsync(dispatcher, reduceMem.localdst, reduceMem.localsrc, stream);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[Reducer][Run]memcpy_async localSrc[%p] localDst[%p] failed", reduceMem.localsrc.ptr(),
                    reduceMem.localdst.ptr()),
                    ret);
            }
        }
    } else {
        CHK_RET(link->RxAsync(rxMems, stream));
        if (link->GetSupportDataReceivedAck()) {
            CHK_RET(link->DataReceivedAck(stream));
        }

        for (RxWithReduceMemoryInfo rxReduceMem : rxWithReduceMems) {
            CHK_RET(HcclReduceAsync(dispatcher, rxReduceMem.reduceSrc, rxReduceMem.reduceDataCount, dataType_,
                reductionOp_, stream, rxReduceMem.reduceDst, INVALID_VALUE_RANKID, LinkType::LINK_ONCHIP,
                reduceAttribute_));
        }
    }

    return HCCL_SUCCESS;
}
} // namespace hccl