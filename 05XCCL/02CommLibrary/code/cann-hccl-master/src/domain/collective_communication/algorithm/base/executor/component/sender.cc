/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sender.h"

namespace hccl {
Sender::Sender(const HcclDataType dataType, const HcclReduceOp reductionOp, const u64 reduceAttribute)
    : dataType_(dataType), reductionOp_(reductionOp), reduceAttribute_(reduceAttribute)
{
}

Sender::~Sender()
{
}

HcclResult Sender::run(const std::shared_ptr<Transport> &link, const u64 dstOffset, DeviceMem &src,
                       Stream &stream, const UserMemType dstMemType) const
{
    // server 内通信并且 reduceAttribute_ 也支持，走该分支
    bool isSpInlineReduce = link->IsSpInlineReduce();
    // 溢出检测为：Warning && INF/NAN 模式时, 支持Write With Reduce
    bool isSpRdmaReduce = RDMA_REDUCE_BITMASK & reduceAttribute_;

    if (link->IsSupportTransportWithReduce() && (link->GetLinkType() == LinkType::LINK_STANDARD_ROCE ||
        isSpRdmaReduce)) {
        // 数据发送端执行Write With Reduce操作
        CHK_RET(link->TxWithReduce(dstMemType, dstOffset, src.ptr(), src.size(), dataType_,
            reductionOp_, stream));
    } else if (isSpInlineReduce && (INLINE_REDUCE_BITMASK & reduceAttribute_)) {
        // link支持inline reduce 并且 reduceAttribute_ 也支持
        // notify 下一个rank做 inline reduce
        CHK_RET(link->TxDataSignal(stream));
    } else {
        // 向下一个节点发送数据
        CHK_RET(link->TxAsync(UserMemType::OUTPUT_MEM, dstOffset, src.ptr(), src.size(), stream));
    }

    return HCCL_SUCCESS;
}

HcclResult Sender::run(const std::shared_ptr<Transport> &link, const std::vector<SenderMemoryInfo> &senderMems,
    Stream &stream) const
{
    LinkType linkType = link->GetLinkType();
    bool isSpInlineReduce = link->IsSpInlineReduce();
    bool isSpRdmaReduce = RDMA_REDUCE_BITMASK & reduceAttribute_;
    bool isSpTransportWithReduce = link->IsSupportTransportWithReduce();

    std::vector<TxMemoryInfo> txMems;
    for (const SenderMemoryInfo& senderMem : senderMems) {
        txMems.emplace_back(TxMemoryInfo{UserMemType::INPUT_MEM, senderMem.dstOffset,
            senderMem.src.ptr(), senderMem.src.size()});
    }

    if (isSpTransportWithReduce && (linkType == LinkType::LINK_STANDARD_ROCE || isSpRdmaReduce)) {
        CHK_RET(link->TxWithReduce(txMems, dataType_, reductionOp_, stream));
    } else if (isSpInlineReduce && (INLINE_REDUCE_BITMASK & reduceAttribute_)) {
        // link支持inline reduce 并且 reduceAttribute_ 也支持
        // notify 下一个rank做 inline reduce
        CHK_RET(link->TxDataSignal(stream));
    } else {
        for (TxMemoryInfo& txMem : txMems) {
            txMem.dstMemType = UserMemType::OUTPUT_MEM;
        }
        CHK_RET(link->TxAsync(txMems, stream));
    }

    return HCCL_SUCCESS;
}

}  // namespace hccl