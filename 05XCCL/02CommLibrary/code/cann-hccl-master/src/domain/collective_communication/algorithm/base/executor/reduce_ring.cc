/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_ring.h"

namespace hccl {
ReduceRing::ReduceRing(const HcclDispatcher dispatcher, const u64 reduceAttrBitMap)
    : ExecutorBase(dispatcher), reduceAttr_(reduceAttrBitMap)
{
}

ReduceRing::~ReduceRing()
{
}

// reduce算法的入口函数
HcclResult ReduceRing::RunAsync(const u32 rank, const u32 rankSize,
    const std::vector<std::shared_ptr<Transport> > &links)
{
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    bool bRetNull = (!outputMem_ || !inputMem_);
    CHK_PRT_RET(bRetNull, HCCL_ERROR("[ReduceRing][RunAsync]rank[%u] inputmem or outputmem is null", rank),
        HCCL_E_PARA);

    HcclResult ret = HCCL_SUCCESS;
    HCCL_INFO("ReduceRing run: rank[%u] totalrank[%u] root[%u] inputmem[%p] output[%p] count[%llu]", \
        rank, rankSize, root_, inputMem_.ptr(), outputMem_.ptr(), count_);

    // 如果ranksize为1, inline reduce和普通跨片reduce操作一致，从input->output
    if (rankSize == 1) {
        if (inputMem_ != outputMem_) {
            ret = HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_);
        }
        return ret;
    }

    // 创建reducer & sender
    senderInfo_.reset(new (std::nothrow) Sender(dataType_, reductionOp_, reduceAttr_));
    CHK_SMART_PTR_NULL(senderInfo_);

    reducerInfo_.reset(new (std::nothrow) Reducer(dataType_, reductionOp_, reduceAttr_));
    CHK_SMART_PTR_NULL(reducerInfo_);

    // 获取ring algorithm所需的通信连接
    u32 ringPrevRank = (rank + rankSize - 1) % rankSize;
    u32 ringNextRank = (rank + 1) % rankSize;

    if (links.size() < rankSize) {
        HCCL_ERROR("[ReduceRing][RunAsync]rank[%u] Link size is less than rank size", rank);
        return HCCL_E_INTERNAL;
    }

    linkLeft_ = links[ringPrevRank];
    CHK_SMART_PTR_NULL(linkLeft_);

    linkRight_ = links[ringNextRank];
    CHK_SMART_PTR_NULL(linkRight_);

    scratch_ = DeviceMem::create(inputMem_.ptr(), inputMem_.size());

    u32 dataSize = DataUnitSize(dataType_);
    if (dataSize == 0) {
        HCCL_ERROR("[ReduceRing][RunAsync]rank[%u] unit data size is zero", rank);
        return HCCL_E_INTERNAL;
    }

    // 计算以chunk_size_为最大处理单元时，能够处理的最大数据个数
    // 每轮需要操作的数据个数
    CHK_RET(linkLeft_->TxAck(stream_));
    CHK_RET(linkRight_->RxAck(stream_));
    u64 length = count_ * dataSize;

    if (rank == root_) {
        // root节点只接收数据
        DeviceMem localSrc = scratch_.range(0, length);
        DeviceMem dst = outputMem_.range(0, length);
        HCCL_DEBUG("rank [%u] recv data offset[%llu] size[%llu] reduce", rank, 0, length);

        // 需要从前一节点接收数据,替换reducer接口
        ret = reducerInfo_->run(dispatcher_, linkLeft_, baseOffset_, localSrc, dst, dst,
            stream_, DstMemType::RESULT_OUTPUT_MEM);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[ReduceRing][RunAsync]rank[%u] reduce data offset[%llu] size[%llu]", rank, 0, length), ret);

        // 给前一节点发送同步
        CHK_RET(linkLeft_->TxAck(stream_));
        CHK_RET(linkLeft_->RxWaitDone(stream_));
    } else if (ringPrevRank == root_) {
        // 本rank的前一节点是root节点，本rank数据拷贝到下一rank，不做reduce操作
        // 需要向下一节点拷贝的数据
        DeviceMem localSrc = scratch_.range(0, length);

        // 数据拷贝和向下一节点发送
        HCCL_DEBUG("rank [%u] send offset[%llu] size[%llu]", rank, 0, length);

        ret = senderInfo_->run(linkRight_, baseOffset_, localSrc, stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[ReduceRing][RunAsync]rank[%u] send scratch offset[%llu] size[%llu] "\
            "failed", rank, baseOffset_, length), ret);

        // 等待后一节点同步信号
        CHK_RET(linkRight_->RxAck(stream_));
        CHK_RET(linkRight_->TxWaitDone(stream_));
    } else {
        // 其余节点，先接收数据，和自身数据进行reduce操作，结果放入tx中，发送至下一节点
        // 剩余需要处理的数据大于满chunk size时，以chunksize为处理单位，否则直接处理剩余数据

        // 接收到的数据和scratch数据运算后，放入output
        DeviceMem localSrc = scratch_.range(0, length);
        DeviceMem dst = outputMem_.range(0, length);

        // 用reduce接口封装
        HCCL_DEBUG("rank[%u] recv data reduce offset[%llu] size[%llu]", rank, 0, length);

        ret = reducerInfo_->run(dispatcher_, linkLeft_, baseOffset_, localSrc, localSrc, dst,
            stream_, DstMemType::RESULT_INPUT_MEM);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[ReduceRing][RunAsync]rank[%u] reducer offset[%llu] size[%llu] failed", rank, 0, length), ret);

        // 给前一节点发送同步
        CHK_RET(linkLeft_->TxAck(stream_));
        CHK_RET(linkLeft_->RxWaitDone(stream_));

        // tx数据向下一个节点发送
        // 需要再封装接口，只把数据发到tx_mem,send_only
        HCCL_DEBUG("rank[%u] send localSrc offset[%llu] size[%llu]", rank, 0, length);

        ret = senderInfo_->run(linkRight_, baseOffset_, localSrc, stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[ReduceRing][RunAsync]rank[%u] sender offset[%llu] failed", rank, baseOffset_), ret);

        // 等待后一节点同步信号
        CHK_RET(linkRight_->RxAck(stream_));
        CHK_RET(linkRight_->TxWaitDone(stream_));
    }
    CHK_RET(linkRight_->TxDataSignal(stream_));
    CHK_RET(linkLeft_->RxDataSignal(stream_));
    HCCL_INFO("ReduceRing finished: rank[%u] end", rank);
    return HCCL_SUCCESS;
}
}  // namespace hccl
