/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gather_ring.h"

namespace hccl {
GatherRing::GatherRing(const HcclDispatcher dispatcher)
    : ExecutorBase(dispatcher), interRank_(0), interRankSize_(0)
{
}

GatherRing::~GatherRing()
{
}
// 不带入slice时候，计算gather 输出buffer的偏移
void GatherRing::PrepareSlicesData(const u32 unitSize, const u64 totalCount, const u32 rankSize)
{
    slices_.resize(rankSize);
    u64 sliceSize = totalCount * unitSize;

    for (u32 i = 0; i < rankSize; i++) {
        slices_[i].offset = i * sliceSize;
        slices_[i].size = sliceSize;
        HCCL_DEBUG("rank[%u] default slice[%u]: offset: [%llu] size[%llu]", interRank_, i, i * sliceSize, sliceSize);
    }
}
// root rank只接收数据，需要接收多次
HcclResult GatherRing::RunGatherOnRootRank()
{
    HcclResult ret;
    DeviceMem dst;

    u32 round = interRankSize_ - 1;
    // root第一轮从前一rank的input接收
    for (u32 i = 1; i <= round; i++) {
        u32 rcvSlice = (interRank_ - i + interRankSize_) % interRankSize_;
        u64 rcvOffset = slices_[rcvSlice].offset;
        u64 rcvSize = slices_[rcvSlice].size;

        // 给前一节点发送同步，以便前一rank进行下一轮的操作
        ret = linkLeft_->TxAck(stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Run][GatherOnRootRank]rootrank[%u] tx ack failed", interRank_), ret);
        dst = outputMem_.range(rcvOffset, rcvSize);

        ret = linkLeft_->RxAsync(UserMemType::OUTPUT_MEM, rcvOffset, dst.ptr(), rcvSize, stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][GatherOnRootRank]root rank[%u] rx sync with dstmem[%p] "\
            "failed", interRank_, dst.ptr()), ret);

        HCCL_INFO("GatherRing rootrank[%u] round[%u] rx data ouputoffset[%llu] size[%llu]", \
            interRank_, i, rcvOffset, rcvSize);
    }
    return HCCL_SUCCESS;
}
// root的后一节点。只是进行发送
HcclResult GatherRing::RunGatherOnRootNextRank()
{
    HcclResult ret = HCCL_SUCCESS;
    // 发送自己的input数据向root rank
    u64 sendOffset = slices_[interRank_].offset;
    u64 sendSize = slices_[interRank_].size;
    ret = linkRight_->RxAck(stream_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][GatherOnRootNextRank]rank[%u] rx ack failed", interRank_), ret);

    DeviceMem sendMem = outputMem_.range(sendOffset, sendSize);
    ret = linkRight_->TxAsync(UserMemType::OUTPUT_MEM, sendOffset, sendMem.ptr(), sendSize, stream_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][GatherOnRootNextRank]rank[%u] TxAsync offset[%llu] size[%llu] "\
        "failed", interRank_, sendOffset, sendSize), ret);
    HCCL_DEBUG("GatherRing root next rank[%u] tx async offset[%llu] size[%llu]",
        interRank_, sendOffset, sendSize);
    ret = linkRight_->TxWaitDone(stream_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][GatherOnRootNextRank]TxWaitDone failed"), ret);

    return ret;
}
// 非root，非root后一节点rank。先发送自己数据，再接收
HcclResult GatherRing::RunGatherOnOtherRank()
{
    DeviceMem src;
    DeviceMem dst;
    HcclResult ret = HCCL_SUCCESS;
    u64 sendOffset = slices_[interRank_].offset;
    u64 sendSize = slices_[interRank_].size;

    // 把自己数据发送至下一rank
    ret = linkLeft_->TxAck(stream_);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Run][GatherOnOtherRank]rank[%u] tx ack failed", interRank_), ret);
    ret = linkRight_->RxAck(stream_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][GatherOnOtherRank]rank[%u] rx ack failed", interRank_), ret);
    DeviceMem sendMem = outputMem_.range(sendOffset, sendSize);
    ret = linkRight_->TxAsync(UserMemType::OUTPUT_MEM, sendOffset, sendMem.ptr(), sendSize, stream_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][GatherOnOtherRank]rank[%u] TxAsync offset[%llu] size[%llu] "\
        "failed", interRank_, sendOffset, sendSize), ret);

    u32 round = (interRank_ + interRankSize_ - root_ - 1) % interRankSize_;
    // 需要接收的和发送的轮数，包含接收自己的数据
    for (u32 i = 1; i < round; i++) {
        u32 rcvRank = (interRank_ + interRankSize_ - i) % interRankSize_; // 收到的数据应当是哪个rank的slice
        u64 rcvOffset = slices_[rcvRank].offset;
        u64 rcvSize = slices_[rcvRank].size;
        // 使用本端oupt接收，第一轮从对端的input，否则从对端的output
        dst = outputMem_.range(rcvOffset, rcvSize);
        // 从前一个节点收数据
        ret = linkLeft_->RxAsync(UserMemType::OUTPUT_MEM, rcvOffset, dst.ptr(), rcvSize, stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Run][GatherOnOtherRank]rank[%u] rx async failed", interRank_), ret);

        ret = linkLeft_->RxWaitDone(stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][GatherOnOtherRank]RxWaitDone failed"), ret);
        ret = linkRight_->TxWaitDone(stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][GatherOnOtherRank]TxWaitDone failed"), ret);

        ret = linkLeft_->TxAck(stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][GatherOnOtherRank]rank[%u] round[%u] tx ack  failed",
            interRank_, i), ret);

        // 从后一rank接收同步信号
        ret = linkRight_->RxAck(stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][GatherOnOtherRank]rank[%u] round[%u] rx ack failed",
            interRank_, i), ret);
        // 向后一rank发送数据
        ret = linkRight_->TxAsync(UserMemType::OUTPUT_MEM, rcvOffset, dst.ptr(), rcvSize, stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][GatherOnOtherRank]rank[%u] round[%u] tx async failed",
            interRank_, i), ret);
        HCCL_DEBUG("GatherRing rank[%u] round[%u] tx async offset[%llu] size[%llu]",
            interRank_, i, rcvOffset, rcvSize);
    }
    // 最后一次接收后，再发送，与for循环内动作相比 缺少leftlink->txack,最后一轮收的一定是root next的数据,root的next next节点不走for循环
    u32 rcvSlice = (root_ + 1) % interRankSize_;
    u64 rcvOffset = slices_[rcvSlice].offset;
    u64 rcvSize = slices_[rcvSlice].size;
    dst = outputMem_.range(rcvOffset, rcvSize);

    // 从前一个节点收数据
    ret = linkLeft_->RxAsync(UserMemType::OUTPUT_MEM, rcvOffset, dst.ptr(), rcvSize, stream_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][GatherOnOtherRank]rank[%u] rx async failed", interRank_), ret);
    ret = linkLeft_->RxWaitDone(stream_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][GatherOnOtherRank]RxWaitDone failed"), ret);
    ret = linkRight_->TxWaitDone(stream_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][GatherOnOtherRank]TxWaitDone failed"), ret);
    // 从后一rank接收同步信号
    ret = linkRight_->RxAck(stream_);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Run][GatherOnOtherRank]rank[%u] rx ack failed", interRank_), ret);
    // 向后一rank发送数据
    ret = linkRight_->TxAsync(UserMemType::OUTPUT_MEM, rcvOffset, dst.ptr(), rcvSize, stream_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][GatherOnOtherRank]rank[%u] tx async failed", interRank_), ret);
    ret = linkRight_->TxWaitDone(stream_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][GatherOnOtherRank]TxWaitDone failed"), ret);
    return HCCL_SUCCESS;
}
// gather的入口函数
HcclResult GatherRing::RunAsync(const u32 rank, const u32 rankSize,
    const std::vector<std::shared_ptr<Transport> > &links)
{
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    if (!outputMem_ || !inputMem_) {
        HCCL_ERROR("[GatherRing][RunAsync]run_async inputmem or outputmem is null");
        return HCCL_E_PTR;
    }

    interRank_ = rank;
    interRankSize_ = rankSize;

    HCCL_INFO("GatherRing run: rank[%u] totalrank[%u] count[%llu] input[%p] output[%p]",
              interRank_, interRankSize_,  count_, inputMem_.ptr(), outputMem_.ptr());

    // ranksize为1时，只有当input!=ouput 时候进行拷贝
    if (interRankSize_ == 1) {
        if (inputMem_ != outputMem_) {
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_));
        }
        return HCCL_SUCCESS;
    }

    u32 unitSize = DataUnitSize(dataType_);
    CHK_PRT_RET(unitSize == 0, HCCL_ERROR("[GatherRing][RunAsync]rank[%u] unit data size is zero", rank),
        HCCL_E_INTERNAL);

    // 带入vecotr为空，计算每个rank的结果偏移和大小
    if (slices_.size() == 0) {
        PrepareSlicesData(unitSize, count_, interRankSize_);
    }

    u32 ringPrevRank = (rank + rankSize - 1) % rankSize;
    u32 ringNextRank = (rank + 1) % rankSize;

    if (links.size() < rankSize) {
        HCCL_ERROR("[GatherRing][RunAsync]rank[%u] link size[%llu] is less than rank size", rank, links.size());
        return HCCL_E_INTERNAL;
    }

    linkLeft_ = links[ringPrevRank];
    CHK_SMART_PTR_NULL(linkLeft_);

    linkRight_ = links[ringNextRank];
    CHK_SMART_PTR_NULL(linkRight_);

    if (interRank_ == root_) {
        CHK_RET(RunGatherOnRootRank());
    } else if (ringPrevRank == root_) {
        CHK_RET(RunGatherOnRootNextRank());
    } else {
        CHK_RET(RunGatherOnOtherRank());
    }

    if (barrierSwitchOn_) {
        // 执行barrier，保证数据收发完成
        CHK_RET(ExecuteBarrier(linkLeft_, linkRight_));
    }

    HCCL_INFO("GatherRing finished: rank:[%u] end", interRank_);

    return HCCL_SUCCESS;
}
}  // namespace hccl
