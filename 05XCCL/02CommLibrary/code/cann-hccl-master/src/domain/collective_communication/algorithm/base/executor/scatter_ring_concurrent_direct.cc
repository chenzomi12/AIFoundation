/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "scatter_ring_concurrent_direct.h"

namespace hccl {
ScatterRingConcurrentDirect::ScatterRingConcurrentDirect(const HcclDispatcher dispatcher,
                                                         const HcomCollOpInfo *opInfo, const u32 userRank,
                                                         std::vector<Stream>                             &subStreams,
                                                         const std::vector<std::shared_ptr<LocalNotify>> &mainSignals,
                                                         const std::vector<std::shared_ptr<LocalNotify>> &subSignals,
                                                         const std::vector<u32>                          &ringsOrder,
                                                         const std::vector<Slice> &userMemInputSlices)
    : ExecutorBase(dispatcher), opInfo_(opInfo), userRank_(userRank), subStreams_(subStreams),
      mainSignals_(mainSignals), subSignals_(subSignals), ringsOrder_(ringsOrder),
      userMemInputSlices_(userMemInputSlices)
{
}

ScatterRingConcurrentDirect::~ScatterRingConcurrentDirect()
{
}

// reduce scatter ring direct算法的函数入口
HcclResult ScatterRingConcurrentDirect::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links)
{
    // 基本的检查
    CHK_RET(CheckParameters(rank, rankSize, links));

    // 判断rank_size == 1, 若inputMem_ != outputMem_，才需要搬运
    if (rankSize == 1) {
        CHK_RET(MemcpyByOneRank());
        return HCCL_SUCCESS;
    }
    // 收集邻居信息
    CHK_RET(GetInitializedNeighborLinks(rank, rankSize, links));
    // 填充slice_
    CHK_RET(SetSlices(rank, rankSize));

    // 运行scatter, ring算法
    CHK_RET(RunScatter(rank, rankSize));

    if (barrierSwitchOn_) {
        // 执行barrier，保证数据收发完成
        CHK_RET(ExecuteBarrier(leftLink_, rightLink_));
    }

    HCCL_INFO("ScatterRingConcurrentDirect finished: rank[%u] end", rank);
    return HCCL_SUCCESS;
}

HcclResult ScatterRingConcurrentDirect::CheckParameters(const u32 rank, const u32 rankSize,
                                                        const std::vector<LINK> &links)
{
    CHK_PTR_NULL(opInfo_);
    CHK_RET(CheckConcurrentDirectParameters(rank, rankSize, links));
    // 判断subStreams数量是否正确
    CHK_PRT_RET(subStreams_.size() < 1,
                HCCL_ERROR("[ScatterRingConcurrentDirect] subStreams size[%u] is less than 1", subStreams_.size()),
                HCCL_E_PARA);
    for (auto s : subStreams_) {
        CHK_PTR_NULL(s.ptr());
    }
    // 判断mainSignals数量是否正确
    CHK_PRT_RET(mainSignals_.size() < 1,
                HCCL_ERROR("[ScatterRingConcurrentDirect] mainSignals size[%u] is less than 1", mainSignals_.size()),
                HCCL_E_PARA);
    // 判断subSignals数量是否正确
    CHK_PRT_RET(subSignals_.size() < 1,
                HCCL_ERROR("[ScatterRingConcurrentDirect] subSignals size[%u] is less than 1", subSignals_.size()),
                HCCL_E_PARA);
    // 判断ringsOrder数量是否正确
    CHK_PRT_RET(ringsOrder_.size() != rankSize,
                HCCL_ERROR("[ScatterRingConcurrentDirect] ringsOrder size[%u] is not equal to rank size[%u]",
                           ringsOrder_.size(), rankSize),
                HCCL_E_PARA);
    // 判断userMemInputSlices数量是否正确
    CHK_PRT_RET(userMemInputSlices_.size() != rankSize,
                HCCL_ERROR("[ScatterRingConcurrentDirect] userMemInputSlices size[%u] is not equal to rank size[%u]",
                           userMemInputSlices_.size(), rankSize),
                HCCL_E_PARA);
    HCCL_INFO("ScatterRingConcurrentDirect finished to CheckParameters");
    return HCCL_SUCCESS;
}

HcclResult ScatterRingConcurrentDirect::MemcpyByOneRank()
{
    const Slice &srcSlice = userMemInputSlices_[0];
    const Slice &dstSlice = slices_[0];
    DeviceMem    src      = DeviceMem::create(static_cast<u8 *>(opInfo_->inputAddr) + srcSlice.offset, srcSlice.size);
    DeviceMem    dst;
    if (opInfo_->outputAddr != nullptr) {
        // opInfo_->outputAddr != nullptr指示要将输出发送至user output
        u64 stepOffset = slices_[ringsOrder_[0]].offset;
        HCCL_DEBUG("Memcpy operation: stream[main], rank[%u] starts to rcv offset[%llu], size[%llu] at userMemOut_",
                   userRank_, stepOffset, dstSlice.size);
        dst = DeviceMem::create(static_cast<u8 *>(opInfo_->outputAddr) + stepOffset, dstSlice.size);
    } else {
        // opInfo_->outputAddr == nullptr指示要将输出发送至CCL buffer
        HCCL_DEBUG("Memcpy operation: stream[main], rank[%u] starts to rcv offset[%llu], size[%llu] at outputMem_",
                   userRank_, dstSlice.offset, dstSlice.size);
        dst = outputMem_.range(dstSlice.offset, dstSlice.size);
    }
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));
    return HCCL_SUCCESS;
}

HcclResult ScatterRingConcurrentDirect::GetInitializedNeighborLinks(const u32 rank, const u32 rankSize,
                                                                    const std::vector<LINK> &links)
{
    // 收集左邻居信息
    leftLink_ = links[(rank + rankSize - 1) % rankSize];
    CHK_SMART_PTR_NULL(leftLink_);

    // 收集右邻居信息
    rightLink_ = links[(rank + 1) % rankSize];
    CHK_SMART_PTR_NULL(rightLink_);
    HCCL_INFO("ScatterRingConcurrentDirect finished to GetInitializedNeighborLinks");
    return HCCL_SUCCESS;
}

HcclResult ScatterRingConcurrentDirect::SetSlices(const u32 rank, const u32 rankSize)
{
    if (slices_.size() == 0) {
        slices_.resize(rankSize);

        // 生成std::vector<Slice> slices_
        u64 sliceSize = count_ * SIZE_TABLE[dataType_];
        ;

        for (u32 i = 0; i < rankSize; i++) {
            slices_[i].size = sliceSize;
            // 用于DMA消减过程中，消除src与dst不对位的风险
            slices_[i].offset = RoundUpWithDivisor(i * sliceSize, HCCL_MIN_SLICE_ALIGN);

            HCCL_DEBUG("rank[%u], slices[%u].offset=[%llu], slices[%u].size=[%llu]", rank, i, slices_[i].offset, i,
                       slices_[i].size);
        }
    }
    for (u32 i = 0; i < slices_.size(); i++) {
        HCCL_DEBUG("[ScatterRingConcurrentDirect][SetSlices]rank[%u], slices[%u].offset=[%llu], slices[%u].size=[%llu]",
                   rank, i, slices_[i].offset, i, slices_[i].size);
    }
    // 最后一步搬到userMemOut_的offset, 不同的ring环offset不一样
    lastStepOffset_ = slices_[ringsOrder_[0]].offset;
    HCCL_INFO("ScatterRingConcurrentDirect finished to SetSlices");
    return HCCL_SUCCESS;
}

HcclResult ScatterRingConcurrentDirect::RunInitStep(const u32 rank, const u32 rankSize)
{
    // 例如rank[0,1,2,3]中，rank0的rxSliceIdx = 2，txSliceIdx = 3
    u32 initSlice0Idx = 0;
    initSlice0Idx     = (rank + rankSize - 1) % rankSize;
    // 第-1步，片内将部分数据从userIn搬到cclIn
    if (rank == root_) {
        CHK_RET(MainRecordSub()); // 主流通知从流开始通信
        CHK_RET(SubWaitMain());   // 从流等待主流通知
        const Slice &srcInitSlice0 = userMemInputSlices_[initSlice0Idx];
        DeviceMem    srcInit
            = DeviceMem::create(static_cast<u8 *>(opInfo_->inputAddr) + srcInitSlice0.offset, srcInitSlice0.size);
        const Slice &dstInitSlice0 = slices_[initSlice0Idx];
        DeviceMem    dstInit       = inputMem_.range(dstInitSlice0.offset, dstInitSlice0.size);
        HCCL_DEBUG("Memcpy operation: step[-1] stream[sub] src rank[%u] starts to copy(rcv) offset[%llu], size[%llu] "
                   "on userMemInput to offset[%llu], size[%llu] on CCL",
                   userRank_, srcInitSlice0.offset, srcInitSlice0.size, dstInitSlice0.offset, dstInitSlice0.size);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstInit, srcInit, subStreams_[0]));
        CHK_RET(SubRecordMain()); // 从流通知主流通信完成
        CHK_RET(MainWaitSub());   // 主流等待从流通知
    }
    return HCCL_SUCCESS;
}

HcclResult ScatterRingConcurrentDirect::RunMainStream(const u32 stepsFromRank2Root, const u32 step,
                                                      const Slice &txSlice, const Slice &rxSlice, const u32 rankSize)
{
    bool needSend    = stepsFromRank2Root <= step;
    bool needReceive = stepsFromRank2Root > 0 && stepsFromRank2Root <= (step + 1);
    // Ack
    if (needReceive) {
        CHK_RET(leftLink_->TxAck(stream_));
    }
    if (needSend) {
        CHK_RET(rightLink_->RxAck(stream_));
    }

    DeviceMem src;
    // 不同的rank会在不同的step开始持续发送操作，距离root节点越近，越早step开始发送操作
    if (needSend) {
        src = inputMem_.range(txSlice.offset, txSlice.size);
        CHK_RET(rightLink_->TxAsync(UserMemType::INPUT_MEM, txSlice.offset + baseOffset_, src.ptr(), txSlice.size,
                                    stream_));
    }
    // 不同的rank会在不同的step开始持续发送操作，距离root节点越近，越早step开始发送操作
    DeviceMem dst;
    if (needReceive) {
        HCCL_DEBUG("MemcpyAsync operation: step[%u] stream[main], src rank[%u] starts to send offset[%llu] size[%llu] "
                   "from leftMem_",
                   step, leftLink_->GetRemoteRank(), rxSlice.offset, rxSlice.size);
        if (step == rankSize - DMA_REDUCE_TWO_OFFSET && opInfo_->outputAddr != nullptr) {
            HCCL_DEBUG("MemcpyAsync operation: step[%u] stream[main], dst rank[%u] starts to rcv offset[%llu], "
                       "size[%llu] "
                       "at userMemOut_",
                       step, userRank_, lastStepOffset_, rxSlice.size);
            dst = DeviceMem::create(static_cast<u8 *>(opInfo_->outputAddr) + lastStepOffset_, rxSlice.size);
        } else {
            HCCL_DEBUG("MemcpyAsync operation: step[%u] stream[main], dst rank[%u] starts to rcv offset[%llu], "
                       "size[%llu] "
                       "at inputMem_",
                       step, userRank_, rxSlice.offset, rxSlice.size);
            dst = inputMem_.range(rxSlice.offset, rxSlice.size);
        }
        CHK_RET(
            leftLink_->RxAsync(UserMemType::INPUT_MEM, rxSlice.offset + baseOffset_, dst.ptr(), rxSlice.size, stream_));
    }
    return HCCL_SUCCESS;
}

HcclResult ScatterRingConcurrentDirect::RunSubStream(const u32 step, const Slice &subSlice, const Slice &cclSlice,
                                                     const u32 rank, const u32 rankSize)
{
    if (rank == root_) {
        HCCL_DEBUG("Memcpy operation: step[%u] stream[sub], src rank[%u] starts to send offset[%llu], size[%llu] "
                   "from userMemIn_",
                   step, userRank_, subSlice.offset, subSlice.size);
        DeviceMem src = DeviceMem::create(static_cast<u8 *>(opInfo_->inputAddr) + subSlice.offset, subSlice.size);
        DeviceMem dst;
        if (step == rankSize - DMA_REDUCE_TWO_OFFSET && opInfo_->outputAddr != nullptr) {
            HCCL_DEBUG("Memcpy operation: step[%u] stream[sub], dst rank[%u] starts to rcv offset[%llu], size[%llu] "
                       "to userMemOut_",
                       step, userRank_, lastStepOffset_, subSlice.size);
            dst = DeviceMem::create(static_cast<u8 *>(opInfo_->outputAddr) + lastStepOffset_, subSlice.size);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, subStreams_[0]));
        } else {
            HCCL_DEBUG("Memcpy operation: step[%u] stream[sub], dst rank[%u] starts to rcv offset[%llu], size[%llu] "
                       "to inputMem_",
                       step, userRank_, cclSlice.offset, cclSlice.size);
            dst = inputMem_.range(cclSlice.offset, cclSlice.size);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, subStreams_[0]));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ScatterRingConcurrentDirect::RunScatter(const u32 rank, const u32 rankSize)
{
    HCCL_INFO("ScatterRingConcurrentDirect starts, the input param rank[%u]", rank);
    // 空拷贝用于后续操作附着
    CHK_RET(ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));

    CHK_RET(RunInitStep(rank, rankSize));

    // 例如rank[0,1,2,3]中，rank0的rxSliceIdx = 2，txSliceIdx = 3, subSliceIdx = 1
    u32 txSliceIdx  = (rank + rankSize - 1) % rankSize;
    u32 rxSliceIdx  = (rank + rankSize - DMA_REDUCE_TWO_OFFSET) % rankSize;
    u32 subSliceIdx = (rank + rankSize - DMA_REDUCE_TWO_OFFSET) % rankSize; // 只存在于根节点

    u32 stepsFromRank2Root = (rank + rankSize - root_) % rankSize;
    for (u32 step = 0; step < rankSize - 1; step++) {
        const Slice &subSlice = userMemInputSlices_[subSliceIdx];
        const Slice &cclSlice = slices_[subSliceIdx];
        const Slice &txSlice  = slices_[txSliceIdx];
        const Slice &rxSlice  = slices_[rxSliceIdx];

        // 并发
        CHK_RET(MainRecordSub()); // 主流通知从流开始通信
        CHK_RET(SubWaitMain());   // 从流等待主流通知

        // 空拷贝用于主从流任务并发
        CHK_RET(ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));
        CHK_RET(ExecutorBase::ExecEmptyTask(inputMem_, outputMem_, subStreams_[0], dispatcher_));

        // 主流
        CHK_RET(RunMainStream(stepsFromRank2Root, step, txSlice, rxSlice, rankSize));

        // 从流
        CHK_RET(RunSubStream(step, subSlice, cclSlice, rank, rankSize));

        CHK_RET(SubRecordMain()); // 从流通知主流通信完成
        CHK_RET(MainWaitSub());   // 主流等待从流通知

        // 更新索引
        subSliceIdx = (subSliceIdx + rankSize - 1) % rankSize;
        txSliceIdx  = (txSliceIdx + rankSize - 1) % rankSize;
        rxSliceIdx  = (rxSliceIdx + rankSize - 1) % rankSize;
    }
    HCCL_INFO("ScatterRingConcurrentDirect finished to RunScatter");
    return HCCL_SUCCESS;
}
// 主流通知从流干活
HcclResult ScatterRingConcurrentDirect::MainRecordSub()
{
    for (u32 signalIndex = 0; signalIndex < subSignals_.size(); signalIndex++) {
        CHK_RET(LocalNotify::Post(stream_, dispatcher_, subSignals_[signalIndex],
            profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}
// 从流等待主流
HcclResult ScatterRingConcurrentDirect::SubWaitMain()
{
    for (u32 streamIndex = 0; streamIndex < subSignals_.size(); streamIndex++) {
        CHK_RET(LocalNotify::Wait(subStreams_[streamIndex], dispatcher_, subSignals_[streamIndex],
            profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}
// 主流等待从流
HcclResult ScatterRingConcurrentDirect::MainWaitSub()
{
    for (u32 signalIndex = 0; signalIndex < mainSignals_.size(); signalIndex++) {
        CHK_RET(LocalNotify::Wait(stream_, dispatcher_, mainSignals_[signalIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}
// 从流告诉主流活干完了
HcclResult ScatterRingConcurrentDirect::SubRecordMain()
{
    for (u32 streamIndex = 0; streamIndex < mainSignals_.size(); streamIndex++) {
        CHK_RET(LocalNotify::Post(subStreams_[streamIndex], dispatcher_, mainSignals_[streamIndex],
            profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}
} // namespace hccl
