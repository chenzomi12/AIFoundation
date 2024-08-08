/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_ring_concurrent_direct.h"

namespace hccl {
ReduceScatterRingConcurrentDirect::ReduceScatterRingConcurrentDirect(
    const HcclDispatcher dispatcher, const u64 reduceAttrBitMap, const HcomCollOpInfo *opInfo,
    const u32 userRank, std::vector<Stream> &subStreams, const std::vector<std::shared_ptr<LocalNotify>> &mainSignals,
    const std::vector<std::shared_ptr<LocalNotify>> &subSignals, const std::vector<u32> &ringsOrder,
    const std::vector<Slice> &userMemInputSlices)
    : ExecutorBase(dispatcher), reduceAttr_(reduceAttrBitMap), opInfo_(opInfo), userRank_(userRank),
      subStreams_(subStreams), mainSignals_(mainSignals), subSignals_(subSignals), ringsOrder_(ringsOrder),
      userMemInputSlices_(userMemInputSlices)
{
}

ReduceScatterRingConcurrentDirect::~ReduceScatterRingConcurrentDirect()
{
}

// reduce scatter ring direct算法的函数入口
HcclResult ReduceScatterRingConcurrentDirect::RunAsync(const u32 rank, const u32 rankSize,
                                                       const std::vector<LINK> &links)
{
    // 基本的检查
    CHK_RET(CheckParameters(rank, rankSize, links));

    // 判断rank_size == 1的情况，并拷贝
    if (rankSize == 1) {
        CHK_RET(MemcpyByOneRank());
        return HCCL_SUCCESS;
    }
    // 收集本地mem信息
    CHK_RET(InitSenderReducer());

    // 收集邻居信息
    CHK_RET(GetInitializedNeighborLinks(rank, rankSize, links));

    // 填充slice_
    CHK_RET(SetSlices(rank, rankSize));

    // 运行reduce-scatter, ring算法
    CHK_RET(RunReduceScatter(rank, rankSize));

    if (barrierSwitchOn_) {
        // 执行barrier，保证数据收发完成
        CHK_RET(ExecuteBarrier(leftLink_, rightLink_));
    }

    HCCL_INFO("ReduceScatterRingConcurrentDirect finished: rank[%u] end", rank);
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterRingConcurrentDirect::CheckParameters(const u32 rank, const u32 rankSize,
                                                              const std::vector<LINK> &links)
{
    CHK_PTR_NULL(opInfo_);
    CHK_RET(CheckConcurrentDirectParameters(rank, rankSize, links));
    // 判断subStreams数量是否正确
    CHK_PRT_RET(
        subStreams_.size() < 1,
        HCCL_ERROR("[ReduceScatterRingConcurrentDirect] subStreams size[%u] is less than 1", subStreams_.size()),
        HCCL_E_PARA);
    for (auto s : subStreams_) {
        CHK_PTR_NULL(s.ptr());
    }
    // 判断mainSignals数量是否正确
    CHK_PRT_RET(
        mainSignals_.size() < 1,
        HCCL_ERROR("[ReduceScatterRingConcurrentDirect] mainSignals size[%u] is less than 1", mainSignals_.size()),
        HCCL_E_PARA);
    // 判断subSignals数量是否正确
    CHK_PRT_RET(
        subSignals_.size() < 1,
        HCCL_ERROR("[ReduceScatterRingConcurrentDirect] subSignals size[%u] is less than 1", subSignals_.size()),
        HCCL_E_PARA);
    // 判断ringsOrder数量是否正确
    CHK_PRT_RET(ringsOrder_.size() != rankSize,
                HCCL_ERROR("[ReduceScatterRingConcurrentDirect] ringsOrder size[%u] is not equal to rank size[%u]",
                           ringsOrder_.size(), rankSize),
                HCCL_E_PARA);
    // 判断userMemInputSlices数量是否正确
    CHK_PRT_RET(userMemInputSlices_.size() % rankSize != 0,
        HCCL_ERROR("[ReduceScatterRingConcurrentDirect] userMemInputSlices size[%u] can not divided by size[%u]",
                   userMemInputSlices_.size(), rankSize),
        HCCL_E_PARA);
    HCCL_INFO("ReduceScatterRingConcurrentDirect finished to CheckParameters");
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterRingConcurrentDirect::MemcpyByOneRank()
{
    for (u32 sliceIdx = 0; sliceIdx < slices_.size(); sliceIdx++) {
        const Slice &srcSlice = userMemInputSlices_[sliceIdx];
        const Slice &dstSlice = slices_[sliceIdx];
        DeviceMem src = DeviceMem::create(static_cast<u8 *>(opInfo_->inputAddr) + srcSlice.offset, srcSlice.size);
        DeviceMem dst;
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
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterRingConcurrentDirect::InitSenderReducer()
{
    // 创建reducer & sender
    senderInfo_.reset(new (std::nothrow) Sender(dataType_, reductionOp_, reduceAttr_));
    CHK_SMART_PTR_NULL(senderInfo_);

    reducerInfo_.reset(new (std::nothrow) Reducer(dataType_, reductionOp_, reduceAttr_));
    CHK_SMART_PTR_NULL(reducerInfo_);
    HCCL_INFO("ReduceScatterRingConcurrentDirect finished to InitSenderReducer");
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterRingConcurrentDirect::GetInitializedNeighborLinks(const u32 rank, const u32 rankSize,
                                                                          const std::vector<LINK> &links)
{
    // 收集左邻居信息
    leftLink_ = links[(rank + rankSize - 1) % rankSize];
    CHK_SMART_PTR_NULL(leftLink_);

    // 收集右邻居信息
    rightLink_ = links[(rank + 1) % rankSize];
    CHK_SMART_PTR_NULL(rightLink_);
    HCCL_INFO("ReduceScatterRingConcurrentDirect finished to GetInitializedNeighborLinks");
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterRingConcurrentDirect::SetSlices(const u32 rank, const u32 rankSize)
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
        HCCL_DEBUG(
            "[ReduceScatterRingConcurrentDirect][SetSlices]rank[%u], slices[%u].offset=[%llu], slices[%u].size=[%llu]",
            rank, i, slices_[i].offset, i, slices_[i].size);
    }
    // 最后一步搬到userMemOut_的offset, 不同的ring环offset不一样
    lastStepOffset_ = slices_[ringsOrder_[0]].offset;
    HCCL_INFO("ReduceScatterRingConcurrentDirect finished to SetSlices");
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterRingConcurrentDirect::RunInitStep(const u32 rank, const u32 rankSize)
{
    // 例如rank[0,1,2,3]中，rank0的rxSliceIdx = 2，txSliceIdx = 3
    u32 initSlice0Idx = 0;
    u32 initSlice1Idx = 0;
    initSlice0Idx     = (rank + rankSize - 1) % rankSize;
    initSlice1Idx     = (rank + rankSize - DMA_REDUCE_TWO_OFFSET) % rankSize;
    u32 sliceSize = slices_.size() / rankSize;

    for (u32 sliceIdx = 0; sliceIdx < sliceSize; sliceIdx++) {
        // 第-1步，片内将部分数据从userIn搬到cclIn
        const Slice &srcInitSlice0 = userMemInputSlices_[initSlice0Idx * sliceSize + sliceIdx];
        DeviceMem    srcSubInit
            = DeviceMem::create(static_cast<u8 *>(opInfo_->inputAddr) + srcInitSlice0.offset, srcInitSlice0.size);
        const Slice &dstInitSlice0 = slices_[initSlice0Idx * sliceSize + sliceIdx];
        DeviceMem    dstSubInit    = inputMem_.range(dstInitSlice0.offset, dstInitSlice0.size);
        const Slice &srcInitSlice1 = userMemInputSlices_[initSlice1Idx * sliceSize + sliceIdx];
        DeviceMem    srcInit
            = DeviceMem::create(static_cast<u8 *>(opInfo_->inputAddr) + srcInitSlice1.offset, srcInitSlice1.size);
        const Slice &dstInitSlice1 = slices_[initSlice1Idx * sliceSize + sliceIdx];
        DeviceMem    dstInit       = inputMem_.range(dstInitSlice1.offset, dstInitSlice1.size);
        // 第-1步并发
        CHK_RET(MainRecordSub()); // 主流通知从流开始通信
        CHK_RET(SubWaitMain());   // 从流等待主流通知
        if (rankSize == TWO_RANK_SIZE && opInfo_->outputAddr != nullptr) {
            HCCL_DEBUG(
                "Memcpy operation: step[-1] stream[main] src rank[%u] starts to copy(rcv) offset[%llu], size[%llu] on "
                "userMemInput to offset[%llu], size[%llu] on userMemOut_",
                userRank_, srcInitSlice1.offset, srcInitSlice1.size, lastStepOffset_, dstInitSlice1.size);
            dstInit = DeviceMem::create(static_cast<u8 *>(opInfo_->outputAddr) + lastStepOffset_, dstInitSlice1.size);
        } else {
            HCCL_DEBUG(
                "Memcpy operation: step[-1] stream[main] src rank[%u] starts to copy(rcv) offset[%llu], size[%llu] on "
                "userMemInput to offset[%llu], size[%llu] on CCL",
                userRank_, srcInitSlice1.offset, srcInitSlice1.size, dstInitSlice1.offset, dstInitSlice1.size);
        }
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstInit, srcInit, stream_));
        HCCL_DEBUG("Memcpy operation: step[-1] stream[sub] src rank[%u] starts to copy(rcv) offset[%llu], "
            " size[%llu] on userMemInput to offset[%llu], size[%llu] on CCL",
            userRank_, srcInitSlice0.offset, srcInitSlice0.size, dstInitSlice0.offset, dstInitSlice0.size);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstSubInit, srcSubInit, subStreams_[0]));
        CHK_RET(SubRecordMain()); // 从流通知主流通信完成
        CHK_RET(MainWaitSub());   // 主流等待从流通知
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterRingConcurrentDirect::RunMainStream(const u32 step, std::vector<Slice> txSliceVector,
    std::vector<Slice> rxSliceVector, const u32 rank, const u32 rankSize)
{
    CHK_RET(leftLink_->TxAck(stream_));
    CHK_RET(rightLink_->RxAck(stream_));

    u32 sliceSize = slices_.size() / rankSize;

    // 通信，如果是最后一步，则做消减拷贝
    std::vector<SenderMemoryInfo> txMems;
    std::vector<ReducerMemoryInfo> rxReduceMems;
    DeviceMem dst;
    for (u32 sliceIdx = 0; sliceIdx < sliceSize; sliceIdx++) {
        // Ack
        HCCL_DEBUG("Reduce operation: step[%u] stream[main], src rank[%u] starts to send offset[%llu] size[%llu] "
            "from leftMem_",
            step, leftLink_->GetRemoteRank(), rxSliceVector[sliceIdx].offset, rxSliceVector[sliceIdx].size);
        if (step == rankSize - DMA_REDUCE_TWO_OFFSET && opInfo_->outputAddr != nullptr) {
            HCCL_DEBUG("Reduce operation: step[%u] stream[main], dst rank[%u] starts to rcv offset[%llu], size[%llu] "
                "at userMemOut_",
                step, userRank_, lastStepOffset_, rxSliceVector[sliceIdx].size);
            dst = DeviceMem::create(static_cast<u8 *>(opInfo_->outputAddr) + lastStepOffset_,
                rxSliceVector[sliceIdx].size);
        } else {
            HCCL_DEBUG("Reduce operation: step[%u] stream[main], dst rank[%u] starts to rcv offset[%llu], size[%llu] "
                "at inputMem_",
                step, userRank_, rxSliceVector[sliceIdx].offset, rxSliceVector[sliceIdx].size);
            dst = inputMem_.range(rxSliceVector[sliceIdx].offset, rxSliceVector[sliceIdx].size);
        }
        // 在inline reduce场景, 需要利用scratchMem_暂存
        DeviceMem srcMemTemp = scratchMem_.range(rxSliceVector[sliceIdx].offset, rxSliceVector[sliceIdx].size);
        DeviceMem srcMem     = inputMem_.range(txSliceVector[sliceIdx].offset, txSliceVector[sliceIdx].size);
        HCCL_DEBUG("Reduce operation: step[%u] stream[main], senderInfo_ rank[%u] starts to rcv offset[%llu], "
            " size[%llu]",
            step, rightLink_->GetRemoteRank(), txSliceVector[sliceIdx].offset, txSliceVector[sliceIdx].size);
        rxReduceMems.emplace_back(ReducerMemoryInfo{baseOffset_ + rxSliceVector[sliceIdx].offset,
            dst, dst, srcMemTemp});
        txMems.emplace_back(SenderMemoryInfo{baseOffset_ + txSliceVector[sliceIdx].offset, srcMem});
    }
    CHK_RET(senderInfo_->run(rightLink_, txMems, stream_));
    CHK_RET(reducerInfo_->run(dispatcher_, leftLink_, rxReduceMems, stream_));
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterRingConcurrentDirect::RunSubStream(const u32 step, std::vector<Slice> subSliceVector,
    std::vector<Slice> cclSliceVector, const u32 rank, const u32 rankSize)
{
    for (u32 sliceIdx = 0; sliceIdx < subSliceVector.size(); sliceIdx++) {
        HCCL_DEBUG("Memcpy operation: step[%u] stream[sub], src rank[%u] starts to send offset[%llu], size[%llu] "
            "from userMemIn_", step, userRank_, subSliceVector[sliceIdx].offset, subSliceVector[sliceIdx].size);
        DeviceMem src = DeviceMem::create(static_cast<u8 *>(opInfo_->inputAddr) + subSliceVector[sliceIdx].offset,
            subSliceVector[sliceIdx].size);
        DeviceMem dst;
        if (step == rankSize - DMA_REDUCE_TWO_OFFSET) {
            // do nothing
        } else if (step == rankSize - DMA_REDUCE_THREE_OFFSET && opInfo_->outputAddr != nullptr) {
            HCCL_DEBUG("Memcpy operation: step[%u] stream[sub], dst rank[%u] starts to rcv offset[%llu], size[%llu] "
                "to userMemOut_", step, userRank_, lastStepOffset_, subSliceVector[sliceIdx].size);
            dst = DeviceMem::create(static_cast<u8 *>(opInfo_->outputAddr) + lastStepOffset_,
                subSliceVector[sliceIdx].size);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, subStreams_[0]));
        } else {
            HCCL_DEBUG("Memcpy operation: step[%u] stream[sub], dst rank[%u] starts to rcv offset[%llu], size[%llu] "
                "to inputMem_",
                step, userRank_, cclSliceVector[sliceIdx].offset, cclSliceVector[sliceIdx].size);
            dst = inputMem_.range(cclSliceVector[sliceIdx].offset, cclSliceVector[sliceIdx].size);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, subStreams_[0]));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterRingConcurrentDirect::RunReduceScatter(const u32 rank, const u32 rankSize)
{
    HCCL_INFO("ReduceScatterRingConcurrentDirect starts, the input param rank[%u]", rank);
    // 空拷贝用于后续操作附着
    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));

    CHK_RET(RunInitStep(rank, rankSize));

    u32 sliceSize = slices_.size() / rankSize;

    // 例如rank[0,1,2,3]中，rank0的rxSliceIdx = 2，txSliceIdx = 3, subSliceIdx = 1
    u32 txSliceIdx  = (rank + rankSize - 1) % rankSize;
    u32 rxSliceIdx  = (rank + rankSize - DMA_REDUCE_TWO_OFFSET) % rankSize;
    u32 subSliceIdx = (rank + rankSize - DMA_REDUCE_THREE_OFFSET) % rankSize;
    for (u32 step = 0; step < rankSize - 1; step++) {
        std::vector<Slice> rxSliceVector;
        std::vector<Slice> cclSliceVector;
        std::vector<Slice> txSliceVector;
        std::vector<Slice> subSliceVector;
        for (u32 sliceIdx = 0; sliceIdx < sliceSize; sliceIdx++) {
            rxSliceVector.push_back(slices_[rxSliceIdx * sliceSize + sliceIdx]);
            cclSliceVector.push_back(slices_[subSliceIdx * sliceSize + sliceIdx]);
            txSliceVector.push_back(slices_[txSliceIdx * sliceSize + sliceIdx]);
            subSliceVector.push_back(userMemInputSlices_[subSliceIdx * sliceSize + sliceIdx]);
        }

        // 并发
        CHK_RET(MainRecordSub()); // 主流通知从流开始通信
        CHK_RET(SubWaitMain());   // 从流等待主流通知

        // 空拷贝用于主从流任务并发
        CHK_RET(ExecutorBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));
        CHK_RET(ExecutorBase::ExecEmptyTask(inputMem_, outputMem_, subStreams_[0], dispatcher_));

        // 主流
        CHK_RET(RunMainStream(step, txSliceVector, rxSliceVector, rank, rankSize));

        // 从流
        CHK_RET(RunSubStream(step, subSliceVector, cclSliceVector, rank, rankSize));

        CHK_RET(SubRecordMain()); // 从流通知主流通信完成
        CHK_RET(MainWaitSub());   // 主流等待从流通知

        // 更新索引
        subSliceIdx = (subSliceIdx + rankSize - 1) % rankSize;
        txSliceIdx  = (txSliceIdx + rankSize - 1) % rankSize;
        rxSliceIdx  = (rxSliceIdx + rankSize - 1) % rankSize;
    }
    HCCL_INFO("ReduceScatterRingConcurrentDirect finished to RunReduceScatter");
    return HCCL_SUCCESS;
}

// 主流通知从流干活
HcclResult ReduceScatterRingConcurrentDirect::MainRecordSub()
{
    for (u32 signalIndex = 0; signalIndex < subSignals_.size(); signalIndex++) {
        CHK_RET(LocalNotify::Post(stream_, dispatcher_, subSignals_[signalIndex],
            profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}
// 从流等待主流
HcclResult ReduceScatterRingConcurrentDirect::SubWaitMain()
{
    for (u32 streamIndex = 0; streamIndex < subSignals_.size(); streamIndex++) {
        CHK_RET(LocalNotify::Wait(subStreams_[streamIndex], dispatcher_, subSignals_[streamIndex],
            profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}
// 主流等待从流
HcclResult ReduceScatterRingConcurrentDirect::MainWaitSub()
{
    for (u32 signalIndex = 0; signalIndex < mainSignals_.size(); signalIndex++) {
        CHK_RET(LocalNotify::Wait(stream_, dispatcher_, mainSignals_[signalIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}
// 从流告诉主流活干完了
HcclResult ReduceScatterRingConcurrentDirect::SubRecordMain()
{
    for (u32 streamIndex = 0; streamIndex < mainSignals_.size(); streamIndex++) {
        CHK_RET(LocalNotify::Post(subStreams_[streamIndex], dispatcher_, mainSignals_[streamIndex],
            profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}
} // namespace hccl
