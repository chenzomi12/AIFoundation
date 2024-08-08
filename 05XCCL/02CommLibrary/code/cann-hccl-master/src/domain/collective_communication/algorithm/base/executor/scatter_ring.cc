/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "scatter_ring.h"

namespace hccl {
ScatterRing::ScatterRing(const HcclDispatcher dispatcher)
    : ExecutorBase(dispatcher), interRank_(0), interRankSize_(0)
{
}

ScatterRing::~ScatterRing()
{
}

HcclResult ScatterRing::RunScatterOnRootRank()
{
    DeviceMem src;
    DeviceMem dst;
    // rank存放scatter 结果的偏移
    u64 scatterOffset = slices_[interRank_].offset;
    u64 scatterResult = slices_[interRank_].size;

    HcclResult ret = HCCL_SUCCESS;
    // 需要判断input不等于outputmem，scatter 输入只有一个input时不用拷贝
    if (inputMem_ != outputMem_) {
        src = inputMem_.range(scatterOffset, scatterResult);
        dst = outputMem_.range(scatterOffset, scatterResult);

        HCCL_DEBUG("rootrank[%u] copy input[%p] to output[%p] scatter_offset[%llu] copysize[%llu]", \
                   interRank_, src.ptr(), dst.ptr(), scatterOffset, scatterResult);
        ret = HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Run][ScatterOnRootRank]root rank[%u] memcpy async from input[%p] "\
            "failed to output[%p]", interRank_, inputMem_.ptr(), outputMem_.ptr()), ret);
    }

    // 数据向下一个rank发送，依次发送后继所有rank的数据
    for (u32 i = 1; i < interRankSize_; i++) {
        u32 preRank = (interRank_ - i + interRankSize_) % interRankSize_;
        scatterOffset = slices_[preRank].offset;
        scatterResult = slices_[preRank].size;

        src = inputMem_.range(scatterOffset, scatterResult);
        // 等待后一节点同步信号，进行下一轮操作
        CHK_RET(linkRight_->RxAck(stream_));

        // 向root rank的后一rank发送
        HCCL_DEBUG(" root rank[%u] sendto dstrank[%u] from srcmem offset[%llu] size[%llu]", \
                   interRank_, preRank, scatterOffset, scatterResult);
        CHK_RET(linkRight_->TxAsync(UserMemType::OUTPUT_MEM, scatterOffset + baseOffset_, src.ptr(),
            scatterResult, stream_));

        HCCL_DEBUG("root rank[%u] will rx_ack", interRank_);
        ret = linkRight_->TxWaitDone(stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][ScatterOnRootRank]TxWaitDone failed"), ret);
    }
    return HCCL_SUCCESS;
}
HcclResult ScatterRing::RunScatterOnEndRank()
{
    DeviceMem src;
    DeviceMem dst;
    u64 scatterOffset = slices_[interRank_].offset;
    u64 scatterResult = slices_[interRank_].size;
    // 给前一节点发送同步，以便前一rank进行下一轮的操作
    CHK_RET(linkLeft_->TxAck(stream_));

    dst = outputMem_.range(scatterOffset, scatterResult);
    HCCL_DEBUG("last rank[%u] rx data ouputoffset[%llu] size[%llu]", \
        interRank_, scatterOffset, scatterResult);
    HcclResult ret = linkLeft_->RxAsync(UserMemType::OUTPUT_MEM, scatterOffset + baseOffset_, dst.ptr(),
        scatterResult, stream_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][ScatterOnEndRank]last rank[%u] rx sync failed", \
        interRank_), ret);
    ret = linkLeft_->RxWaitDone(stream_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][ScatterOnRootRank]RxWaitDone failed"), ret);
    return HCCL_SUCCESS;
}
HcclResult ScatterRing::RunScatterOnMidRank()
{
    DeviceMem src;
    DeviceMem dst;
    DeviceMem dstLast;
    // 与root的rank号之差 + 接收的轮数 = rank_size,  每个rank 接收的次数为 root_+ranksize-rank%interRankSize_
    u32 round = (root_ + interRankSize_ - interRank_) % interRankSize_;
    HCCL_DEBUG("rank:[%u] will receive %u rounds data", interRank_, round);

    UserMemType memType = (interRank_ == ((root_ + 1) % interRankSize_)) ?
        UserMemType::INPUT_MEM : UserMemType::OUTPUT_MEM;

    HcclResult ret = HCCL_SUCCESS;
    // 需要接收的和发送的轮数，包含接收自己的数据
    for (u32 i = 1; i <= round; i++) {
        u32 dataRank = (interRank_ + round - i) % interRankSize_; // 收到的数据应当是哪个rank的
        u64 scatterOffset = slices_[dataRank].offset;
        u64 scatterResult = slices_[dataRank].size;

        u32 lastDataRank = (interRank_ + round - i + 1) % interRankSize_; // 加1得到发送的数据应当是哪个rank的
        u64 scatterLastOffset = slices_[lastDataRank].offset;
        u64 scatterLastResult = slices_[lastDataRank].size;

        dst = outputMem_.range(scatterOffset, scatterResult);
        dstLast = outputMem_.range(scatterLastOffset, scatterLastResult);

        if (i != 1) {
            // 给前一节点发送同步，以便前一rank进行下一轮的操作
            ret = linkLeft_->TxAck(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][ScatterOnMidRank]rank[%u] round[%u] tx ack  failed",
                interRank_, i), ret);
            // 从后一rank接收同步信号
            ret = linkRight_->RxAck(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][ScatterOnMidRank]rank[%u]round[%u] rx ack failed",
                interRank_, i), ret);
            // 向后一rank发送数据
            HCCL_DEBUG("rank[%u] round[%u] tx async offset[%llu] size[%llu]", interRank_, \
                i, scatterLastOffset, scatterLastResult);
            ret = linkRight_->TxAsync(UserMemType::OUTPUT_MEM, scatterLastOffset + baseOffset_, dstLast.ptr(),
                scatterLastResult, stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][ScatterOnMidRank]rank[%u] round[%u] tx async failed",
                interRank_, i), ret);
        } else { // 最后一轮接收数据，拷贝到自己的outputmem
            // 给前一节点发送同步，以便前一rank进行下一轮的操作
            ret = linkLeft_->TxAck(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][ScatterOnMidRank]rank[%u] round[%u]  tx ack  failed",
                interRank_, i), ret);
        }
        HCCL_DEBUG("rank[%u] round[%u] rcv with rank[%u]'s offset[%llu] size[%llu]", \
            interRank_, i, dataRank, scatterOffset, scatterResult);
        CHK_RET(linkLeft_->RxAsync(memType, scatterOffset + baseOffset_, dst.ptr(), scatterResult, stream_));
        ret = linkRight_->TxWaitDone(stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][ScatterOnMidRank]TxWaitDone failed"), ret);
        ret = linkLeft_->RxWaitDone(stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][ScatterOnMidRank]RxWaitDone failed"), ret);
    }
    return HCCL_SUCCESS;
}

void ScatterRing::PrepareSlicesData(const u32 unitSize, const u64 totalCount, const u32 rankSize) const
{
    slices_.resize(rankSize);
    u64 sliceSize = (totalCount / rankSize) * unitSize;

    for (u32 i = 0; i < rankSize; i++) {
        slices_[i].offset = i * sliceSize;
        slices_[i].size = sliceSize;
        HCCL_DEBUG("rank[%u] default slice[%u]: offset: [%llu] size[%llu]", interRank_, i, i * sliceSize, sliceSize);
    }
}

// scatter的入口函数
HcclResult ScatterRing::RunAsync(const u32 rank, const u32 rankSize,
    const std::vector<std::shared_ptr<Transport> > &links)
{
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    if (!outputMem_ || !inputMem_) {
        HCCL_ERROR("[ScatterRing][RunAsync]run_async inputmem or outputmem is null");
        return HCCL_E_PTR;
    }

    interRank_ = rank;
    interRankSize_ = rankSize;

    HCCL_INFO("ScatterRing run: rank[%u] totalrank[%u]  count[%llu] input[%p] output[%p]",
              interRank_, interRankSize_,  count_, inputMem_.ptr(), outputMem_.ptr());

    // ranksize为1时，只有当input!=ouput 时候进行拷贝
    if (interRankSize_ == 1) {
        if (inputMem_ != outputMem_) {
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_));
        }
        return HCCL_SUCCESS;
    }

    u32 unitSize = DataUnitSize(dataType_);
    CHK_PRT_RET(unitSize == 0, HCCL_ERROR("[ScatterRing][RunAsync]rank[%u] unit data size is zero", rank),
        HCCL_E_INTERNAL);

    // 带入vecotr为空，计算每个rank的结果偏移和大小
    if (slices_.size() == 0) {
        PrepareSlicesData(unitSize, count_, interRankSize_);
    }

    // 获取link的收、发缓存, 计算chunk_size
    u32 ringPrevRank = (rank + rankSize - 1) % rankSize;
    u32 ringNextRank = (rank + 1) % rankSize;

    if (links.size() < rankSize) {
        HCCL_ERROR("[ScatterRing][RunAsync]rank[%u] link size[%llu] is less than rank size", rank, links.size());
        return HCCL_E_INTERNAL;
    }

    linkLeft_ = links[ringPrevRank];
    CHK_SMART_PTR_NULL(linkLeft_);

    linkRight_ = links[ringNextRank];
    CHK_SMART_PTR_NULL(linkRight_);

    CHK_RET(ScatterSlicesPrep(rankSize, nicRankList_.size()));

    // 单环场景下 nicRankList_ 长度默认为 8。
    // 多环场景下 nicRankList_ 长度为网口数量。此时若 rankSize != nicRankList_ 则为网口裁剪场景
    if (rankSize != HCCL_NIC_MAX_NUM || nicRankList_.size() == HCCL_NIC_MAX_NUM) {
        // 非网口裁剪场景:
        // root rank向其他rank发送数据,
        if (interRank_ == root_) {
            CHK_RET(RunScatterOnRootRank());
        } else if (ringNextRank == root_) { // 最后一个节点只负责接收数据，拷贝至outputmem
            CHK_RET(RunScatterOnEndRank());
        } else {
            CHK_RET(RunScatterOnMidRank());
        }
    } else {
        // 网口裁剪场景：当前仅在 910A 8P_RING (4环)，且网口不满配情况下使用
        CHK_RET(RunScatterChunk(rank, rankSize, slices_));
    }

    if (barrierSwitchOn_) {
        // 执行barrier，保证数据收发完成
        CHK_RET(ExecuteBarrier(linkLeft_, linkRight_));
    }

    HCCL_INFO("ScatterRing finished: rank:[%u] end", interRank_);

    return HCCL_SUCCESS;
}

HcclResult ScatterRing::RunScatterChunk(const u32 rank, const u32 rankSize, const std::vector<Slice> &outputSlices)
{
    HcclResult ret;
    DeviceMem dst;
    u32 sendSliceLen = rankSliceLists_[rank].size();
    u32 chunkSize = HCCL_NIC_MAX_NUM / nicRankList_.size();
    if (sendSliceLen >= chunkSize) {
        CHK_RET(HeadScatterChunk(rank, rankSize, outputSlices));
        for (u32 midRankIdx = 1; midRankIdx < sendSliceLen - 1; midRankIdx++) {
            ret = MidScatterChunk(rank, rankSize, midRankIdx, outputSlices);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Run][ScatterChunk]rank[%u] run mid[%u] reduce scatter chunk failed",
                rank, midRankIdx), HCCL_E_INTERNAL);
        }
        CHK_RET(TailScatterChunk(rank, rankSize, sendSliceLen - 1, outputSlices));
    } else if (rankSliceLists_[(rank + rankSize - 1) % rankSize].size() != 0) {
        for (u32 sliceIdx = 0; sliceIdx < chunkSize; sliceIdx++) {
            std::vector<u32>::iterator iterNic = std::find(nicRankList_.begin(), nicRankList_.end(), rank);
            u32 nicIdx = distance(nicRankList_.begin(), iterNic);
            u32 chunkStart = nicIdx * chunkSize;
            u32 rxSliceIndex = chunkStart + sliceIdx;
            u64 rxScatterOffset = slices_[rxSliceIndex].offset;
            u64 rxScatterResult = slices_[rxSliceIndex].size;
            dst = outputMem_.range(rxScatterOffset, rxScatterResult);
            CHK_RET(linkLeft_->TxAck(stream_));

            ret = linkLeft_->RxAsync(UserMemType::OUTPUT_MEM, rxScatterOffset + baseOffset_, dst.ptr(),
                rxScatterResult, stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Run][ScatterChunk]rank[%u] Left Link rx outputSlices[%u] Failed",
                rank, rxSliceIndex), ret);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ScatterRing::HeadScatterChunk(u32 rank, u32 rankSize, const std::vector<Slice> &outputSlices)
{
    HcclResult ret;
    DeviceMem dst;
    u32 rxSliceIndex = rankSliceLists_[rank][0];
    u32 txSliceIndex = rxSliceIndex;
    u64 scatterOffset = slices_[rxSliceIndex].offset;
    u64 scatterResult = slices_[rxSliceIndex].size;
    dst = outputMem_.range(scatterOffset, scatterResult);
    std::vector<u32> preRankSlices(rankSliceLists_[(rank - 1 + rankSize) % rankSize]);
    std::vector<u32>::iterator iterSlice = std::find(preRankSlices.begin(), preRankSlices.end(), rxSliceIndex);
    if (iterSlice != preRankSlices.end()) {
        CHK_RET(linkLeft_->TxAck(stream_));

        ret = linkLeft_->RxAsync(UserMemType::OUTPUT_MEM, scatterOffset + baseOffset_, dst.ptr(),
            scatterResult, stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[ScatterRing][HeadScatterChunk]rank[%u] Left Link rx "\
            "outputSlices[%u] Failed", rank, rxSliceIndex), ret);
    }
    iterSlice = std::find(preRankSlices.begin(), preRankSlices.end(), rankSliceLists_[rank][1]);
    if (iterSlice != preRankSlices.end()) {
        CHK_RET(MidScatterChunk(rank, rankSize, 0, outputSlices));
    } else {
        CHK_RET(linkRight_->RxAck(stream_));

        ret = linkRight_->TxAsync(UserMemType::OUTPUT_MEM, scatterOffset + baseOffset_, dst.ptr(),
            scatterResult, stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[ScatterRing][HeadScatterChunk]rank[%u] Right Link tx "\
            "outputSlices[%u] Failed", rank, txSliceIndex), ret);
    }
    return HCCL_SUCCESS;
}

HcclResult ScatterRing::MidScatterChunk(u32 rank, u32 rankSize, u32 sliceIdx, const std::vector<Slice> &outputSlices)
{
    (void)outputSlices;
    HcclResult ret;
    DeviceMem dst;
    u32 rxSliceIndex = rankSliceLists_[rank][sliceIdx + 1];
    u32 txSliceIndex = rankSliceLists_[rank][sliceIdx];
    u64 rxScatterOffset = slices_[rxSliceIndex].offset;
    u64 rxScatterResult = slices_[rxSliceIndex].size;
    u64 txScatterOffset = slices_[txSliceIndex].offset;
    u64 txScatterResult = slices_[txSliceIndex].size;

    std::vector<u32> preRankSlices(rankSliceLists_[(rank - 1 + rankSize) % rankSize]);
    std::vector<u32>::iterator iterSlice = std::find(preRankSlices.begin(), preRankSlices.end(), rxSliceIndex);
    if (iterSlice != preRankSlices.end()) {
        CHK_RET(linkLeft_->TxAck(stream_));

        dst = outputMem_.range(txScatterOffset, txScatterResult);
        CHK_RET(linkRight_->RxAck(stream_));

        ret = linkRight_->TxAsync(UserMemType::OUTPUT_MEM, txScatterOffset + baseOffset_, dst.ptr(),
            txScatterResult, stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[ScatterRing][MidScatterChunk]rank[%u] Right Link tx "\
            "outputSlices[%u] Failed", rank, txSliceIndex), ret);
        dst = outputMem_.range(rxScatterOffset, rxScatterResult);
        ret = linkLeft_->RxAsync(UserMemType::OUTPUT_MEM, rxScatterOffset + baseOffset_, dst.ptr(),
            rxScatterResult, stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[ScatterRing][MidScatterChunk]rank[%u] Left Link rx "\
            "outputSlices[%u] Failed", rank, rxSliceIndex), ret);
    } else {
        dst = outputMem_.range(txScatterOffset, txScatterResult);
        CHK_RET(linkRight_->RxAck(stream_));

        ret = linkRight_->TxAsync(UserMemType::OUTPUT_MEM, txScatterOffset + baseOffset_, dst.ptr(),
            txScatterResult, stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[ScatterRing][MidScatterChunk]rank[%u] Right Link tx "\
            "outputSlices[%u] Failed", rank, txSliceIndex), ret);
    }
    return HCCL_SUCCESS;
}

HcclResult ScatterRing::TailScatterChunk(u32 rank, u32 rankSize, u32 sliceIdx, const std::vector<Slice> &outputSlices)
{
    (void)rankSize;
    (void)outputSlices;
    HcclResult ret;
    DeviceMem dst;
    u32 chunkSize = HCCL_NIC_MAX_NUM / nicRankList_.size();
    u32 txSliceIndex = rankSliceLists_[rank][sliceIdx];
    u64 txScatterOffset = slices_[txSliceIndex].offset;
    u64 txScatterResult = slices_[txSliceIndex].size;
    std::vector<u32>::iterator iterNic = std::find(nicRankList_.begin(), nicRankList_.end(), rank);
    if (iterNic != nicRankList_.end() && rank != root_) {
        u32 nicIdx = distance(nicRankList_.begin(), iterNic);
        u32 chunkStart = nicIdx * chunkSize;
        u32 rxSliceIndex = chunkStart;
        u64 rxScatterOffset = slices_[rxSliceIndex].offset;
        u64 rxScatterResult = slices_[rxSliceIndex].size;
        CHK_RET(linkLeft_->TxAck(stream_));

        dst = outputMem_.range(txScatterOffset, txScatterResult);
        CHK_RET(linkRight_->RxAck(stream_));

        ret = linkRight_->TxAsync(UserMemType::OUTPUT_MEM, txScatterOffset + baseOffset_, dst.ptr(),
            txScatterResult, stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[ScatterRing][TailScatterChunk]rank[%u] Right Link tx "\
            "outputSlices[%u] Failed", rank, txSliceIndex), ret);
        dst = outputMem_.range(rxScatterOffset, rxScatterResult);
        ret = linkLeft_->RxAsync(UserMemType::OUTPUT_MEM, rxScatterOffset + baseOffset_, dst.ptr(),
            rxScatterResult, stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[ScatterRing][TailScatterChunk]rank[%u] Left Link rx "\
            "outputSlices[%u] Failed", rank, rxSliceIndex), ret);

        for (u32 sliceIdx = 1; sliceIdx < chunkSize; sliceIdx++) {
            rxSliceIndex = chunkStart + sliceIdx;
            u64 rxScatterOffset = slices_[rxSliceIndex].offset;
            u64 rxScatterResult = slices_[rxSliceIndex].size;
            dst = outputMem_.range(rxScatterOffset, rxScatterResult);
            CHK_RET(linkLeft_->TxAck(stream_));

            ret = linkLeft_->RxAsync(UserMemType::OUTPUT_MEM, rxScatterOffset + baseOffset_, dst.ptr(),
                rxScatterResult, stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[ScatterRing][TailScatterChunk]rank[%u] Left Link rx "\
                "outputSlices[%u] Failed", rank, rxSliceIndex), ret);
        }
    } else {
        dst = outputMem_.range(txScatterOffset, txScatterResult);
        CHK_RET(linkRight_->RxAck(stream_));

        ret = linkRight_->TxAsync(UserMemType::OUTPUT_MEM, txScatterOffset + baseOffset_, dst.ptr(),
            txScatterResult, stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[ScatterRing][TailScatterChunk]rank[%u] Right Link tx "\
            "outputSlices[%u] Failed", rank, txSliceIndex), ret);
    }
    return HCCL_SUCCESS;
}

HcclResult ScatterRing::ScatterSlicesPrep(u32 rankSize, u32 nicSize)
{
    u32 chunkSize = HCCL_NIC_MAX_NUM / nicSize;
    for (u32 rankIdx = 0; rankIdx < rankSize; rankIdx++) {
        std::vector<u32> sliceList;    // 单个rank上的发送slice编号
        for (u32 nicDis = 1; nicDis <= rankSize; nicDis++) {    // 递减从root遍历至当前rank的位置
            u32 nicIdx = (root_ + rankSize - nicDis) % rankSize;
            if (rankIdx == nicIdx) {
                break;
            }
            std::vector<u32>::iterator iterNic = std::find(nicRankList_.begin(), nicRankList_.end(), nicIdx);
            if (iterNic != nicRankList_.end()) {    // 当前rank为网口所在位置，将网口对应的chunksize份silce放入sliceList
                u32 nicListIdx = distance(nicRankList_.begin(), iterNic);
                for (u32 chunkIdx = 0; chunkIdx < chunkSize; chunkIdx++) {
                    sliceList.push_back(chunkSize * nicListIdx + chunkIdx);
                }
            }
        }
        rankSliceLists_.push_back(sliceList);
    }
    return HCCL_SUCCESS;
}
}  // namespace hccl
