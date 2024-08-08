/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "multi_root_scatter_ring.h"

namespace hccl {
bool DscendSortWithSliceSendEnd(const SliceSendRange &a, const SliceSendRange &b)
{
    return (a.endRank > b.endRank);
}

MultiRootScatterRing::MultiRootScatterRing(const HcclDispatcher dispatcher)
    : ExecutorBase(dispatcher), interRank_(0), interRankSize_(0)
{
}

MultiRootScatterRing::~MultiRootScatterRing()
{
}

void MultiRootScatterRing::SlicesDataPrepare(const u32 unitSize, const u64 totalCount, const u32 rankSize) const
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
HcclResult MultiRootScatterRing::RunAsync(const u32 rank, const u32 rankSize,
    const std::vector<std::shared_ptr<Transport>> &links)
{
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    if (!outputMem_ || !inputMem_) {
        HCCL_ERROR("[MultiRootScatterRing][RunAsync]run_async inputmem or outputmem is null");
        return HCCL_E_PTR;
    }

    interRank_ = rank;
    interRankSize_ = rankSize;

    HCCL_INFO("MultiRootScatterRing run: rank[%u] totalrank[%u] count[%llu] input[%p] output[%p]",
              interRank_, interRankSize_,  count_, inputMem_.ptr(), outputMem_.ptr());

    // ranksize为1时，只有当input!=ouput 时候进行拷贝
    if (interRankSize_ == 1) {
        if (inputMem_ != outputMem_) {
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_));
        }
        return HCCL_SUCCESS;
    }

    u32 unitSize = DataUnitSize(dataType_);
    CHK_PRT_RET(unitSize == 0, HCCL_ERROR("[MultiRootScatterRing][RunAsync]rank[%u] unit data size is zero", rank),
        HCCL_E_INTERNAL);

    // 带入vecotr为空，计算每个rank的结果偏移和大小
    if (slices_.size() == 0) {
        SlicesDataPrepare(unitSize, count_, interRankSize_);
    }

    // 获取link的收、发缓存, 计算chunk_size
    u32 ringPrevRank = (rank + rankSize - 1) % rankSize;
    u32 ringNextRank = (rank + 1) % rankSize;

    if (links.size() < rankSize) {
        HCCL_ERROR("[MultiRootScatterRing][RunAsync]rank[%u] link size[%llu] is less than rank size",
            rank, links.size());
        return HCCL_E_INTERNAL;
    }

    linkLeft_ = links[ringPrevRank];
    CHK_SMART_PTR_NULL(linkLeft_);

    linkRight_ = links[ringNextRank];
    CHK_SMART_PTR_NULL(linkRight_);

    CHK_RET(MultiRootScatterSlicesPrep(rankSize, nicRankList_.size()));

    CHK_RET(RunMultiRootScatterChunk(rank, rankSize, slices_));

    if (barrierSwitchOn_) {
        // 执行barrier，保证数据收发完成
        CHK_RET(ExecuteBarrier(linkLeft_, linkRight_));
    }

    return HCCL_SUCCESS;
}

HcclResult MultiRootScatterRing::RunMultiRootScatterChunk(const u32 rank, const u32 rankSize,
                                                          const std::vector<Slice> &outputSlices)
{
    HcclResult ret;
    DeviceMem dstMem;
    u32 sendSliceLen = rankSliceLists_[rank].size();
    if (sendSliceLen >= 1) { // 如果slice序列大于等于1则，存在头结点，进行相应slice的发送
        CHK_RET(HeadScatterChunk(rank, rankSize, outputSlices));
        for (u32 midRankIdx = 1; midRankIdx < sendSliceLen - 1; midRankIdx++) {
            ret = MidScatterChunk(rank, rankSize, midRankIdx, outputSlices);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Run][MultiRootScatterChunk]rank[%u] run mid[%u] reduce scatter chunk "\
                    "failed", rank, midRankIdx), HCCL_E_INTERNAL);
        }
    }

    if (sendSliceLen >= 2) { // 如果slice序列大于等于2则，存在尾结点，进行相应slice的发送
        CHK_RET(TailScatterChunk(rank, rankSize, sendSliceLen - 1, outputSlices));
    }

    if (sendSliceLen == 0) { // 如果slice序列长度为0，则接受当前rank会最终保存的slice即可
        u32 rxSliceIndex = (rank - nicRankList_[0] + HCCL_NIC_MAX_NUM) % HCCL_NIC_MAX_NUM;
        u64 rxScatterOffset = slices_[rxSliceIndex].offset;
        u64 rxScatterResult = slices_[rxSliceIndex].size;
        std::vector<u32> preRankSlices(rankSliceLists_[(rank - 1 + rankSize) % rankSize]);
        std::vector<u32>::iterator iterSlice = std::find(preRankSlices.begin(), preRankSlices.end(), rxSliceIndex);
        if (iterSlice != preRankSlices.end()) {
            CHK_RET(linkLeft_->TxAck(stream_));

            dstMem = outputMem_.range(rxScatterOffset, rxScatterResult);
            ret = linkLeft_->RxAsync(UserMemType::OUTPUT_MEM, baseOffset_ + rxScatterOffset, dstMem.ptr(),
                rxScatterResult, stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Run][MultiRootScatterChunk]rank[%u] Left Link rx outputSlices"\
                    "[%u] Failed", rank, rxSliceIndex), ret);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult MultiRootScatterRing::HeadScatterChunk(u32 rank, u32 rankSize, const std::vector<Slice> &outputSlices)
{
    HcclResult ret;
    // 头结点发送及接收slice均为rankSliceLists_的第一个元素
    u32 rxSliceIndex = rankSliceLists_[rank][0];
    // 得到发送及接收slice的偏移和长度
    u64 scatterOffset = slices_[rxSliceIndex].offset;
    u64 scatterResult = slices_[rxSliceIndex].size;
    DeviceMem dstMem = outputMem_.range(scatterOffset, scatterResult);
    // 判断当前rank是否需要接收头结点的数据, 得到前一个rank的发送序列，判断当前发送的slice是否在该序列中
    std::vector<u32> preRankSlices(rankSliceLists_[(rank - 1 + rankSize) % rankSize]);
    std::vector<u32>::iterator iterSlice = std::find(preRankSlices.begin(), preRankSlices.end(), rxSliceIndex);
    if (iterSlice != preRankSlices.end()) { // 若发送slice在前一个rank的发送序列中，则需先从前一个rank中接收对应数据
        CHK_RET(linkLeft_->TxAck(stream_));

        ret = linkLeft_->RxAsync(UserMemType::OUTPUT_MEM, baseOffset_ + scatterOffset,
            dstMem.ptr(), scatterResult, stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[MultiRootScatterRing][HeadScatterChunk]rank[%u] Left Link rx "\
                "outputSlices[%u] Failed", rank, rxSliceIndex), ret);
    }

    if (rankSliceLists_[rank].size() >= 2) { // 发送序列长度>=2时，需判断发送第一个slice前是否需要接收第二段slice
        iterSlice = std::find(preRankSlices.begin(), preRankSlices.end(), rankSliceLists_[rank][1]);
        if (iterSlice != preRankSlices.end()) { // 需要接收第二段slice，此时头结点行为和中间结点一致
            CHK_RET(MidScatterChunk(rank, rankSize, 0, outputSlices));
            return HCCL_SUCCESS;
        }
    } else if (rankSliceLists_[rank].size() == 1) { // 发送序列只有一个slice，则头结点同时为尾节点，需要接收最终要保存的数据
        u32 rxTailIndex = (rank - nicRankList_[0] + HCCL_NIC_MAX_NUM) % HCCL_NIC_MAX_NUM; // 计算当前rank最终要保存的数据
        u64 rxTailOffset = slices_[rxTailIndex].offset;
        u64 rxTailResult = slices_[rxTailIndex].size;
        // 判断当前rank是否需要接收最终要保存的数据
        std::vector<u32>::iterator iterSlice = std::find(preRankSlices.begin(), preRankSlices.end(), rxSliceIndex);
        if (iterSlice != preRankSlices.end()) { // 接受的数据在前rank的发送序列中
            CHK_RET(linkLeft_->TxAck(stream_));

            CHK_RET(linkRight_->RxAck(stream_));

            ret = linkRight_->TxAsync(UserMemType::OUTPUT_MEM, baseOffset_ + scatterOffset,
                dstMem.ptr(), scatterResult, stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[MultiRootScatterRing][HeadScatterChunk]rank[%u] Right Link tx "\
                    "outputSlices[%u] Failed", rank, rxSliceIndex), ret);

            dstMem = outputMem_.range(rxTailOffset, rxTailResult);
            ret = linkLeft_->RxAsync(UserMemType::OUTPUT_MEM, baseOffset_ + rxTailOffset, dstMem.ptr(),
                rxTailResult, stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[MultiRootScatterRing][HeadScatterChunk]rank[%u] Left Link rx "\
                    "outputSlices[%u] Failed", rank, rxTailIndex), ret);
            return HCCL_SUCCESS;
        }
    }
    // 其他情况直接发送当前头结点slice
    CHK_RET(linkRight_->RxAck(stream_));

    ret = linkRight_->TxAsync(UserMemType::OUTPUT_MEM, baseOffset_ + scatterOffset, dstMem.ptr(),
        scatterResult, stream_);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[MultiRootScatterRing][HeadScatterChunk]rank[%u] Right Link tx "\
            "outputSlices[%u] Failed", rank, rxSliceIndex), ret);

    return HCCL_SUCCESS;
}

HcclResult MultiRootScatterRing::MidScatterChunk(u32 rank, u32 rankSize, u32 sliceIdx,
                                                 const std::vector<Slice> &outputSlices)
{
    (void)outputSlices;
    HcclResult ret;
    DeviceMem dstMem;
    // 头结点发送slice为rankSliceLists_的第sliceIdx个元素，接收slice为rankSliceLists_的第sliceIdx+1个元素
    u32 rxSliceIndex = rankSliceLists_[rank][sliceIdx + 1];
    u32 txSliceIndex = rankSliceLists_[rank][sliceIdx];
    u64 rxScatterOffset = slices_[rxSliceIndex].offset;
    u64 rxScatterResult = slices_[rxSliceIndex].size;
    u64 txScatterOffset = slices_[txSliceIndex].offset;
    u64 txScatterResult = slices_[txSliceIndex].size;
    // 判断当前rank是否需要接收第sliceIdx+1个元素, 得到前一个rank的发送序列，判断当前发送的slice是否在该序列中
    std::vector<u32> preRankSlices(rankSliceLists_[(rank - 1 + rankSize) % rankSize]);
    std::vector<u32>::iterator iterSlice = std::find(preRankSlices.begin(), preRankSlices.end(), rxSliceIndex);
    if (iterSlice != preRankSlices.end()) { // 若发送slice在前一个rank的发送序列中，则需先从前一个rank中接收对应数据
        CHK_RET(linkLeft_->TxAck(stream_));

        dstMem = outputMem_.range(txScatterOffset, txScatterResult);
        CHK_RET(linkRight_->RxAck(stream_));

        ret = linkRight_->TxAsync(UserMemType::OUTPUT_MEM, baseOffset_ + txScatterOffset,
            dstMem.ptr(), txScatterResult, stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[MultiRootScatterRing][MidScatterChunk]rank[%u] Right Link tx "\
            "outputSlices[%u] Failed", rank, txSliceIndex), ret);

        dstMem = outputMem_.range(rxScatterOffset, rxScatterResult);
        ret = linkLeft_->RxAsync(UserMemType::OUTPUT_MEM, baseOffset_ + rxScatterOffset,
            dstMem.ptr(), rxScatterResult, stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[MultiRootScatterRing][MidScatterChunk]rank[%u] Left Link rx "\
            "outputSlices[%u] Failed", rank, rxSliceIndex), ret);
    } else { // 其他情况直接发送当前中间结点slice
        dstMem = outputMem_.range(txScatterOffset, txScatterResult);
        CHK_RET(linkRight_->RxAck(stream_));

        ret = linkRight_->TxAsync(UserMemType::OUTPUT_MEM, baseOffset_ + txScatterOffset,
            dstMem.ptr(), txScatterResult, stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[MultiRootScatterRing][MidScatterChunk]rank[%u] Right Link tx "\
            "outputSlices[%u] Failed", rank, txSliceIndex), ret);
    }
    return HCCL_SUCCESS;
}

HcclResult MultiRootScatterRing::TailScatterChunk(u32 rank, u32 rankSize, u32 sliceIdx,
                                                  const std::vector<Slice> &outputSlices)
{
    (void)outputSlices;
    HcclResult ret;
    DeviceMem dstMem;
    // 尾结点发送slice为rankSliceLists_的第sliceIdx个元素，接收slice为scatter最终会保存的slice位置
    u32 txSliceIndex = rankSliceLists_[rank][sliceIdx];
    u64 txScatterOffset = slices_[txSliceIndex].offset;
    u64 txScatterResult = slices_[txSliceIndex].size;

    u32 rxSliceIndex = (rank - nicRankList_[0] + HCCL_NIC_MAX_NUM) % HCCL_NIC_MAX_NUM;
    u64 rxScatterOffset = slices_[rxSliceIndex].offset;
    u64 rxScatterResult = slices_[rxSliceIndex].size;

    // 判断当前rank是否需要接收第sliceIdx+1个元素, 得到前一个rank的发送序列，判断当前发送的slice是否在该序列中
    std::vector<u32> preRankSlices(rankSliceLists_[(rank - 1 + rankSize) % rankSize]);
    std::vector<u32>::iterator iterSlice = std::find(preRankSlices.begin(), preRankSlices.end(), rxSliceIndex);
    if (iterSlice != preRankSlices.end()) { // 若接收slice在前一个rank的发送序列中，则需先从前一个rank中接收对应数据
        CHK_RET(linkLeft_->TxAck(stream_));

        dstMem = outputMem_.range(txScatterOffset, txScatterResult);
        CHK_RET(linkRight_->RxAck(stream_));

        ret = linkRight_->TxAsync(UserMemType::OUTPUT_MEM, baseOffset_ + txScatterOffset, dstMem.ptr(),
            txScatterResult, stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[MultiRootScatterRing][TailScatterChunk]rank[%u] Right Link tx "\
            "outputSlices[%u] Failed", rank, txSliceIndex), ret);

        dstMem = outputMem_.range(rxScatterOffset, rxScatterResult);
        ret = linkLeft_->RxAsync(UserMemType::OUTPUT_MEM, baseOffset_ + rxScatterOffset, dstMem.ptr(),
            rxScatterResult, stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[MultiRootScatterRing][TailScatterChunk]rank[%u] Left Link rx "\
            "outputSlices[%u] Failed", rank, rxSliceIndex), ret);
    } else { // 其他情况直接发送当前尾结点slice
        dstMem = outputMem_.range(txScatterOffset, txScatterResult);
        CHK_RET(linkRight_->RxAck(stream_));

        ret = linkRight_->TxAsync(UserMemType::OUTPUT_MEM, baseOffset_ + txScatterOffset,
            dstMem.ptr(), txScatterResult, stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[MultiRootScatterRing][TailScatterChunk]rank[%u] Right Link tx "\
            "outputSlices[%u] Failed", rank, txSliceIndex), ret);
    }
    return HCCL_SUCCESS;
}

HcclResult MultiRootScatterRing::MultiRootScatterSlicesPrep(u32 rankSize, u32 nicSize)
{
    u32 chunkSize = HCCL_NIC_MAX_NUM / nicSize;
    std::vector<SliceSendRange> sliceSendRangeVec;
    for (u32 nicIdx = 0; nicIdx < nicSize; nicIdx++) { // 计算每个网口负责的slice发送顺序
        for (u32 sliceIdx = 0; sliceIdx < chunkSize; sliceIdx++) { // 记录每个网口发送slice的起点和终点
            SliceSendRange tempSliceSendRange;
            tempSliceSendRange.sliceIdx = nicIdx * chunkSize + sliceIdx;
            tempSliceSendRange.startRank = nicRankList_[nicIdx];
            tempSliceSendRange.endRank = (nicIdx * chunkSize + sliceIdx + nicRankList_[0]) % HCCL_NIC_MAX_NUM;
            if (tempSliceSendRange.endRank < tempSliceSendRange.startRank) {
                tempSliceSendRange.endRank = tempSliceSendRange.endRank + HCCL_NIC_MAX_NUM;
            }
            sliceSendRangeVec.push_back(tempSliceSendRange);
        }
    }

    for (u32 rankIdx = 0; rankIdx < rankSize; rankIdx++) { // 计算每个rank发送slice的顺序
        std::vector<u32> sliceList;                        // 单个rank上的发送slice编号
        std::vector<SliceSendRange> rankSliceSendVec;
        // 从后往前依次遍历slice, 判断当前rank是否需要发送当前slice
        std::vector<SliceSendRange>::iterator sliceSendIdx = sliceSendRangeVec.end() - 1;
        for (; sliceSendIdx >= sliceSendRangeVec.begin(); sliceSendIdx--) {
            SliceSendRange rankSliceSend;
            if (rankIdx >= sliceSendIdx->startRank) { // slice终点rank号大于起点rank号
                if (rankIdx < sliceSendIdx->endRank) {
                    rankSliceSend.sliceIdx = sliceSendIdx->sliceIdx;
                    rankSliceSend.endRank = sliceSendIdx->endRank - rankIdx;
                    rankSliceSendVec.push_back(rankSliceSend);
                }
            } else { // slice终点rank号小于起点rank号
                u32 tempRankIdx = rankIdx + HCCL_NIC_MAX_NUM;
                if (tempRankIdx < sliceSendIdx->endRank) {
                    rankSliceSend.sliceIdx = sliceSendIdx->sliceIdx;
                    rankSliceSend.endRank = sliceSendIdx->endRank - tempRankIdx;
                    rankSliceSendVec.push_back(rankSliceSend);
                }
            }
        }
        std::sort(rankSliceSendVec.begin(), rankSliceSendVec.end(), DscendSortWithSliceSendEnd);
        for (u32 sliceIdx = 0; sliceIdx < rankSliceSendVec.size(); sliceIdx++) {
            sliceList.push_back(rankSliceSendVec[sliceIdx].sliceIdx);
        }
        rankSliceLists_.push_back(sliceList);
    }

    return HCCL_SUCCESS;
}
}  // namespace hccl
