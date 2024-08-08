/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_recursive_hd.h"
#include "reduce_scatter_halving_doubling_pub.h"

namespace hccl {
ReduceRecursiveHalvingDoubling::ReduceRecursiveHalvingDoubling(
    const HcclDispatcher dispatcher, const u64 reduceAttrBitMap)
    : RecursiveHalvingDoublingBase(dispatcher), reduceAttr(reduceAttrBitMap)
{
}

ReduceRecursiveHalvingDoubling::~ReduceRecursiveHalvingDoubling()
{
}

// 算法的主入口
HcclResult ReduceRecursiveHalvingDoubling::RunAsync(const u32 rank, const u32 rankSize,
                                                    const std::vector<std::shared_ptr<Transport> > &links)
{
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    if (!outputMem_ || !inputMem_) {
        HCCL_ERROR("[ReduceRecursiveHalvingDoubling][RunAsync]rank[%u] run_async inputmem or outputmem is null",
            rank);
        return HCCL_E_PTR;
    }
    HCCL_INFO("ReduceRecursiveHalvingDoubling run: rank[%u] root[%u] totalrank[%u] inputMem[%p] outputMem[%p] "
        "count[%llu]", rank, root_, rankSize, inputMem_.ptr(), outputMem_.ptr(), count_);

    HcclResult ret = HCCL_SUCCESS;

    if (rankSize == 1) {
        if (inputMem_ != outputMem_) {
            ret = HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_);
        }
        return ret;
    }

    senderInfo_.reset(new (std::nothrow) Sender(dataType_, reductionOp_, reduceAttr));
    CHK_SMART_PTR_NULL(senderInfo_);

    reducerInfo_.reset(new (std::nothrow) Reducer(dataType_, reductionOp_, reduceAttr));
    CHK_SMART_PTR_NULL(reducerInfo_);

    bool bRetSize = (links.size() < rankSize);
    CHK_PRT_RET(bRetSize,
        HCCL_ERROR("[ReduceRecursiveHalvingDoubling][RunAsync]rank[%u] linksize[%llu] is error",
        rank, links.size()), HCCL_E_INTERNAL);

    CHK_RET(CalcPartOneSizeAndBlockSize(rankSize));

    u32 bytesPerData = DataUnitSize(dataType_);
    u64 dataBytes = count_ * bytesPerData;
    CHK_RET(CalculateSlices(dataBytes));

    // 结果完成需要放在input
    CHK_RET(ReduceInPartOne(rank, links));

    // 此步骤完成后，结果放在ouput中
    CHK_RET(ReduceScatterInBlock(rank, rankSize, links));

    // 使用output进行gather
    CHK_RET(GatherInBlock(rank, rankSize, links));

    HCCL_INFO("ReduceRecursiveHalvingDoubling finished: rank[%u] finished", rank);
    return HCCL_SUCCESS;
}

HcclResult ReduceRecursiveHalvingDoubling::ReduceInPartOne(u32 rank, const std::vector<LINK> &links)
{
    HCCL_INFO("rank[%u] part1Size_[%u] root[%u]", rank, part1Size_, root_);

    if (rank >= part1Size_) { // rank在第二部分，不参与ReduceInPartOne
        HCCL_INFO("rank[%u] not in part1, don't need reduce", rank);
        return HCCL_SUCCESS;
    }
    // root在第二部分，需要选取第一部分偶数rank接收，以0作为判断标准，否则在第一部分，与root奇偶性相同rank接收
    u32 rootFlag = (root_ >= part1Size_) ? 0 : root_;

    if (rank % 2 == rootFlag % 2) {  // 1.从下一个rank接收数据到output，2. reduce到本rank的input
        u32 peerRank = (rank % 2) == 0 ? (rank + 1) : (rank - 1);
        HCCL_INFO("rank[%u] outputMem receives from PeerRank[%u] inputMem, Offset[%llu], Size[%llu]", \
                  rank, peerRank, baseOffset_, outputMem_.size());

        if (peerRank < links.size()) {
            CHK_SMART_PTR_NULL(links[peerRank]);

            HcclResult ret = links[peerRank]->TxAck(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Reduce][InPartOne]tx ack to peerrank[%u] failed", peerRank), ret);
            ret = links[peerRank]->RxAck(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Reduce][InPartOne]rx ack from peerank[%u] failed", peerRank), ret);

            //  接收数据到本端的 output
            HCCL_DEBUG("send mem[%p] size[%llu] to peerank[%u]", outputMem_.ptr(), outputMem_.size(), peerRank);
            ret = links[peerRank]->TxAsync(UserMemType::INPUT_MEM, baseOffset_, outputMem_.ptr(), 0, stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Reduce][InPartOneToEven]TxAsync: tx async size[%llu] "\
                "failed", 0), ret);
            CHK_RET(reducerInfo_->run(dispatcher_, links[peerRank], baseOffset_,
                outputMem_, inputMem_, outputMem_, stream_));
            ret = links[peerRank]->RxWaitDone(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Reduce][InPartOne]RxWaitDone failed"), ret);
        }
    } else if ((rank % 2) != (rootFlag % 2)) { //  向上一个rank的output发数据 2
        u32 peerRank = (rank % 2 == 0) ? (rank + 1) : (rank -1);

        if (peerRank < links.size()) {
            CHK_SMART_PTR_NULL(links[peerRank]);
            HcclResult ret = links[peerRank]->TxAck(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Reduce][InPartOne]tx ack to peerrank[%u] failed", peerRank), ret);
            ret = links[peerRank]->RxAck(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Reduce][InPartOne]rx ack from peerank[%u] failed", peerRank), ret);
            //  发送到对端的output
            HCCL_DEBUG("rank[%u] sends inputMem[%p] to PeerRank[%u] Offset[%llu], Size[%llu]", \
                rank, inputMem_.ptr(), peerRank, baseOffset_, inputMem_.size());
            ret = senderInfo_->run(links[peerRank], baseOffset_, inputMem_, stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Reduce][InPartOne]tx sync to peerank[%u] failed", peerRank), ret);
            ret = links[peerRank]->RxAsync(UserMemType::OUTPUT_MEM, baseOffset_, inputMem_.ptr(), 0, stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[ExecutorBase][ExecuteTxSync]ExecuteTxSync: rx async size[%llu] failed", 0), ret);
            ret = links[peerRank]->DataReceivedAck(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[ExecutorBase][ExecuteTxSync]ExecuteTxSync: data received ack failed"), ret);
            ret = links[peerRank]->TxWaitDone(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Reduce][InPartOne]TxWaitDone failed"), ret);
        }
    }
    return HCCL_SUCCESS;
}


HcclResult ReduceRecursiveHalvingDoubling::ReduceScatterInBlock(u32 rank, u32 rankSize,
    const std::vector<LINK> &links)
{
    u32 rankInBlock = 0;

    u32 rootFlag = (root_ >= part1Size_) ? 0 : root_;
    // 需要根据root判断，让root节点必然参加reducescatter,在第一部分的rank若与root奇偶性不同，直接返回
    if (rank < part1Size_ && (rank % 2) != (rootFlag % 2)) {     // 模2判断奇偶性，本rank处于第一部分，奇偶性与root不同
        return HCCL_SUCCESS;
    } else if (rank < part1Size_) {     // 模2判断奇偶性，本rank 处于第一部分，奇偶性与root相同
        rankInBlock = rank / 2;                            // 除2计算block内的rank值
    } else {           // 本rank不属于第一部分
        rankInBlock = rank - part1Size_ / 2;               // 除2计算block内的part1的范围
    }
    // 直接调用block的reducscatterhd算法
    ReduceScatterHalvingDoubling executor(blockSize_, dispatcher_, reduceAttr,
        UserMemType::INPUT_MEM, UserMemType::OUTPUT_MEM);
    CHK_RET(executor.Prepare(inputMem_, outputMem_, outputMem_, count_, dataType_, stream_,
        reductionOp_, -1, slices_, baseOffset_));

    CHK_RET(executor.RegisterProfiler(profilerInput_.planeID, profilerInput_.stage, profilerInput_.step,
        stream_));

    // 重新建立reducscatterscatter需要的链接
    std::vector<LINK> subLinks;
    CHK_RET(BuildRootSubLinks(links, subLinks, rankSize));

    CHK_PRT_RET(subLinks.size() == 0, HCCL_ERROR("[ReduceRecursiveHalvingDoubling][ReduceScatterInBlock]rank[%u] "\
        "BuildSubLinks failed", rank), HCCL_E_PARA);

    CHK_RET(executor.RunAsync(rankInBlock, blockSize_, subLinks));

    return HCCL_SUCCESS;
}

HcclResult ReduceRecursiveHalvingDoubling::CalculateStepSlices(const std::vector<Slice> &inputSlices, u32 stepNum,
                                                               u32 rank, SliceType type, std::vector<Slice> &sliceOut)
{
    std::vector<Slice> slice(stepNum);

    for (u32 step = 0; step < stepNum; step++) {
        // all-gather操作, halving_bitmask从低往高循环, size倍增
        u32 halvingBitmask = (1 << step);
        u32 peerRank = rank ^ halvingBitmask;

        // 计算tx_slice/rx_slice
        u32 sliceId = (type == SliceType::SLICE_TYPE_RX) ? \
            (peerRank & (~(halvingBitmask - 1))) : (rank & (~(halvingBitmask - 1)));

        slice[step].offset = inputSlices[sliceId].offset;
        CHK_RET(Sum(inputSlices, sliceId, halvingBitmask, slice[step].size));

        HCCL_DEBUG("Slice Info: rank[%u], slices[%u].offset=%llu, slices[%u].size=%llu", \
                   rank, step, slice[step].offset, step, slice[step].size);
    }

    sliceOut = std::move(slice);
    return HCCL_SUCCESS;
}
HcclResult ReduceRecursiveHalvingDoubling::BuildRootSubLinks(const std::vector<LINK> &links,
                                                             std::vector<LINK> &subLinks, u32 rankSize) const
{
    std::vector<LINK>::const_iterator iter = links.begin();
    subLinks.resize(blockSize_);
    u32 rootFlag = (root_ >= part1Size_) ? 0 : root_;
    for (u32 i = 0; i < rankSize; i++) {
        if (i < part1Size_ && (i % 2) != rootFlag % 2) {  // 模2与root模2比较代表当前rank在part1的内且与root奇偶性不同，不参与block内的建链
            continue;
        } else if (i < part1Size_) {
            std::vector<LINK>::const_iterator niter = std::next(iter, i);
            if (niter != links.end()) {
                subLinks[i / 2] = *niter;              // 除2计算出在block内的rank号
            }
        } else {
            std::vector<LINK>::const_iterator niter = std::next(iter, i);
            if (niter != links.end()) {
                subLinks[i - part1Size_ / 2] = *niter; // rank在part2中，用原始rank减part1除2，计算出在block内的rank号
            }
        }
    }

    return HCCL_SUCCESS;
}
// 结果在output中，直接使用oupt进行数据收发
HcclResult ReduceRecursiveHalvingDoubling::GatherInBlock(u32 rank, u32 rankSize,
                                                         const std::vector<LINK> &links)
{
    u32 rankInBlock = 0;

    u32 rootFlag = (root_ >= part1Size_) ? 0 : root_;
    if (rank < part1Size_ && (rank % 2) != (rootFlag % 2)) {    // 模2判断奇偶性，本rank 处于第一部分，并且和root rank奇偶不同
        return HCCL_SUCCESS;
    } else if (rank < part1Size_) { // 模2判断奇偶性，本rank 处于第一部分，并且奇偶性和root相同
        rankInBlock = rank / 2;                        // 在block内的rank为实际rank除以2
    } else {
        rankInBlock = rank - part1Size_ / 2;           // 除2计算block内的part1的范围
    }
    u32 rootInBlock = (root_ > part1Size_) ? (root_ - part1Size_ / 2) : (root_ / 2);
    // 重新建立gather需要的链接
    std::vector<LINK> subLinks;

    CHK_RET(BuildRootSubLinks(links, subLinks, rankSize));

    CHK_PRT_RET(subLinks.size() == 0,
        HCCL_ERROR("[Gather][InBlock]rank[%u] build sub links failed", rank), HCCL_E_PARA);

    CHK_RET(CalculateStepSlices(slices_, round_, rankInBlock, SliceType::SLICE_TYPE_TX, txSlices_));

    CHK_RET(CalculateStepSlices(slices_, round_, rankInBlock, SliceType::SLICE_TYPE_RX, rxSlices_));

    for (u32 step = 0; step < round_; step++) {
        u32 peerRankBitmask = (1 << step);
        u32 opBitmask = peerRankBitmask - 1 ; // 判断本轮是否进行收发
        // 断rank是否和root在同一轮次接收发送的block内，第一轮为total，第二轮为1/2，第三轮为1/4....
        if ((step != 0) && ((rankInBlock & opBitmask) != (rootInBlock & opBitmask))) {
            return HCCL_SUCCESS; // rank在本轮同root不在一个操作块内，不操作，直接返回
        }
        u32 peerRank = rankInBlock ^ peerRankBitmask;
        CHK_SMART_PTR_NULL(subLinks[peerRank]);
        // 再次判断是否和root在同一1/2，1/4,用来判断数据是收还是发
        if ((rankInBlock & peerRankBitmask) == (rootInBlock & peerRankBitmask)) {
            DeviceMem rxMem = outputMem_.range(rxSlices_[step].offset, rxSlices_[step].size);
                HcclResult ret = subLinks[peerRank]->TxAck(stream_);
                CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Gather][InBlock]rank[%u] tx ack from peerank[%u] failed",
                    rank, peerRank), ret);
                ret = subLinks[peerRank]->RxAck(stream_);
                CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Gather][InBlock]rank[%u] rx ack from peerank[%u] failed",
                    rank, peerRank), ret);

                // 等待对端可以接收数据
                HCCL_DEBUG("rank[%u] outputMem[%p] recieve from PeerRank[%u] outputMem, Offset[%llu], "\
                           "Size[%llu]", rank, outputMem_.ptr(), peerRank,
                           baseOffset_ + rxSlices_[step].offset, rxSlices_[step].size);

                ret = ExecuteRxSync(subLinks[peerRank], UserMemType::OUTPUT_MEM, baseOffset_ + rxSlices_[step].offset,
                    rxMem.ptr(), rxSlices_[step].size, stream_);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[Gather][InBlock]rank[%u] rx sync from PeerRank[%u] failed", rank, peerRank), ret);
                ret = subLinks[peerRank]->RxWaitDone(stream_);
                CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Reduce][InPartOne]RxWaitDone failed"), ret);
        } else {
            DeviceMem txMem = outputMem_.range(txSlices_[step].offset, txSlices_[step].size);
                HcclResult ret = subLinks[peerRank]->TxAck(stream_);
                CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Gather][InBlock]rank[%u] tx ack from peerank[%u] failed",
                    rank, peerRank), ret);
                ret = subLinks[peerRank]->RxAck(stream_);
                CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Gather][InBlock]rank[%u] rx ack from peerank[%u] failed",
                    rank, peerRank), ret);
                HCCL_DEBUG("rank[%u] outputMem[%p] sends to peerrank[%u] outputmem, offset[%llu], " \
                           "size[%llu]", rank, outputMem_.ptr(), peerRank,
                           baseOffset_ + txSlices_[step].offset, txSlices_[step].size);
                ret = ExecuteTxSync(subLinks[peerRank], UserMemType::OUTPUT_MEM, baseOffset_ + txSlices_[step].offset,
                    txMem.ptr(), txSlices_[step].size, stream_);
                CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Gather][InBlock]rank[%u] tx sync to PeerRank[%u] failed",
                    rank, peerRank), ret);
                ret = subLinks[peerRank]->TxWaitDone(stream_);
                CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Reduce][InPartOne]TxWaitDone failed"), ret);
        }
    }

    return HCCL_SUCCESS;
}
}  // namespace hccl
