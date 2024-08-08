/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bcast_recursive_halvingdoubling.h"
#include <cmath>

namespace hccl {
BcastRecursiveHalvingDoubling::BcastRecursiveHalvingDoubling(const HcclDispatcher dispatcher)
    : RecursiveHalvingDoublingBase(dispatcher), hasData_(false)
{
}

BcastRecursiveHalvingDoubling::~BcastRecursiveHalvingDoubling()
{
}

// recursiveHD broadcast算法主入口
HcclResult BcastRecursiveHalvingDoubling::RunAsync(const u32 rank, const u32 rankSize,
                                                   const std::vector<std::shared_ptr<Transport> > &links)
{
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    CHK_PRT_RET(!inputMem_, HCCL_ERROR("[BcastRecursiveHalvingDoubling][RunAsync]rank[%u] run_async inputmem is null",
        rank), HCCL_E_PTR);

    HCCL_INFO("BcastRecursiveHalvingDoubling run: rank[%u] rootRank[%u] totalrank[%u] \
        inputMem[%p] outputMem[%p] count[%llu]", \
              rank, root_, rankSize, inputMem_.ptr(), outputMem_.ptr(), count_);

    if (rankSize == 1) {
        return HCCL_SUCCESS;
    }

    if (rank == root_) {
        hasData_ = true;
    }

    CHK_PRT_RET(links.size() < rankSize,
        HCCL_ERROR("[BcastRecursiveHalvingDoubling][RunAsync]rank[%u] linksize[%llu] is less than rankSize[%u]",
        rank, links.size(), rankSize), HCCL_E_INTERNAL);

    // 计算recursive算法第一部分相关参数
    HcclResult ret = CalcPartOneSizeAndBlockSize(rankSize);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[BcastRecursiveHalvingDoubling][RunAsync]rank[%u] Calculate "\
        "Par1Size[%u] And BlockSize[%u] Failed! rankSize[%u]", rank, part1Size_, blockSize_,
        rankSize), ret);

    HCCL_DEBUG("rank[%u] BroadcastInBlock... blockSize_[%u], part1Size_[%u]", \
        rank, blockSize_, part1Size_);

    // 先进行block内部的bcast
    CHK_RET(BroadcastInBlock(rank, links));

    HCCL_DEBUG("rank[%u] BroadcastOutOfBlock", rank);

    if (rank < part1Size_ && (rank % 2 == 0)) { // 模2是否为0判断rank奇偶性
        CHK_RET(EvenNumberRankProcess(rank, links));
    } else if (rank < part1Size_ && (rank % 2 == 1)) { // 模2是否为1判断rank奇偶性
        CHK_RET(OddNumberRankProcess(rank, links));
    }

    HCCL_INFO("BcastRecursiveHalvingDoubling finished: rank[%u] finished", rank);
    return HCCL_SUCCESS;
}

HcclResult BcastRecursiveHalvingDoubling::ReceiveData(const u32 destRank,
                                                      const std::vector<std::shared_ptr<Transport> > &links)
{
    if (destRank < links.size()) {
        if (links[destRank] == nullptr) {
            HCCL_ERROR("[Receive][Data]errNo[0x%016llx] links[destRank[%u]] ptr is NULL, return HCCL_E_PTR",
                       HCCL_ERROR_CODE(HCCL_E_PTR), destRank);
            return HCCL_E_PTR;
        }
        HcclResult ret = links[destRank]->TxAck(stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Receive][Data]tx ack to dstrank[%u] failed", destRank), ret);
        ret = links[destRank]->RxAck(stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Receive][Data]rx ack to dstrank[%u] failed", destRank), ret);

        u64 dataBytes = count_ * DataUnitSize(dataType_);
        DeviceMem rcvMem = inputMem_.range(baseOffset_, dataBytes);
        HCCL_DEBUG("rx async from dstrank[%u] with rcvMem[%p] inputmem's offset[%llu] size[%llu]", \
            destRank, rcvMem.ptr(), baseOffset_, dataBytes);

        ret = ExecuteRxSync(links[destRank], UserMemType::INPUT_MEM, baseOffset_, rcvMem.ptr(), dataBytes, stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Receive][Data]rx sync from rank[%u] failed",
            destRank), ret);
    }
    return HCCL_SUCCESS;
}

HcclResult BcastRecursiveHalvingDoubling::SendData(const u32 destRank,
                                                   const std::vector<std::shared_ptr<Transport> > &links)
{
    if (destRank < links.size()) {
        if (links[destRank] == nullptr) {
            HCCL_ERROR("[Send][Data]errNo[0x%016llx] links[destRank[%u]] ptr is NULL, return HCCL_E_PTR",
                       HCCL_ERROR_CODE(HCCL_E_PTR), destRank);
            return HCCL_E_PTR;
        }
        HcclResult ret = links[destRank]->TxAck(stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Send][Data]tx ack from rank[%u] failed", destRank), ret);
        ret = links[destRank]->RxAck(stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Send][Data]rx ack from rank[%u] failed", destRank), ret);

        u64 dataBytes = count_ * DataUnitSize(dataType_);
        DeviceMem sendMem = inputMem_.range(baseOffset_, dataBytes);
        HCCL_DEBUG("tx async to dstrank[%u] from sendMem[%p] inputmem's offset[%llu] size[%llu]", \
            destRank, sendMem.ptr(), baseOffset_, dataBytes);

        ret = ExecuteTxSync(links[destRank], UserMemType::INPUT_MEM, baseOffset_, sendMem.ptr(), dataBytes, stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Send][Data]tx sync to rank[%u] failed", destRank),
            ret);
    }
    return HCCL_SUCCESS;
}

u32 BcastRecursiveHalvingDoubling::GetRankIndexReal(const u32 rankInBlock) const
{
    u32 res = 0;
    /* 如果根节点在第一部分的偶数位置或其他部分 */
    if ((root_ < part1Size_ && (root_ % 2) == 0) || root_ >= part1Size_) { // 模2判断奇偶性
        if (rankInBlock < part1Size_ / 2) {                                // 除2计算block内part1的rank范围
            res = rankInBlock * 2;                                         // 乘2计算block内part1的rank范围
            return res;
        } else {
            res = part1Size_ / 2 + rankInBlock;                            // 除2加rankInBlock计算真实的rank值
            return res;
        }
    } else {
        if (rankInBlock < part1Size_ / 2) {                                // 除2计算block内part1的rank范围
            res = rankInBlock * 2 + 1;                                     // 乘2加1计算计算真实的rank值
            return res;
        } else {
            res = part1Size_ / 2 + rankInBlock;                            // 除2加rankInBlock计算真实的rank值
            return res;
        }
    }
}

u32 BcastRecursiveHalvingDoubling::GetRankIndexInBlock(const u32 rank) const
{
    // root在第一部分，并且root是偶数rank，或者root在第二部分
    if ((root_ < part1Size_ && (root_ % 2) == 0) || root_ >= part1Size_) { // 模2判断奇偶性
        // rank在第一部分，并且本rank是偶数rank，除以2就为在block内的index
        if (rank < part1Size_ && rank % 2 == 0) {                          // 模2判断奇偶性
            return rank / 2;                                               // 除2计算block内rank值
        } else if (rank < part1Size_ && rank % 2 == 1) {                   // 模2判断奇偶性，奇数的话不在block内
            return INVALID_VALUE_RANKID;
        } else {
            return rank - part1Size_ / 2;                                  // 除2计算block内part1的rank范围
        }
    } else { // root在第一部分属于奇数rank
        if (rank < part1Size_ && rank % 2 == 0) {                          // 模2判断奇偶性，偶数rank不在block内
            return INVALID_VALUE_RANKID;
        } else if (rank < part1Size_ && rank % 2 == 1) {                   // 模2判断奇偶性，为奇计算block内rank号
            return (rank - 1) / 2;                                         // 通过减1再除2得到block内rank号
        } else {
            return rank - part1Size_ / 2;                                  // 除2计算block内part1的rank范围
        }
    }
}

HcclResult BcastRecursiveHalvingDoubling::BroadcastInBlock(const u32 rank,
    const std::vector<std::shared_ptr<Transport> > &links)
{
    u32 rankInBlock = GetRankIndexInBlock(rank);
    if (rankInBlock == INVALID_VALUE_RANKID) { // 非block内的节点，不做操作
        return HCCL_SUCCESS;
    }

    u32 rootInBlock = GetRankIndexInBlock(root_);
    for (u32 i = 0; i < round_; i++) {
        u32 peerRankBitmask = 1 << (round_ - i - 1); // 进入此条件，round必然不小于1
        u32 peerRankInBlock = rankInBlock ^ peerRankBitmask;
        u32 andOprand = (1 << (round_ - i - 1)) - 1;
        u32 peerRankReal = GetRankIndexReal(peerRankInBlock);

        HcclResult ret = HCCL_SUCCESS;
        // 本rank在第round轮需要接收数据
        if (((rankInBlock & andOprand) == (rootInBlock & andOprand)) && (rank != root_) &&
                !hasData_) {
            HCCL_DEBUG("rank[%u] receive memsize[%llu] from rank[%u] in round[%u]",  \
                rank, DataUnitSize(dataType_)*count_, peerRankReal, i);
            ret = ReceiveData(peerRankReal, links);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[BcastRecursiveHalvingDoubling][BroadcastInBlock]rank[%u] "\
                "Receive Data from rank[%u] failed.", rank, peerRankReal), ret);
            hasData_ = true;
            if (peerRankReal < links.size()) {
                ret = links[peerRankReal]->RxWaitDone(stream_);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[BcastRecursiveHalvingDoubling][BroadcastInBlock]RxWaitDone failed"), ret);
            }
            continue;
        }

        // 需要向目的rank发送数据，前提是收到数据后(root 节点每轮都发)
        if (hasData_) {
            HCCL_DEBUG("rank[%u] send mem[%llu] to rank[%u] in round:%u", \
                       rank, DataUnitSize(dataType_)*count_, peerRankReal, i);
            ret = SendData(peerRankReal, links);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[BcastRecursiveHalvingDoubling][BroadcastInBlock]rank[%u] Send "\
                    "Data to rank[%u] failed.", rank, peerRankReal), ret);
        }
        if (peerRankReal < links.size()) {
            ret = links[peerRankReal]->TxWaitDone(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[BcastRecursiveHalvingDoubling][BroadcastInBlock]TxWaitDone failed"), ret);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult BcastRecursiveHalvingDoubling::EvenNumberRankProcess(const u32 rank,
    const std::vector<std::shared_ptr<Transport> > &links)
{
    HcclResult ret;
    if (root_ % 2 == 0 || root_ >= part1Size_) { // 模2是否为0判断rank_奇偶性
        HCCL_DEBUG("rank[%u] stream[%p] send memsize[%llu] to rank[%u]",  \
            rank, stream_.ptr(), DataUnitSize(dataType_)*count_, rank + 1);
        // 该rank需要向第一部分的后续奇数rank发送数据
        ret = SendData(rank + 1, links);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[BcastRecursiveHalvingDoubling][RunAsync]rank[%u] stream[%p] Send data to "\
                "Rank[%u] failed", rank, stream_.ptr(), rank + 1), ret);
        if (rank + 1 < links.size()) {
            ret = links[rank + 1]->TxWaitDone(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[BcastRecursiveHalvingDoubling][RunAsync]TxWaitDone failed"), ret);
        }
    } else {
        HCCL_DEBUG("rank[%u] stream[%p] receive memsize[%llu] from rank[%u]",  \
            rank, stream_.ptr(), DataUnitSize(dataType_)*count_, rank + 1);

        // root为奇数，本rank为偶数，需要从邻接的奇数rank接收数据
        ret = ReceiveData(rank + 1, links);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[BcastRecursiveHalvingDoubling][RunAsync]rank[%u] stream[%p] Receive data "\
                "from Rank[%u] failed", rank, stream_.ptr(), rank + 1), ret);
        if (rank + 1 < links.size()) {
            ret = links[rank + 1]->RxWaitDone(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[BcastRecursiveHalvingDoubling][RunAsync]RxWaitDone failed"), ret);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult BcastRecursiveHalvingDoubling::OddNumberRankProcess(const u32 rank,
    const std::vector<std::shared_ptr<Transport> > &links)
{
    HcclResult ret;
    if (root_ % 2 == 0 || root_ >= part1Size_) {  // 模2是否为0判断rank_奇偶性
        HCCL_DEBUG("rank[%u]  stream[%p] receive memsize[%llu] from rank[%u]",  \
            rank, stream_.ptr(), DataUnitSize(dataType_)*count_, rank - 1);

        // root是偶数节点，rank从前面邻接的偶数节点接收数据
        ret = ReceiveData(rank - 1, links);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[BcastRecursiveHalvingDoubling][RunAsync]rank[%u] stream[%p] Receive data "\
                "from Rank[%u] failed", rank, stream_.ptr(), rank - 1), ret);
        if (rank - 1 < links.size()) {
            ret = links[rank - 1]->RxWaitDone(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[BcastRecursiveHalvingDoubling][RunAsync]RxWaitDone failed"), ret);
        }
    } else {
        HCCL_DEBUG("rank[%u] stream[%p] send memsize[%llu] to rank[%u]", \
            rank, stream_.ptr(), DataUnitSize(dataType_)*count_, rank - 1);
        ret = SendData(rank - 1, links);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[BcastRecursiveHalvingDoubling][RunAsync]rank[%u] stream[%p] Send data to "\
                "Rank[%u] failed", rank, stream_.ptr(), rank - 1), ret);
        if (rank - 1 < links.size()) {
            ret = links[rank - 1]->TxWaitDone(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[BcastRecursiveHalvingDoubling][RunAsync]TxWaitDone failed"), ret);
        }
    }
    return HCCL_SUCCESS;
}

}  // namespace hccl
