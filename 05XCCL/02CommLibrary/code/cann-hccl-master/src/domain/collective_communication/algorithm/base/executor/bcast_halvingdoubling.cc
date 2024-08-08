/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bcast_halvingdoubling.h"

namespace hccl {
BcastHalvingDoubling::BcastHalvingDoubling(const HcclDispatcher dispatcher)
    : ExecutorBase(dispatcher), rootBlockOffset_(0), rootBlockSize_(0),
      stepsInBlock_(0), myBlockSize_(0), rankInMyBlock_(0), myBlockOffset_(0),
      higherBlockSize_(0), lowerBlockSize_(0), highestStep_(0)
{
}

BcastHalvingDoubling::~BcastHalvingDoubling()
{
}

void BcastHalvingDoubling::PrepareBlockParams(const u32 rank, const u32 rankSize)
{
    u32 offset = rankSize;
    u32 blockSize = 1;
    u32 currentBlockSize = 0;
    u32 preBlockSize = 0;

    while (offset > 0) {
        if ((rankSize & blockSize) != 0) {
            preBlockSize = currentBlockSize;
            currentBlockSize = blockSize;
            if (blockSize != 0) {
                offset -= blockSize;
            }

            if (myBlockSize_ != 0) {
                higherBlockSize_ = currentBlockSize;
                break;
            }

            if (offset <= rank) {
                myBlockOffset_ = offset;
                myBlockSize_ = currentBlockSize;
                lowerBlockSize_ = preBlockSize;
            }
        }

        blockSize <<= 1;
    }

    stepsInBlock_ = SalLog2(myBlockSize_);
    highestStep_ = SalLog2(rankSize);

    if (myBlockSize_ != 0) {
        rankInMyBlock_ = rank % myBlockSize_;
    }
}

void BcastHalvingDoubling::PrepareRootBlockParam(const u32 rootRank, const u32 rankSize)
{
    u32 offset = rankSize;
    u32 blockSize = 1;
    u32 currentBlockSize = 0;

    while (offset > 0) {
        if ((rankSize & blockSize) != 0) {
            currentBlockSize = blockSize;
            if (blockSize != 0) {
                offset -= blockSize;
            }

            if (rootBlockSize_ != 0) {
                break;
            }

            if (offset <= rootRank) {
                rootBlockOffset_ = offset;
                rootBlockSize_ = currentBlockSize;
            }
        }

        blockSize <<= 1;
    }
}

//  binaryblock hd的broadcast算法入口函数
HcclResult BcastHalvingDoubling::RunAsync(const u32 rank, const u32 rankSize,
                                          const std::vector<std::shared_ptr<Transport> > &links)
{
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    if (!inputMem_) {
        HCCL_ERROR("[BcastHalvingDoubling][RunAsync]rank[%u] run_async inputmem is null", rank);
        return HCCL_E_PTR;
    }
    HCCL_INFO("BcastHalvingDoubling run: rank[%u] rootRank[%u] totalrank[%u] \
        inputMem[%p] outputMem[%p] count[%llu]", \
              rank, root_, rankSize, inputMem_.ptr(), outputMem_.ptr(), count_);

    if (rankSize == 1) {
        return HCCL_SUCCESS;
    }
    // 计算halvingdouble运算过程中参数
    PrepareBlockParams(rank, rankSize);

    // 计算root 所在block的偏移，大小
    PrepareRootBlockParam(root_, rankSize);
    HCCL_DEBUG(\
        "rank:[%u] start ranksize[%u] root[%u] root_block_offset[%u] rootBlockSize_[%u] \
        my_block_offset[%u] my_block_size[%u] lower_block_size[%u]", \
        rank, rankSize, root_, rootBlockOffset_, rootBlockSize_, myBlockOffset_, myBlockSize_, \
        lowerBlockSize_);

    bool bRetSize = (links.size() < rankSize);
    CHK_PRT_RET(bRetSize, HCCL_ERROR("[BcastHalvingDoubling][RunAsync]rank[%u] linksize[%llu] is error", rank,
        links.size()), HCCL_E_INTERNAL);

    // 本rank 所在block低于root所在block
    if (rank >= rootBlockOffset_ + rootBlockSize_) {
        HCCL_DEBUG("rank[%u] is lower than root[%u]'s block ,root_block is from rank[%u] blocksize[%u]", \
                   rank, root_, rootBlockOffset_, rootBlockSize_);
        CHK_RET(GatherBetweenBlocksLowerBlock(rank, rankSize, links));
    } else if (rank < rootBlockOffset_) { // 本rank所在block高于root所在block
        HCCL_DEBUG("rank[%u] is higher than root[%u]'s block ,root_block is from rank[%u] rootblocksize[%u]", \
                   rank, root_, rootBlockOffset_, rootBlockSize_);
        CHK_RET(GatherBetweenBlocksHigherBlock(rank, rankSize, links));
    } else { // 与root所在同一个block内，先进行block内的bcast，再进行block间操作
        HCCL_DEBUG("rank[%u] is in the same block with root's bolck ,block_offset[%u]", \
                   rank, rootBlockOffset_);

        // 先进行block内部的bcast操作
        CHK_RET(BcastInRootBlock(rank, rankSize, links));
        // 再进行block间的bcast操作
        CHK_RET(GatherBetweenBlocksRootBlock(rank, rankSize, links));
    }

    HCCL_INFO("BcastHalvingDoubling finished: rank[%u], finished", rank);
    return HCCL_SUCCESS;
}


HcclResult BcastHalvingDoubling::ReceiveData(const std::shared_ptr<Transport> &link, u64 dataCount)
{
    // 向目的rank发送同步
    HcclResult ret = link->TxAck(stream_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Receive][Data]link tx ack failed"), ret);
    ret = link->RxAck(stream_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Receive][Data]link rx ack failed"), ret);

    // 从对端接收数据
    DeviceMem rcvMem = inputMem_.range(baseOffset_, DataSize(dataCount));

    ret = ExecuteRxSync(link, UserMemType::INPUT_MEM, baseOffset_, rcvMem.ptr(), DataSize(dataCount), stream_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Receive][Data]link rx sync with rcvMem[%p] input's offset[%llu] "\
        "size[%llu] failed", rcvMem.ptr(), baseOffset_, DataSize(dataCount)), ret);

    return HCCL_SUCCESS;
}

HcclResult BcastHalvingDoubling::SendData(const std::shared_ptr<Transport> &link, u64 dataCount)
{
    //  等待接收信号，以进行下一轮发送
    HcclResult ret = link->TxAck(stream_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Send][Data]tx ack failed"), ret);
    ret = link->RxAck(stream_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Send][Data]rx ack failed"), ret);

    //  从本端的Inputmem发送数据给对端的Inputmem
    DeviceMem sendMem = inputMem_.range(baseOffset_, DataSize(dataCount));

    ret = ExecuteTxSync(link, UserMemType::INPUT_MEM, baseOffset_, sendMem.ptr(), DataSize(dataCount), stream_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Send][Data]tx sync sendMem[%p] input's offset[%llu] size[%llu] "\
        "failed", sendMem.ptr(), baseOffset_, DataSize(dataCount)), ret);

    return HCCL_SUCCESS;
}

// root 所在block内的bcast操作
HcclResult BcastHalvingDoubling::BcastInRootBlock(const u32 rank, const u32 rankSize,
                                                  const std::vector<std::shared_ptr<Transport> > &links)
{
    (void)rankSize;
    // block内一个rank时，不操作
    if (myBlockSize_ == 1) {
        return HCCL_SUCCESS;
    }

    u32 rootOffset = root_ - rootBlockOffset_;
    // steps_in_block=0，my_blocksize已经为0, 此时必然不小于1
    CHK_PRT_RET(stepsInBlock_ < 1, HCCL_ERROR("[BcastHalvingDoubling][BcastInRootBlock]rank[%u] para "\
        "error", rank), HCCL_E_INTERNAL);

    u32 destRankBitmask = (1 << (stepsInBlock_ - 1));
    u32 sendFlag = 0;

    for (u32 round = 1; round <= stepsInBlock_; round++) {
        u32 destRank = rank ^ destRankBitmask;
        u32 andOprand = (1 << (stepsInBlock_ - round)) - 1;

        HCCL_INFO("rank[%u] round[%u] rankInMyBlock_[%u] dstrank[%u] and_oprand[%u] root_offset[%u] ", \
            rank, round, rankInMyBlock_, destRank, andOprand, rootOffset);

        // 本rank在第round轮需要接收数据
        if (((rankInMyBlock_ & andOprand) == (rootOffset & andOprand)) && (rank != root_) && sendFlag == 0) {
            std::vector<std::shared_ptr<Transport> >::const_iterator niter = std::next(links.begin(), destRank);
            if (niter != links.end()) {
                HCCL_DEBUG("rank[%u] receive memsize[%llu] from rank[%u] in round[%u]",  \
                           rank, DataSize(count_), destRank, round);
                HcclResult ret = ReceiveData(*niter, count_);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[BcastHalvingDoubling][BcastInRootBlock]rank[%u] receive mem failed in "\
                        "round[%u]", rank, round), ret);
                ret = (*niter)->RxWaitDone(stream_);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[BcastHalvingDoubling][BcastInRootBlock]RxWaitDone failed"), ret);
            }
            // 标记数据已经接收，后面的轮次不用再接收
            sendFlag = 1;
            destRankBitmask >>= 1;
            continue;
        }

        if (sendFlag == 1 || rank == root_) {
            std::vector<std::shared_ptr<Transport> >::const_iterator niter = std::next(links.begin(), destRank);
            if (niter != links.end()) {
                HCCL_DEBUG("rank[%u] send size[%llu] to rank[%u] in round[%u]", \
                           rank, DataSize(count_), destRank, round);
                HcclResult ret = SendData(*niter, count_);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[BcastHalvingDoubling][BcastInRootBlock]rank[%u] send mem failed in "\
                        "round[%u]", rank, round), ret);
                ret = (*niter)->TxWaitDone(stream_);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[BcastHalvingDoubling][BcastInRootBlock]TxWaitDone failed"), ret);
            }
        }

        destRankBitmask >>= 1;
    }
    return HCCL_SUCCESS;
}

// root block内所有rank的处理过程，向低阶发和向高阶发
HcclResult BcastHalvingDoubling::GatherBetweenBlocksRootBlock(const u32 rank, const u32 rankSize,
    const std::vector<std::shared_ptr<Transport> > &links)
{
    (void)rankSize;
    // 本rank和root 在同一block内部，数据向高阶和低阶发送
    u32 dstRank = 0;
    // 低阶不为空，向低阶发送数据,block内rank较小的去给低阶发
    if (lowerBlockSize_ != 0) {
        u32 myBlockLowerBlockTimes = myBlockSize_ / lowerBlockSize_;
        // 该flag为0代表本block内rank偏移是低阶blocksize的整数倍，需要发送数据
        u32 useMyRankFlag = rankInMyBlock_ % myBlockLowerBlockTimes;

        if (useMyRankFlag == 0) {
            dstRank = myBlockOffset_ + myBlockSize_ + rankInMyBlock_ / myBlockLowerBlockTimes;
            std::vector<std::shared_ptr<Transport> >::const_iterator niter = std::next(links.begin(), dstRank);
            if (niter != links.end()) {
                HCCL_DEBUG("rank[%u] send memsize[%llu] to rank[%u]", \
                           rank, DataSize(count_), dstRank);
                CHK_RET(SendData(*niter, count_));
                HcclResult ret = (*niter)->TxWaitDone(stream_);
                CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("TxWaitDone failed"), ret);
            }
        }
    }

    // 每个rank都是需要向高阶多个rank发送数据
    if (higherBlockSize_ != 0) {
        u32 sendRounds = higherBlockSize_ / myBlockSize_;

        // 找到向高阶发的第一次目的rank, my_block_offset_必然不小于higher_block_size_
        dstRank = (myBlockOffset_ - higherBlockSize_) + rankInMyBlock_ * sendRounds;

        for (u32 round = 0; round < sendRounds; round++) {
            std::vector<std::shared_ptr<Transport> >::const_iterator niter = std::next(links.begin(), dstRank);
            if (niter != links.end()) {
                HCCL_DEBUG("rank[%u] send memsize[%llu] to rank[%u]", \
                           rank, DataSize(count_), dstRank);
                HcclResult ret = SendData(*niter, count_);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[Gather][BetweenBlocksRootBlock]rank[%u] send mem to dstrank[%u] failed",
                        rank, dstRank), ret);
                ret = (*niter)->TxWaitDone(stream_);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[Gather][BetweenBlocksRootBlock]TxWaitDone failed"), ret);
            }
            dstRank += 1;
        }
    }

    return HCCL_SUCCESS;
}

// rank 所在block高于root所在block，从低阶收数据，向高阶发数据
HcclResult BcastHalvingDoubling::GatherBetweenBlocksHigherBlock(const u32 rank, const u32 rankSize,
    const std::vector<std::shared_ptr<Transport> > &links)
{
    (void)rankSize;
    // 从低阶接收数据， 本rank又比root所在rank高，那么本rank必然有低阶
    if (lowerBlockSize_ != 0) {
        u32 myBlockLowerBlockTimes = myBlockSize_ / lowerBlockSize_;
        u32 rcvRank = (myBlockOffset_ + myBlockSize_ + rankInMyBlock_ / myBlockLowerBlockTimes);

        std::vector<std::shared_ptr<Transport> >::const_iterator niter = std::next(links.begin(), rcvRank);
        if (niter != links.end()) {
            HCCL_DEBUG("rank[%u] receive memsize[%llu] from rank[%u]", \
                       rank, DataSize(count_), rcvRank);
            HcclResult ret = ReceiveData(*niter, count_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Gather][BetweenBlocksHigherBlock]rank[%u]  receive mem from rank[%u] failed",
                    rank, rcvRank), ret);
            ret = (*niter)->RxWaitDone(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Gather][BetweenBlocksRootBlock]RxWaitDone failed"), ret);
        }
    }

    // 如果高阶存在，向高阶发送数据
    if (higherBlockSize_ != 0) {
        // 计算与高阶发送数据的dst_rank_
        u32 dstRankTotal = higherBlockSize_ / myBlockSize_;
        u32 dstRank = (myBlockOffset_ - higherBlockSize_) + rankInMyBlock_ * dstRankTotal;
        for (u32 i = 0; i < dstRankTotal; i++) {
            // 获取与目的rank的link信息
            std::vector<std::shared_ptr<Transport> >::const_iterator niter = std::next(links.begin(), dstRank);
            if (niter != links.end()) {
                HCCL_DEBUG("rank[%u] send mem[%llu] to rank[%u]", \
                           rank, DataSize(count_), dstRank);
                HcclResult ret = SendData(*niter, count_);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[Gather][BetweenBlocksHigherBlock]rank[%u] send mem to dstrank[%u] "\
                        "failed", rank, dstRank), ret);
                ret = (*niter)->TxWaitDone(stream_);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[Gather][BetweenBlocksHigherBlock]TxWaitDone failed"), ret);
            }
            dstRank += 1;
        }
    }

    return HCCL_SUCCESS;
}

// 本rank所在block 低于root所在block
HcclResult BcastHalvingDoubling::GatherBetweenBlocksLowerBlock(const u32 rank, const u32 rankSize,
    const std::vector<std::shared_ptr<Transport> > &links)
{
    (void)rankSize;
    // 从高阶接收数据
    if (higherBlockSize_ != 0) {
        u32 higherBlockMyBlockTimes = higherBlockSize_ / myBlockSize_;
        u32 rcvRank = (myBlockOffset_ - higherBlockSize_) + rankInMyBlock_ * higherBlockMyBlockTimes;

        std::vector<std::shared_ptr<Transport> >::const_iterator niter = std::next(links.begin(), rcvRank);
        // 从对端接收数据
        if (niter != links.end()) {
            HCCL_DEBUG("rank[%u] receive memsize[%llu] from rank[%u]", \
                       rank, DataSize(count_), rcvRank);
            CHK_RET(ReceiveData(*niter, count_));
            HcclResult ret = (*niter)->RxWaitDone(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Gather][BetweenBlocksRootBlock]RxWaitDone failed"), ret);
        }
    }

    // 如果存在低阶，且rank数处于本block内的前n个，n为低阶blocksizie，向低阶发送
    if (lowerBlockSize_ != 0) {
        u32 myBlockLowerBlockTimes = myBlockSize_ / lowerBlockSize_;
        u32 useMyRankFlag = rankInMyBlock_ % myBlockLowerBlockTimes;

        if (useMyRankFlag == 0) {
            u32 dstRank = myBlockOffset_ + myBlockSize_ + (rankInMyBlock_ / myBlockLowerBlockTimes);
            // 获取与目的rank的link信息
            std::vector<std::shared_ptr<Transport> >::const_iterator niter = std::next(links.begin(), dstRank);
            if (niter != links.end()) {
                HCCL_DEBUG("rank[%u] send mem[%llu] to rank[%u]", \
                           rank, DataSize(count_), dstRank);
                HcclResult ret = SendData(*niter, count_);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[Gather][BetweenBlocksLowerBlock]rank[%u] send mem to dstrank[%u] failed",
                        rank, dstRank), ret);
                ret = (*niter)->TxWaitDone(stream_);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[Gather][BetweenBlocksLowerBlock]TxWaitDone failed"), ret);
            }
        }
    }

    return HCCL_SUCCESS;
}
}  // namespace hccl
