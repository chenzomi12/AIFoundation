/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "broadcast_star.h"

namespace hccl {
// Gather的入口函数
BroadcastStar::BroadcastStar(const HcclDispatcher dispatcher, u32 userRank)
    : ExecutorBase(dispatcher), userRank_(userRank)
{
}

BroadcastStar::~BroadcastStar()
{
}

HcclResult BroadcastStar::RunAsync(const u32 rank, const u32 rankSize,
    const std::vector<std::shared_ptr<Transport>> &links)
{
    // task下发接口
    CHK_SMART_PTR_NULL(dispatcher_);
    // ==1的处理
    if (rankSize == 1) {
        if (inputMem_ != outputMem_) {
            HcclResult ret = HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[BroadcastStar][RunAsync]rank[%u] copy input[%p] to output[%p] failed",
                rank, inputMem_.ptr(), outputMem_.ptr()), ret);
        }
        return HCCL_SUCCESS;
    }
    // links本rank_id与通信域内其它rank的通信连接,rankSize本executor所在通信域的rank个数
    if (links.size() < rankSize) {
        HCCL_ERROR("[BroadcastStar][RunAsync]rank[%u] linksize[%llu] is less than rankSize[%u]",
            rank, links.size(), rankSize);
        return HCCL_E_INTERNAL;
    }

    Slice sendSlice;
    sendSlice.offset = 0;
    sendSlice.size = dataBytes_;
    Slice recvSlice;
    recvSlice.offset = 0;
    recvSlice.size = dataBytes_;
    if (rank == root_) {
        // root 给其他rank发
        HcclResult ret;
        for (u32 dstRank = 0; dstRank < rankSize; dstRank++) {
            ret = RunSendBroadcast(dstRank, sendSlice, links);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[BroadcastStar][RunAsync] root [%u] send broadcast to"\
                "other rank[%u] run failed!", root_, dstRank), ret);
        }
    } else {
        // 非root 接收来自root的数据
        HcclResult ret = RunRecvBroadcast(root_, rank, recvSlice, links);
        CHK_PRT_RET(ret == HCCL_E_AGAIN,
            HCCL_WARNING("[BroadcastStar][RunAsync]group has been destroyed. Break!"), ret);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[BroadcastStar][RunAsync] dstrank [%u] recv broadcast from"\
            "root [%u] run failed!", rank, root_), ret);
    }
    HCCL_INFO("BroadBastStar finished: rank[%u], end", rank);
    return HCCL_SUCCESS;
}

HcclResult BroadcastStar::RunRecvBroadcast(const u32 srcRank, const u32 dstRank, const Slice &slice,
    const std::vector<LINK> &links)
{
    // 非root 接受数据
    DeviceMem dst;
    if (slice.size > 0) {
        if (srcRank >= links.size()) {
            HCCL_ERROR("[RunRecvBroadcast] root [%u] is out of range, linksize[%llu]", srcRank, links.size());
            return HCCL_E_INTERNAL;
        }
        dst = outputMem_.range(slice.offset, slice.size);
        HCCL_DEBUG("rank [%u] will recv with output's offset[%llu], size[%llu], dstmem[%p]", \
            dstRank, slice.offset, slice.size, dst.ptr());

        if (links[srcRank]->IsTransportRoce()) {
            CHK_RET(links[srcRank]->RxEnv(stream_));
        } else {
            CHK_RET(links[srcRank]->TxAck(stream_));
        }

        HcclResult ret = links[srcRank]->RxAsync(UserMemType::OUTPUT_MEM, slice.offset, dst.ptr(), slice.size, stream_);
        CHK_PRT_RET(ret == HCCL_E_AGAIN, HCCL_WARNING("[RunRecvBroadcast]group has been destroyed. Break!"), ret);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[RunRecvBroadcast]root rank[%u] rx async to dstrank[%u] run "\
            "failed", srcRank, dstRank), ret);

        if (!links[srcRank]->IsTransportRoce()) {
            ret = ExecuteBarrier(links[srcRank], stream_); // 多server走rdma可以不用
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[RunRecvBroadcast]dstRank[%u] Broadcast star run executor barrier failed", dstRank), ret);
            ret = links[srcRank]->RxWaitDone(stream_); // 多server走rdma可以不用
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[RunRecvBroadcast]RxWaitDone failed"), ret);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult BroadcastStar::RunSendBroadcast(const u32 dstRank, const Slice &slice,
    const std::vector<LINK> &links)
{
    DeviceMem src;
    // root发送数据
    if (slice.size > 0 && dstRank!= root_) {
        src = inputMem_.range(slice.offset, slice.size);

        if (links[dstRank]->IsTransportRoce()) {
            CHK_RET(links[dstRank]->TxEnv(src.ptr(), slice.size, stream_));
        } else {
            CHK_RET(links[dstRank]->RxAck(stream_));
        }

        HcclResult ret = links[dstRank]->TxAsync(UserMemType::OUTPUT_MEM, slice.offset, src.ptr(), slice.size, stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[RunSendBroadcast]rank[%u] tx async with output's offset[%llu] failed",
            dstRank, slice.offset), ret);

        if (!links[dstRank]->IsTransportRoce()) {
            ret = ExecuteBarrierSrcRank(links[dstRank], stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[RunSendBroadcast] srcRank[%u] broadcast star run executor barrier failed", root_), ret);

            ret = links[dstRank]->TxWaitDone(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[RunSendBroadcast]TxWaitDone failed"), ret);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult BroadcastStar::ExecuteBarrierSrcRank(std::shared_ptr<Transport> link, Stream &stream) const
{
    CHK_RET(link->RxAck(stream));

    CHK_RET(link->TxAck(stream));

    CHK_RET(link->RxDataSignal(stream));

    CHK_RET(link->TxDataSignal(stream));

    return HCCL_SUCCESS;
}
}