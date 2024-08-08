/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gather_star.h"

namespace hccl {
// Gather的入口函数
GatherStar::GatherStar(const HcclDispatcher dispatcher, u32 userRank)
    : ExecutorBase(dispatcher),
      userRank_(userRank)
{
}

GatherStar::~GatherStar()
{
}

HcclResult GatherStar::RunAsync(const u32 rank, const u32 rankSize,
    const std::vector<std::shared_ptr<Transport>> &links)
{
    HCCL_INFO("GatherStar rank[%u] root[%u] linksize[%u]", rank, root_, links.size());
    // task下发接口
    CHK_SMART_PTR_NULL(dispatcher_);
    // ==1的处理
    if (rankSize == 1) {
        if (inputMem_ != outputMem_) {
            HcclResult ret = HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[GatherStar][RunAsync]rank[%u] copy input[%p] to output[%p] "\
                "failed", rank, inputMem_.ptr(), outputMem_.ptr()), ret);
        }
        return HCCL_SUCCESS;
    }
    // links本rank_id与通信域内其它rank的通信连接,rankSize本executor所在通信域的rank个数
    if (links.size() < rankSize) {
        HCCL_ERROR("[GatherStar][RunAsync]rank[%u] linksize[%llu] is less than rankSize[%u]",
            rank, links.size(), rankSize);
        return HCCL_E_INTERNAL;
    }
    // 计算offset
    u32 unitSize = DataUnitSize(dataType_);
    if (unitSize == 0) {
        HCCL_ERROR("[GatherStar][RunAsync]rank[%u] unit data size is zero", rank);
        return HCCL_E_INTERNAL;
    }
    if (slices_.size() == 0) {
        PrepareSlicesData(unitSize, count_, rankSize);
    }

    Slice sendSlice;
    sendSlice.offset = dataBytes_ * rank;
    sendSlice.size = dataBytes_;
    Slice recvSlice;
    recvSlice.offset = 0;
    recvSlice.size = dataBytes_;
    if (rank == root_) {
        // root 从对端其他rank收
        for (u32 srcRank = 0; srcRank < rankSize; srcRank++) {
            HcclResult ret = RunRecvGather(srcRank, recvSlice, links);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[GatherStar][RunAsync] srcrank[%u] recv broadcast from"\
                "root rank[%u] run failed!", srcRank, root_), ret);
        }
    } else {
        // 非root 给root发
        CHK_RET(RunSendGather(root_, sendSlice, links));
    }

    return HCCL_SUCCESS;
}

void GatherStar::PrepareSlicesData(const u32 unitSize, const u64 totalCount, const u32 rankSize) const
{
    slices_.resize(rankSize);
    u64 sliceSize = totalCount * unitSize;

    for (u32 i = 0; i < rankSize; i++) {
        slices_[i].offset = i * sliceSize;
        slices_[i].size = sliceSize;
        HCCL_DEBUG("default slice[%u]: offset: [%llu] size[%llu]", i, i * sliceSize, sliceSize);
    }
}

HcclResult GatherStar::RunRecvGather(const u32 srcRank, const Slice &slice, const std::vector<LINK> &links)
{
    // root 接受数据
    DeviceMem dst;
    if (slice.size > 0 && srcRank != root_) {
        if (srcRank >= links.size()) {
                HCCL_ERROR("[Run][GatherStar][RecvGather]SrcRank[%u] is out of range, linkSize[%llu]", \
                    srcRank, links.size());
                return HCCL_E_INTERNAL;
        }

        dst = outputMem_.range(slice.size * srcRank, slice.size);
        HCCL_DEBUG("rank[%u] will rcv with ouput's offset[%llu], size[%llu] dstmem[%p]", \
            root_, slice.offset, slice.size, dst.ptr());

        if (links[srcRank]->IsTransportRoce()) {
            CHK_RET(links[srcRank]->TxEnv(dst.ptr(), slice.size, stream_));
        } else {
            CHK_RET(links[srcRank]->TxAck(stream_));
        }

        HcclResult ret = links[srcRank]->RxAsync(UserMemType::OUTPUT_MEM, slice.offset, dst.ptr(), slice.size, stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Run][GatherStar][RecvGather]rank[%u] rx async with output's offset[%llu] failed", \
            root_, slice.offset), ret);

        if (!links[srcRank]->IsTransportRoce()) {
            ret = ExecuteBarrier(links[srcRank], stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Run][GatherStar][SendGather]srcRank[%u] gather mesh run executor barrier failed", \
                    srcRank), ret);

            ret = links[srcRank]->RxWaitDone(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Run][GatherStar][GatherStar][SendGather]RxWaitDone failed"), ret);
        }
    } else if (srcRank == root_) {
        CHK_RET(HcclMemcpyAsync(dispatcher_, static_cast<u8 *>(outputMem_.ptr()) + slice.offset,
            outputMem_.size() - slice.offset, inputMem_.ptr(), slice.size,
            HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE, stream_,
            INVALID_VALUE_RANKID, LinkType::LINK_ONCHIP));
    }

    return HCCL_SUCCESS;
}

HcclResult GatherStar::RunSendGather(const u32 dstRank, const Slice &slice,
    const std::vector<LINK> &links)
{
    // 非root发送数据
    if (slice.size > 0) {
        HCCL_DEBUG("root rank[%u] tx input[%p] offset[%llu] to srcrank size[%llu] ", \
            dstRank, inputMem_.ptr(), slice.offset, slice.size);
        if (dstRank >= links.size()) {
            HCCL_ERROR("[Run][GatherStar][SendGather]DstRank[%u] is out of range, link Size[%llu]", \
                dstRank, links.size());
            return HCCL_E_INTERNAL;
        }

        if (links[dstRank]->IsTransportRoce()) {
            CHK_RET(links[dstRank]->RxEnv(stream_));
        } else {
            CHK_RET(links[dstRank]->RxAck(stream_));
        }

        HcclResult ret = links[dstRank]->TxAsync(UserMemType::OUTPUT_MEM, slice.offset, inputMem_.ptr(),
            slice.size, stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][GatherStar][SendGather]srcrank[%u] tx async to root" \
            "rank[%u] run failed", dstRank, root_), ret);

        if (!links[dstRank]->IsTransportRoce()) {
            ret = ExecuteBarrierSrcRank(links[dstRank], stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][GatherStar][SendGather]dstRank[%u] gather mesh run" \
                "executor barrier failed", dstRank), ret);
            ret = links[dstRank]->TxWaitDone(stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][GatherStar][SendGather]TxWaitDone failed"), ret);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult GatherStar::ExecuteBarrierSrcRank(std::shared_ptr<Transport> link, Stream &stream) const
{
    CHK_RET(link->RxAck(stream));

    CHK_RET(link->TxAck(stream));

    CHK_RET(link->RxDataSignal(stream));

    CHK_RET(link->TxDataSignal(stream));

    return HCCL_SUCCESS;
}
}