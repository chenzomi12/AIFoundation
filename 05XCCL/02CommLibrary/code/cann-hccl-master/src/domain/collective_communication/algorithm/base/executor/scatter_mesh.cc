/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "scatter_mesh.h"

namespace hccl {
ScatterMesh::ScatterMesh(const HcclDispatcher dispatcher,
    const u32 interRank, const u32 interRankSize)
    : ExecutorBase(dispatcher), interRank_(interRank), interRankSize_(interRankSize)
{
}

ScatterMesh::~ScatterMesh()
{
}

void ScatterMesh::PrepareSlicesData(const u32 unitSize, const u64 totalCount, const u32 rankSize) const
{
    slices_.resize(rankSize);
    u64 sliceSize = (totalCount / rankSize) * unitSize;

    for (u32 i = 0; i < rankSize; i++) {
        slices_[i].offset = i * sliceSize;
        slices_[i].size = sliceSize;
        HCCL_DEBUG(" default slice[%u]: offset: [%llu] size[%llu]", i, i * sliceSize, sliceSize);
    }
}
// scatter的入口函数
HcclResult ScatterMesh::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links)
{
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    HCCL_INFO("ScatterMesh run: rank[%u] rankSize[%u] inputMem[%p] to outputMem[%p] count[%llu]", \
              rank, rankSize, inputMem_.ptr(), outputMem_.ptr(), count_);
    // ranksize ==1 的处理
    if (rankSize == 1) {
        if (inputMem_ != outputMem_) {
            HcclResult ret = HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[ScatterMesh][RunAsync]rank[%u] copy input[%p] to output[%p] "\
                "failed", rank, inputMem_.ptr(), outputMem_.ptr()), ret);
        }
        return HCCL_SUCCESS;
    }

    if (links.size() < rankSize) {
        HCCL_ERROR("[ScatterMesh][RunAsync]rank[%u] linksize[%llu] is less than rankSize[%u]",
            rank, links.size(), rankSize);
        return HCCL_E_INTERNAL;
    }

    u32 unitSize = DataUnitSize(dataType_);
    if (unitSize == 0) {
        HCCL_ERROR("[ScatterMesh][RunAsync]rank[%u] unit data size is zero", rank);
        return HCCL_E_INTERNAL;
    }
    if (slices_.size() == 0) {
        PrepareSlicesData(unitSize, count_, rankSize);
    }
    // rank存放scatter 结果的偏移
    u64 scatterOffset = slices_[rank].offset;
    HCCL_DEBUG("rank[%u] scatter_offset is [%llu] reslutsize[%llu]", rank, \
        scatterOffset, slices_[rank].size);

    // root rank向其他rank发送数据
    if (rank == root_) {
        for (u32 dstRank = 0; dstRank < rankSize; dstRank++) {
            HcclResult ret = RunSendScatter(dstRank, slices_[dstRank], links);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[ScatterMesh][RunAsync]root rank[%u] send scatter to "\
                "dstrank[%u] run failed", rank, dstRank), ret);
        }
    } else { // 非rootrank 从rootrank接收数据
        CHK_RET(RunRecvScatter(root_, slices_[rank], links));
    }

    // Barrier确保数据收发完成
    if (barrierSwitchOn_) {
        for (u32 dstRank = 0; dstRank < rankSize; dstRank++) {
            if (dstRank != interRank_) {
                CHK_RET(ExecuteBarrier(links[dstRank], stream_));
            }
        }
    }

    HCCL_INFO("ScatterMesh finished: rank[%u] end", rank);
    return HCCL_SUCCESS;
}

HcclResult ScatterMesh::RunSendScatter(const u32 dstRank, const Slice &slice,
    const std::vector<LINK> &links)
{
    DeviceMem dst;
    DeviceMem src;
    // 本rank是root rank，直接进行数据的拷贝，从input拷贝到output
    if (dstRank == interRank_) {
        if (inputMem_ != outputMem_) {
            // root rank给自身拷贝时候不需要同步信号，拷贝到outputmem的偏移不同
            src = inputMem_.range(slices_[interRank_].offset, slices_[interRank_].size);
            dst = outputMem_.range(slices_[interRank_].offset, slices_[interRank_].size);
            HCCL_DEBUG("root rank copy from input[%p] range[%llu] to output[%p] range[%llu], size[%llu]", src.ptr(),
                slices_[interRank_].offset, dst.ptr(), slices_[interRank_].offset, slices_[interRank_].size);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_));
        }
    } else { // root rank给其他rank进行数据发送
        src = inputMem_.range(slice.offset, slice.size);
        if (dstRank >= links.size()) {
            HCCL_ERROR("[Run][SendScatter]DstRank[%u] is out of range, link Size[%llu]", dstRank, links.size());
            return HCCL_E_INTERNAL;
        }

        HCCL_DEBUG("root rank tx input[%p] offset[%llu] to dstrank[%u] size[%llu] ", src.ptr(), slice.offset, dstRank,
            slice.size);
        // 接收目的rank的同步信号，便可进行下一轮发送
        CHK_RET(links[dstRank]->RxAck(stream_));

        HcclResult ret = links[dstRank]->TxAsync(UserMemType::OUTPUT_MEM, slice.offset + baseOffset_, src.ptr(),
            slice.size, stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Run][SendScatter]root rank[%u] tx async to dstrank[%u] run failed", interRank_, dstRank), ret);
        ret = links[dstRank]->TxWaitDone(stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][SendScatter]TxWaitDone failed"), ret);
    }
    return HCCL_SUCCESS;
}
// 只有非root节点才接收数据
HcclResult ScatterMesh::RunRecvScatter(const u32 srcRank, const Slice &slice, const std::vector<LINK> &links)
{
    DeviceMem dst;
    // 判断数据是否需要分片
    if (srcRank >= links.size()) {
        HCCL_ERROR("[Run][RecvScatter]SrcRank[%u] is out of range, linkSize[%llu]", srcRank, links.size());
        return HCCL_E_INTERNAL;
    }
    dst = outputMem_.range(slice.offset, slice.size);
    HCCL_DEBUG("rank[%u]  will rcv with ouput's offset[%llu], size[%llu] ", interRank_, slice.offset, slice.size);

    // 向root节点发送tx同步,rxmem可用
    HcclResult ret = links[srcRank]->TxAck(stream_);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Run][RecvScatter]rank[%u] tx ack to srcrank[%u] failed", interRank_, srcRank), ret);

    ret = links[srcRank]->RxAsync(UserMemType::INPUT_MEM, slice.offset + baseOffset_, dst.ptr(), slice.size, stream_);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Run][RecvScatter]rank[%u] rx async with output's offset[%llu] failed", interRank_, slice.offset),
        ret);
    ret = links[srcRank]->RxWaitDone(stream_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][SendScatter]RxWaitDone failed"), ret);
    return HCCL_SUCCESS;
}
}  // namespace hccl
