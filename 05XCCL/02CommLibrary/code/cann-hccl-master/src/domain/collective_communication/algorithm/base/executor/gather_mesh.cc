/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gather_mesh.h"

namespace hccl {
GatherMesh::GatherMesh(const HcclDispatcher dispatcher,
                       std::vector<Stream> &meshStreams,
                       const std::vector<std::shared_ptr<LocalNotify>> &meshSignal,
                       const std::vector<std::shared_ptr<LocalNotify>> &meshSignalAux,
                       u32 userRank)
    : ExecutorBase(dispatcher),
      meshStreams_(meshStreams),
      meshSignal_(meshSignal),
      meshSignalAux_(meshSignalAux),
      userRank_(userRank),
      round_(0)
{
}

GatherMesh::~GatherMesh()
{
}

void GatherMesh::PrepareSlicesData(const u32 unitSize, const u64 totalCount, const u32 rankSize) const
{
    slices_.resize(rankSize);
    u64 sliceSize = (totalCount / rankSize) * unitSize;

    for (u32 i = 0; i < rankSize; i++) {
        slices_[i].offset = i * sliceSize;
        slices_[i].size = sliceSize;
        HCCL_DEBUG("default slice[%u]: offset: [%llu] size[%llu]", i, i * sliceSize, sliceSize);
    }
}

HcclResult GatherMesh::ExecuteBarrierSrcRank(std::shared_ptr<Transport> link, Stream &stream) const
{
    CHK_RET(link->RxAck(stream));

    CHK_RET(link->TxAck(stream));

    CHK_RET(link->RxDataSignal(stream));

    CHK_RET(link->TxDataSignal(stream));

    return HCCL_SUCCESS;
}

// root rank接收数据
HcclResult GatherMesh::RunRecvGather(const u32 srcRank, const Slice &slice, const std::vector<LINK> &links)
{
    DeviceMem src;
    DeviceMem dst;
    if (srcRank == root_) {
        if (inputMem_ != outputMem_) {
            // root rank给自身拷贝时候不需要同步信号，拷贝到outputmem的偏移不同
            Slice &rootSlice = slices_[root_];
            src = inputMem_.range(rootSlice.offset, rootSlice.size);
            dst = outputMem_.range(rootSlice.offset, rootSlice.size);
            HCCL_DEBUG("root rank copy from input[%p] range[%llu] to output[%p] range[%llu], size[%llu]", src.ptr(),
                rootSlice.offset, dst.ptr(), rootSlice.offset, rootSlice.size);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));
        }
    } else {
        // 判断数据是否需要分片
        if (srcRank >= links.size()) {
            HCCL_ERROR("[Run][RecvGather]SrcRank[%u] is out of range, linkSize[%llu]", srcRank, links.size());
            return HCCL_E_INTERNAL;
        }
        const LINK &link = links[srcRank];
        dst = outputMem_.range(slice.offset, slice.size);
        HCCL_DEBUG("rank[%u] will rcv with ouput's offset[%llu], size[%llu] dstmem[%p]", root_, slice.offset,
            slice.size, dst.ptr());

        // 向非root节点发送tx同步,rxmem可用
        Stream &curStream = (round_ == 0) ? stream_ : meshStreams_[round_ - 1];
        HcclResult ret = link->TxAck(curStream);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Run][RecvGather]root rank[%u] tx ack to srcrank[%u] failed", root_, srcRank), ret);

        ret = link->RxAsync(UserMemType::OUTPUT_MEM, slice.offset, dst.ptr(), slice.size, curStream);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Run][RecvGather]rank[%u] rx async with output's offset[%llu] failed", root_, slice.offset),
            ret);

        CHK_RET(link->TxDataSignal(curStream));
        CHK_RET(link->RxWaitDone(curStream));
    }
    return HCCL_SUCCESS;
}

// 非root rank发送数据
HcclResult GatherMesh::RunSendGather(const u32 dstRank, const Slice &slice,
    const std::vector<LINK> &links)
{
    DeviceMem src;
    src = inputMem_.range(slice.offset, slice.size);
    if (dstRank >= links.size()) {
        HCCL_ERROR("[Run][SendGather]DstRank[%u] is out of range, link Size[%llu]", dstRank, links.size());
        return HCCL_E_INTERNAL;
    }
    const LINK &link = links[dstRank];

    HCCL_DEBUG("root rank[%u] tx input[%p] offset[%llu] to srcrank size[%llu] ", dstRank, src.ptr(), slice.offset,
        slice.size);
    // 接收目的rank的同步信号，便可进行下一轮发送
    CHK_RET(link->RxAck(stream_));

    HcclResult ret = link->TxAsync(UserMemType::OUTPUT_MEM, slice.offset, src.ptr(), slice.size, stream_);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Run][SendGather]srcrank[%u] tx async to root rank[%u] run failed", dstRank, dstRank), ret);

    CHK_RET(link->RxDataSignal(stream_));
    CHK_RET(link->TxWaitDone(stream_));
    return HCCL_SUCCESS;
}

// Gather的入口函数
HcclResult GatherMesh::RunAsync(const u32 rank, const u32 rankSize,
    const std::vector<std::shared_ptr<Transport>> &links)
{
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    HCCL_INFO("GatherMesh run: rank[%u] rankSize[%u] inputMem[%p] to outputMem[%p] count[%llu]", \
              rank, rankSize, inputMem_.ptr(), outputMem_.ptr(), count_);
    // ranksize ==1 的处理
    if (rankSize == 1) {
        if (inputMem_ != outputMem_) {
            HcclResult ret = HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[GatherMesh][RunAsync]rank[%u] copy input[%p] to output[%p] "\
                "failed", rank, inputMem_.ptr(), outputMem_.ptr()), ret);
        }
        return HCCL_SUCCESS;
    }

    if (links.size() < rankSize) {
        HCCL_ERROR("[GatherMesh][RunAsync]rank[%u] linksize[%llu] is less than rankSize[%u]",
            rank, links.size(), rankSize);
        return HCCL_E_INTERNAL;
    }

    u32 unitSize = SIZE_TABLE[dataType_];
    if (unitSize == 0) {
        HCCL_ERROR("[GatherMesh][RunAsync]rank[%u] unit data size is zero", rank);
        return HCCL_E_INTERNAL;
    }
    if (slices_.size() == 0) {
        PrepareSlicesData(unitSize, count_, rankSize);
    }

    if (rank == root_) {
        CHK_RET(AddMainSteamSubStreamSyncPre(rank, rankSize));
        // root rank接收其他rank发送的数据
        round_ = 0;
        for (u32 srcRank = 0; srcRank < rankSize; srcRank++) {
            HcclResult ret = RunRecvGather(srcRank, slices_[srcRank], links);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[GatherMesh][RunAsync]srcrank[%u] send gather to "\
                "root rank[%u] run failed", srcRank, rank), ret);
            if (srcRank != root_) {
                round_++;
            }
        }
        CHK_RET(AddMainSteamSubStreamSyncPost(rank, rankSize));
        // 添加空task,保证子图执行时不乱序
        CHK_RET(ExecutorBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));
    } else {
        // 非root rank向root rank发送数据
        CHK_RET(RunSendGather(root_, slices_[rank], links));
    }
    HCCL_INFO("GatherMesh finished: rank[%u], end", rank);
    return HCCL_SUCCESS;
}

HcclResult GatherMesh::AddMainSteamSubStreamSyncPre(u32 rank, u32 rankSize)
{
    for (u32 streamIndex = 0; streamIndex < rankSize - 2; streamIndex++) { // rankSize-2: stream num
        HCCL_DEBUG("rank[%u] streamindex[%u] wait signalaux[%p]",
            rank, streamIndex, meshSignalAux_[streamIndex]->ptr());
        CHK_RET(LocalNotify::Wait(meshStreams_[streamIndex], dispatcher_, meshSignalAux_[streamIndex],
            profilerInput_.stage));

        HCCL_DEBUG("rank[%u] siganl_aux index[%u] signal record signalaux[%p] ",
            rank, streamIndex, meshSignalAux_[streamIndex]->ptr());
        CHK_RET(LocalNotify::Post(stream_, dispatcher_, meshSignalAux_[streamIndex],
            profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult GatherMesh::AddMainSteamSubStreamSyncPost(u32 rank, u32 rankSize)
{
    for (u32 streamIndex = 0; streamIndex < rankSize - 2; streamIndex++) {  // rankSize - 2 stream num
        HCCL_DEBUG("rank[%u] streamindex[%u] wait signal[%p] ", \
                   rank, streamIndex, meshSignal_[streamIndex]->ptr());
        CHK_RET(LocalNotify::Wait(stream_, dispatcher_, meshSignal_[streamIndex], profilerInput_.stage));

        HCCL_DEBUG("rank[%u] streamindex[%u] record signal[%p]", \
                   rank, streamIndex, meshSignal_[streamIndex]->ptr());
        CHK_RET(LocalNotify::Post(meshStreams_[streamIndex], dispatcher_, meshSignal_[streamIndex],
            profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}
} // namespace hccl