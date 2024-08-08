/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common_operator.h"
#include "executor_impl.h"
#include "stream_active_manager.h"
#include "device_capacity.h"


namespace hccl {
CommonOperator::CommonOperator(std::unique_ptr<hcclImpl> &pImpl, std::unique_ptr<TopoMatcher> &topoMatcher,
                               HcclCMDType opType)
    : CollAlgOperator(pImpl, topoMatcher, opType)
{
}

CommonOperator::~CommonOperator()
{
}

HcclResult CommonOperator::CalUserMemSlices(const HcclDataType dataType, const HcomCollOpInfo *opInfo,
                                            const std::vector<Slice> &singleRingSliceZero, u32 ringIndex,
                                            const std::vector<std::vector<u32>> &multiRingsOrder,
                                            std::vector<Slice>                  &userMemSlices)
{
    if (opInfo == nullptr || opInfo->inputAddr == nullptr || opInfo->outputAddr == nullptr) {
        // 910_73场景下，allreduce算子的userMem上的slice信息
        userMemSlices = singleRingSliceZero;
        return HCCL_SUCCESS;
    }
    // 910_73场景下，reduce scatter和all gather算子的userMem上的slice信息
    std::vector<u32> ring0 = multiRingsOrder[0];
    for (u32 sliceIdx = 0; sliceIdx < singleRingSliceZero.size(); sliceIdx++) {
        Slice userMemSlice;
        u32 deviceId = multiRingsOrder[ringIndex][sliceIdx];
        u32 pos = distance(ring0.begin(), find(ring0.begin(), ring0.end(), deviceId));
        userMemSlice.offset = pos * opInfo->count * SIZE_TABLE[dataType]
                                + singleRingSliceZero[0].offset;
        userMemSlice.size = singleRingSliceZero[sliceIdx].size;
        userMemSlices.push_back(userMemSlice);
        HCCL_DEBUG(
            "[CommonOperator][CalUserMemSlices] Push back userMemSlice offset[%llu], size[%llu] at rank[%u]",
            userMemSlice.offset, userMemSlice.size, userRank_);
    }
    return HCCL_SUCCESS;
}

HcclResult CommonOperator::GetRankOrder(const std::vector<std::vector<u32>> &multiRingsOrder, u32 ringIndex,
    std::vector<u32> &rankOrder)
{
    std::vector<u32> ring0 = multiRingsOrder[0];
    std::vector<u32> ringOrder = multiRingsOrder[ringIndex];
    for (u32 i = 0; i < ringOrder.size(); i++) {
        u32 deviceId = ringOrder[i];
        u32 pos = distance(ring0.begin(), find(ring0.begin(), ring0.end(), deviceId));
        rankOrder.push_back(pos);
    }
    return HCCL_SUCCESS;
}

HcclResult CommonOperator::GetSubStreamInfoOnOneRing(const innerStreamInfo_t &streamInfo, const u32 ringIndex,
                                                     std::vector<Stream>                       &subStreamsInOneRing,
                                                     std::vector<std::shared_ptr<LocalNotify>> &mainSignalsInOneRing,
                                                     std::vector<std::shared_ptr<LocalNotify>> &subSignalsInOneRing)
{
    if (GetExternalInputEnableRdmaSdmaConcurrent() && deviceType_ == DevType::DEV_TYPE_910_73) {
        subStreamsInOneRing.push_back(streamInfo.ringStreams[ringIndex + RDMA_ADD_STREAMS_NUM]);
        mainSignalsInOneRing.push_back(streamInfo.ringSignal[ringIndex + RDMA_ADD_STREAMS_NUM]);
        subSignalsInOneRing.push_back(streamInfo.ringSignalAux[ringIndex + RDMA_ADD_STREAMS_NUM]);
    } else if (streamInfo.ringNum == OUTER_PLANE_NUM_IN_NPRING_DOUBLE * STREAM_NUM_FOR_DMAREDUCE_ONE_RING) {
        // double ring
        subStreamsInOneRing.push_back(streamInfo.ringStreams[ringIndex + 1]);
        mainSignalsInOneRing.push_back(streamInfo.ringSignal[ringIndex + 1]);
        subSignalsInOneRing.push_back(streamInfo.ringSignalAux[ringIndex + 1]);
    } else if (streamInfo.ringNum == OUTER_PLANE_NUM_IN_NPRING_SINGLE * STREAM_NUM_FOR_DMAREDUCE_ONE_RING) {
        // single ring
        subStreamsInOneRing.push_back(streamInfo.ringStreams[ringIndex]);
        mainSignalsInOneRing.push_back(streamInfo.ringSignal[ringIndex]);
        subSignalsInOneRing.push_back(streamInfo.ringSignalAux[ringIndex]);
    }
    return HCCL_SUCCESS;
}

HcclResult CommonOperator::MultiRingGather(const std::string &tag, DeviceMem inputMem, DeviceMem outputMem,
    const u64 count, const HcclDataType dataType, const std::vector<std::vector<Slice> > multRingsSliceZero,
    HcclReduceOp op, u32 root, Stream stream, s32 profStage)
{
    u32 ringNum = multRingsSliceZero.size();
    innerStreamInfo_t *streamInfo = hcclImpl_->GetStreamInfo(tag);
    CHK_PRT_RET(streamInfo == nullptr,
        HCCL_ERROR("[GetStreamInfo]errNo[0x%016llx] tag[%s] can't find in stream info",
            HCCL_ERROR_CODE(HCCL_E_NOT_FOUND), tag.c_str()), HCCL_E_PARA);
    CommInfo *currComm;
    hcclImpl_->GetCommInfo(currComm, tag);

    std::vector<std::vector<u32>> ringNics;
    CHK_RET(hcclImpl_->GetRingNics(tag, ringNics));

    for (u32 ringIndex = 0; ringIndex < ringNum; ringIndex++) {
        std::vector<Slice> singleRingSliceZero = multRingsSliceZero[ringIndex];
        CHK_PRT_RET(singleRingSliceZero.empty(),
            HCCL_ERROR("[CommonOperator][MultiRingGather]singleRingSliceZero is empty"), HCCL_E_INTERNAL);

        std::unique_ptr<CommBase> &commRing = currComm->commOuter[ringIndex];
        CHK_SMART_PTR_NULL(commRing);
        u32 rootRank = 0;
        HcclResult ret = commRing->GetRankByUserRank(root, rootRank);
        CHK_PRT_RET(ret == HCCL_E_PARA,
            HCCL_ERROR("[CommonOperator][MultiRingGather]invalid root rank[%u] to get user rank", root), ret);

        u32 rankSize = commRing->RankSize();
        std::unique_ptr<ExecutorBase> executor(new (std::nothrow) GatherRing(dispatcher_));
        CHK_SMART_PTR_NULL(executor);

        if (ringIndex != (ringNum - 1)) {  // 0~ringNum-2的环
            if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) { // offline
                CHK_RET(StreamActiveManager::GetInstance(deviceLogicId_).StreamActive(
                    streamInfo->ringStreams[ringIndex].ptr(), stream.ptr()));
            }
            ret = LocalNotify::Wait(streamInfo->ringStreams[ringIndex], dispatcher_,
                streamInfo->ringSignalAux[ringIndex], profStage);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[CommonOperator][MultiRingGather]in stream[%u] wait failed", \
                ringIndex), ret);
            if (singleRingSliceZero[0].size != 0) {
            ret = executor->Prepare(inputMem, outputMem, outputMem, count, dataType,
                                    streamInfo->ringStreams[ringIndex], op, rootRank, singleRingSliceZero, 0,
                                    ringNics[ringIndex]);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CommonOperator][MultiRingGather]stream[%u],gather(ring) prepare failed, "\
                "return[%d]", ringIndex, ret), ret);

            ret = executor->RegisterProfiler(commRing ->Rank(), profStage, HCCL_EXEC_STEP_NOT_SET,
                streamInfo->ringStreams[ringIndex]);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CommonOperator][MultiRingGather]stream[%u], gather(ring) register profiler "\
                "failed,return[%d]", ringIndex, ret), ret);

            ret = commRing->RunExecutor(executor);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CommonOperator][MultiRingGather]stream[%u],gather(ring) run failed,return[%d]",
                ringIndex, ret), ret);
            }
            ret = LocalNotify::Post(streamInfo->ringStreams[ringIndex], dispatcher_, streamInfo->ringSignal[ringIndex],
                profStage);

            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[CommonOperator][MultiRingGather]stream[%u] record failed", \
                ringIndex), ret);

            ret = LocalNotify::Post(stream, dispatcher_, streamInfo->ringSignalAux[ringIndex], profStage);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[CommonOperator][MultiRingGather]stream[%u] record failed", \
                ringIndex), ret);
            } else {  // 主环
                    executor.reset(new (std::nothrow) GatherRing(dispatcher_));
                    CHK_SMART_PTR_NULL(executor);

                    ret = executor->Prepare(inputMem, outputMem, outputMem, count, dataType, stream,
                        op, rootRank, singleRingSliceZero, 0, ringNics[ringIndex]);
                    CHK_PRT_RET(ret != HCCL_SUCCESS,
                        HCCL_ERROR("[CommonOperator][MultiRingGather]stream[%u],gather(ring) prepare failed, "\
                        "return[%d]", ringIndex, ret), ret);

                    ret = executor->RegisterProfiler(((ringIndex + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                        (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + commRing ->Rank(),
                        profStage, HCCL_EXEC_STEP_NOT_SET, stream);
                    CHK_PRT_RET(ret != HCCL_SUCCESS,
                        HCCL_ERROR("[CommonOperator][MultiRingGather]stream[%u], gather(ring) register "\
                        "profiler failed,return[%d]", ringIndex, ret), ret);

                    ret = commRing->RunExecutor(executor);
                    CHK_PRT_RET(ret != HCCL_SUCCESS,
                        HCCL_ERROR("[CommonOperator][MultiRingGather]stream[%u],gather(ring) run failed, "\
                        "return[%d]", ringIndex, ret), ret);
                for (u32 ring = 0; ring < (ringNum - 1); ring++) {
                    /* 等待executor执行完毕 , 当前环没有分配数据，跳过此环处理，继续下一个环 */
                    ret = LocalNotify::Wait(stream, dispatcher_, streamInfo->ringSignal[ring], profStage);
                    CHK_PRT_RET(ret != HCCL_SUCCESS,
                        HCCL_ERROR("[CommonOperator][MultiRingGather]stream[%u] wait failed", ring), ret);
                }
            }
        }
    // 添加空task,保证子图执行时不乱序
    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}

HcclResult CommonOperator::MultiRingAllGather(const std::string &tag, DeviceMem inputMem, DeviceMem outputMem,
    const u64 count, const HcclDataType dataType, const std::vector<std::vector<Slice> > multRingsSliceZero,
    Stream stream, s32 profStage, const u64 baseOffset, const HcomCollOpInfo *opInfo,
    const std::vector<std::vector<Slice>> multRingsUserMemSlice)
{
    HcclResult ret = HCCL_SUCCESS;
    u32 ringNum = multRingsSliceZero.size();

    innerStreamInfo_t *streamInfo = hcclImpl_->GetStreamInfo(tag);
    CHK_PRT_RET(streamInfo == nullptr,
        HCCL_ERROR("[GetStreamInfo]errNo[0x%016llx] tag[%s] can't find in stream info",
            HCCL_ERROR_CODE(HCCL_E_NOT_FOUND), tag.c_str()), HCCL_E_PARA);

    CommInfo *currComm;
    ret = hcclImpl_->GetCommInfo(currComm, tag);

    std::vector<std::vector<u32>> ringNics;
    CHK_RET(hcclImpl_->GetRingNics(tag, ringNics));
    // 拿到ring环映射关系
    u32 ranksSize = currComm->commOuter[COMM_INDEX_0]->RankSize();
    std::vector<std::vector<u32>> multiRingsOrder = GetRingsOrderByTopoType(ranksSize, topoType_, nicList_);

    // 空拷贝用于后续操作附着
    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    for (u32 ringIndex = 0; ringIndex < ringNum; ringIndex++) {
        std::vector<Slice> singleRingSliceZero = multRingsSliceZero[ringIndex];
        CHK_PRT_RET(singleRingSliceZero.empty(), HCCL_ERROR("[CommonOperator][MultiRingAllGather]"\
            "singleRingSliceZero is empty"), HCCL_E_INTERNAL);

        // 910_73场景 生成userMemOut_上对应的slices
        std::vector<Slice> userMemOutputSlices;
        if (multRingsUserMemSlice.size() == 0) {
            CHK_RET(CalUserMemSlices(dataType, opInfo, singleRingSliceZero, ringIndex, multiRingsOrder,
                userMemOutputSlices));
        } else {
            userMemOutputSlices = multRingsUserMemSlice[ringIndex];
        }
        std::vector<u32> rankOrder;
        CHK_RET(GetRankOrder(multiRingsOrder, ringIndex, rankOrder));
        std::unique_ptr<CommBase> &commRing = currComm->commOuter[ringIndex];
        CHK_SMART_PTR_NULL(commRing);

        u32 rankSize = commRing->RankSize();
        u32 ringIndexOp = ringIndex;

        // 910_73场景 准备环中的从流
        std::vector<Stream>       subStreamsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> mainSignalsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> subSignalsInOneRing;
        if (opInfo != nullptr) {
            CHK_RET(GetSubStreamInfoOnOneRing(*streamInfo, ringIndex, subStreamsInOneRing, mainSignalsInOneRing,
                                              subSignalsInOneRing));
        }

        if (ringIndex != (ringNum - 1)) { // 最后一个环是主stream，所以这里减1，符合条件的走从stream
            if (!GetExternalInputHcclEnableFfts() &&
                GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                if (opInfo != nullptr) {
                    streamInfo->ringThreadsManage[ringIndex]->Prepare(
                        outputMem, outputMem, inputMem, count, dataType,
                        streamInfo->ringStreams[ringIndex], HcclReduceOp::HCCL_REDUCE_RESERVED, OUTER_BRIDGE_RANK_ID,
                        singleRingSliceZero, baseOffset, ringNics[ringIndex], tag, profStage,
                        commRing.get(), streamInfo->ringSignalAux[ringIndex], streamInfo->ringSignal[ringIndex],
                        ringIndex, ExecutorType::ALLGATHER_RING_DIRECT, 0, opInfo, subStreamsInOneRing,
                        mainSignalsInOneRing, subSignalsInOneRing, rankOrder, userMemOutputSlices);
                } else {
                    streamInfo->ringThreadsManage[ringIndex]->Prepare(outputMem, outputMem, inputMem, count, dataType,
                        streamInfo->ringStreams[ringIndex], HcclReduceOp::HCCL_REDUCE_RESERVED, OUTER_BRIDGE_RANK_ID,
                        singleRingSliceZero, baseOffset, ringNics[ringIndex], tag, profStage,
                        commRing.get(), streamInfo->ringSignalAux[ringIndex], streamInfo->ringSignal[ringIndex],
                        ringIndex, ExecutorType::ALLGATHER_RING);
                }
                streamInfo->ringThreadsManage[ringIndex]->NotifyStart();    // 给线程发信号启动处理
            } else {
                ret = LocalNotify::Wait(streamInfo->ringStreams[ringIndex], dispatcher_,
                    streamInfo->ringSignalAux[ringIndex], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CommonOperator][MultiRingAllGather]stream[%u] wait failed", ringIndex), ret);
                // 如何判断是否环内是否有数据, 以ring的第一个rank的 size为判断依据
                std::unique_ptr<ExecutorBase> executor;
                if (opInfo != nullptr) {
                    executor.reset(new (std::nothrow) AllGatherRingConcurrentDirect(
                        dispatcher_, opInfo, commRing->UserRank(), subStreamsInOneRing,
                        mainSignalsInOneRing, subSignalsInOneRing, rankOrder, userMemOutputSlices));
                } else {
                    executor.reset(new (std::nothrow) AllGatherRing(dispatcher_));
                }
                CHK_SMART_PTR_NULL(executor);
                ret = executor->Prepare(outputMem, outputMem, inputMem, count, dataType,
                    streamInfo->ringStreams[ringIndex], HcclReduceOp::HCCL_REDUCE_RESERVED, OUTER_BRIDGE_RANK_ID,
                    singleRingSliceZero, baseOffset, ringNics[ringIndex]);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CommonOperator][MultiRingAllGather]stream[%u],all gather(ring) prepare "\
                    "failed,return[%d]", ringIndex, ret), ret);
                ret = executor->RegisterProfiler(
                    ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                    (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + commRing ->Rank(),
                    profStage, HCCL_EXEC_STEP_NOT_SET, streamInfo->ringStreams[ringIndex]);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CommonOperator][MultiRingAllGather]stream[%u],all gather(ring) register "\
                    "Profiler failed,return[%d]", ringIndex, ret), ret);

                ret = commRing->RunExecutor(executor);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CommonOperator][MultiRingAllGather]stream[%u],all gather(ring) run failed, "\
                    "return[%d]", ringIndex, ret), ret);

                ret = LocalNotify::Post(streamInfo->ringStreams[ringIndex], dispatcher_,
                    streamInfo->ringSignal[ringIndex], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CommonOperator][MultiRingAllGather]stream[%u] record failed",
                    ringIndex), ret);
            }

            ret = LocalNotify::Post(stream, dispatcher_, streamInfo->ringSignalAux[ringIndex], profStage);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CommonOperator][MultiRingAllGather]stream[%u] record failed", ringIndex), ret);
        } else { // 主环
            std::unique_ptr<ExecutorBase> executor;
            if (opInfo != nullptr) {
                executor.reset(new (std::nothrow) AllGatherRingConcurrentDirect(
                    dispatcher_, opInfo, commRing->UserRank(), subStreamsInOneRing, mainSignalsInOneRing,
                    subSignalsInOneRing, rankOrder, userMemOutputSlices));
            } else {
                executor.reset(new (std::nothrow) AllGatherRing(dispatcher_));
            }
            CHK_SMART_PTR_NULL(executor);
            ret = executor->Prepare(outputMem, outputMem, inputMem, count, dataType, stream, HCCL_REDUCE_RESERVED,
                OUTER_BRIDGE_RANK_ID, singleRingSliceZero, baseOffset, ringNics[ringIndex]);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CommonOperator][MultiRingAllGather]stream[%u],all gather(ring) prepare failed,"\
                "return[%d]", ringIndex, ret), ret);

            ret = executor->RegisterProfiler(
                ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + commRing->Rank(),
                profStage, HCCL_EXEC_STEP_NOT_SET, stream);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CommonOperator][MultiRingAllGather]stream[%u],all gather(ring) register Profiler "\
                "failed,return[%d]", ringIndex, ret), ret);

            ret = commRing->RunExecutor(executor);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CommonOperator][MultiRingAllGather]stream[%u],all gather(ring) run failed,"\
                "return[%d]", ringIndex, ret), ret);

            for (u32 ring = 0; ring < (ringNum - 1); ring++) {
                if (!GetExternalInputHcclEnableFfts() &&
                    GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                    streamInfo->ringThreadsManage[ring]->WaitDone(); // 单算子模式，等待线程处理完成信号
                }
                ret = LocalNotify::Wait(stream, dispatcher_, streamInfo->ringSignal[ring], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CommonOperator][MultiRingAllGather]stream[%u] wait failed", ring), ret);
            }
        }
    }
    // 添加空task,保证执行时不乱序
    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}

HcclResult CommonOperator::MultiRingAllGatherConcurrent(const std::string &tag, DeviceMem inputMem, DeviceMem outputMem,
    const u64 count, const HcclDataType dataType,
    const std::vector<std::pair<bool, std::vector<Slice>>> multRingsSliceZero,
    Stream stream, s32 profStage, const u64 baseOffset, const HcomCollOpInfo *opInfo)
{
    HcclResult ret = HCCL_SUCCESS;
    u32 ringNum = multRingsSliceZero.size(); // 环数, 当前为4环

    innerStreamInfo_t *streamInfo = hcclImpl_->GetStreamInfo(tag);
    CHK_PRT_RET(streamInfo == nullptr,
        HCCL_ERROR("[GetStreamInfo]errNo[0x%016llx] tag[%s] can't find in stream info",
            HCCL_ERROR_CODE(HCCL_E_NOT_FOUND), tag.c_str()), HCCL_E_PARA);

    CommInfo *currComm;
    ret = hcclImpl_->GetCommInfo(currComm, tag);

    std::vector<std::vector<u32>> ringNics;
    CHK_RET(hcclImpl_->GetRingNics(tag, ringNics));
    auto halfRingSize = ringNum;
    if (ringNum > RDMA_PLANE_NUM_IN_NPRING_DOUBLE) {
        halfRingSize = ringNum / 2; // 2环
    }
    // 拿到ring环映射关系
    u32 ranksSize = currComm->commOuter[COMM_INDEX_0]->RankSize();
    std::vector<std::vector<u32>> multiRingsOrder = GetRingsOrderByTopoType(ranksSize, topoType_, nicList_);

    // 空拷贝用于后续操作附着
    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    for (u32 ringIndex = 0; ringIndex < ringNum; ringIndex++) {
        std::vector<Slice> singleRingSliceZero = multRingsSliceZero[ringIndex].second; // 取出sdma/rdma的数据块
        CHK_PRT_RET(singleRingSliceZero.empty(), HCCL_ERROR("[CommonOperator][MultiRingAllGatherConcurrent]"\
            "singleRingSliceZero is empty"), HCCL_E_INTERNAL);

        // 910_73场景 生成userMemOut_上对应的slices
        std::vector<Slice> userMemOutputSlices;
        CHK_RET(
            CalUserMemSlices(dataType, opInfo, singleRingSliceZero, ringIndex, multiRingsOrder, userMemOutputSlices));
        std::vector<u32> rankOrder;
        u32 commIndex = ringIndex % halfRingSize;
        CHK_RET(GetRankOrder(multiRingsOrder, commIndex, rankOrder));
        std::unique_ptr<CommBase> &commRing = multRingsSliceZero[ringIndex].first ? currComm->commOuter[commIndex] :
                                                                                    currComm->commOuterRdma[commIndex];
        CHK_SMART_PTR_NULL(commRing);

        u32 rankSize = commRing->RankSize();
        u32 ringIndexOp = ringIndex;

        // 910_73场景 准备环中的从流
        std::vector<Stream>       subStreamsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> mainSignalsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> subSignalsInOneRing;
        if (opInfo != nullptr) {
            CHK_RET(GetSubStreamInfoOnOneRing(*streamInfo, ringIndex, subStreamsInOneRing, mainSignalsInOneRing,
                                              subSignalsInOneRing));
        }

        if (ringIndex != (ringNum - 1)) { // 最后一个环是主stream，所以这里减1，符合条件的走从stream
            if (!GetExternalInputHcclEnableFfts() &&
                GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                if (opInfo != nullptr) {
                    streamInfo->ringThreadsManage[ringIndex]->Prepare(
                        outputMem, outputMem, inputMem, count, dataType,
                        streamInfo->ringStreams[ringIndex], HcclReduceOp::HCCL_REDUCE_RESERVED, OUTER_BRIDGE_RANK_ID,
                        singleRingSliceZero, baseOffset, ringNics[ringIndex%halfRingSize], tag, profStage,
                        commRing.get(), streamInfo->ringSignalAux[ringIndex], streamInfo->ringSignal[ringIndex],
                        ringIndex, ExecutorType::ALLGATHER_RING_DIRECT, 0, opInfo, subStreamsInOneRing,
                        mainSignalsInOneRing, subSignalsInOneRing, rankOrder, userMemOutputSlices);
                } else {
                    streamInfo->ringThreadsManage[ringIndex]->Prepare(outputMem, outputMem, inputMem, count, dataType,
                        streamInfo->ringStreams[ringIndex], HcclReduceOp::HCCL_REDUCE_RESERVED, OUTER_BRIDGE_RANK_ID,
                        singleRingSliceZero, baseOffset, ringNics[ringIndex%halfRingSize], tag, profStage,
                        commRing.get(), streamInfo->ringSignalAux[ringIndex], streamInfo->ringSignal[ringIndex],
                        ringIndex, ExecutorType::ALLGATHER_RING);
                }
                streamInfo->ringThreadsManage[ringIndex]->NotifyStart();    // 给线程发信号启动处理
            } else {
                ret = LocalNotify::Wait(streamInfo->ringStreams[ringIndex], dispatcher_,
                    streamInfo->ringSignalAux[ringIndex], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CommonOperator][MultiRingAllGatherConcurrent]stream[%u] wait failed", ringIndex), ret);
                // 如何判断是否环内是否有数据, 以ring的第一个rank的 size为判断依据
                std::unique_ptr<ExecutorBase> executor;
                if (opInfo != nullptr) {
                    executor.reset(new (std::nothrow) AllGatherRingConcurrentDirect(
                        dispatcher_, opInfo, commRing->UserRank(), subStreamsInOneRing,
                        mainSignalsInOneRing, subSignalsInOneRing, rankOrder, userMemOutputSlices));
                } else {
                    executor.reset(new (std::nothrow) AllGatherRing(dispatcher_));
                }
                CHK_SMART_PTR_NULL(executor);
                ret = executor->Prepare(outputMem, outputMem, inputMem, count, dataType,
                    streamInfo->ringStreams[ringIndex], HcclReduceOp::HCCL_REDUCE_RESERVED, OUTER_BRIDGE_RANK_ID,
                    singleRingSliceZero, baseOffset, ringNics[ringIndex%halfRingSize]);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CommonOperator][MultiRingAllGatherConcurrent]stream[%u],all gather(ring) prepare "\
                    "failed,return[%d]", ringIndex, ret), ret);
                ret = executor->RegisterProfiler(
                    ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                    (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + commRing ->Rank(),
                    profStage, HCCL_EXEC_STEP_NOT_SET, streamInfo->ringStreams[ringIndex]);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CommonOperator][MultiRingAllGatherConcurrent]stream[%u],all gather(ring) register "\
                    "Profiler failed,return[%d]", ringIndex, ret), ret);

                ret = commRing->RunExecutor(executor);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CommonOperator][MultiRingAllGatherConcurrent]stream[%u],all gather(ring) run failed, "\
                    "return[%d]", ringIndex, ret), ret);

                ret = LocalNotify::Post(streamInfo->ringStreams[ringIndex], dispatcher_,
                    streamInfo->ringSignal[ringIndex], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CommonOperator][MultiRingAllGatherConcurrent]stream[%u] record failed",
                    ringIndex), ret);
            }

            ret = LocalNotify::Post(stream, dispatcher_, streamInfo->ringSignalAux[ringIndex], profStage);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CommonOperator][MultiRingAllGatherConcurrent]stream[%u] record failed", ringIndex), ret);
        } else { // 主环
            std::unique_ptr<ExecutorBase> executor;
            if (opInfo != nullptr) {
                executor.reset(new (std::nothrow) AllGatherRingConcurrentDirect(
                    dispatcher_, opInfo, commRing->UserRank(), subStreamsInOneRing, mainSignalsInOneRing,
                    subSignalsInOneRing, rankOrder, userMemOutputSlices));
            } else {
                executor.reset(new (std::nothrow) AllGatherRing(dispatcher_));
            }
            CHK_SMART_PTR_NULL(executor);
            ret = executor->Prepare(outputMem, outputMem, inputMem, count, dataType, stream, HCCL_REDUCE_RESERVED,
                OUTER_BRIDGE_RANK_ID, singleRingSliceZero, baseOffset, ringNics[ringIndex%halfRingSize]);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CommonOperator][MultiRingAllGatherConcurrent]stream[%u],all gather(ring) prepare failed,"\
                "return[%d]", ringIndex, ret), ret);

            ret = executor->RegisterProfiler(
                ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + commRing->Rank(),
                profStage, HCCL_EXEC_STEP_NOT_SET, stream);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CommonOperator][MultiRingAllGatherConcurrent]stream[%u],all gather(ring) register "\
                "Profiler failed, return[%d]", ringIndex, ret), ret);

            ret = commRing->RunExecutor(executor);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CommonOperator][MultiRingAllGatherConcurrent]stream[%u],all gather(ring) run failed,"\
                "return[%d]", ringIndex, ret), ret);

            for (u32 ring = 0; ring < (ringNum - 1); ring++) {
                if (!GetExternalInputHcclEnableFfts() &&
                    GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                    streamInfo->ringThreadsManage[ring]->WaitDone(); // 单算子模式，等待线程处理完成信号
                }
                ret = LocalNotify::Wait(stream, dispatcher_, streamInfo->ringSignal[ring], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CommonOperator][MultiRingAllGatherConcurrent]stream[%u] wait failed", ring), ret);
            }
        }
    }
    // 添加空task,保证执行时不乱序
    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}

HcclResult CommonOperator::MultiRingReduceScatter(const std::string &tag, DeviceMem inputMem, DeviceMem outputMem,
    const u64 count, const HcclDataType dataType, const HcclReduceOp reductionOp,
    const std::vector<std::vector<Slice> > multRingsSliceZero, Stream stream, s32 profStage,
    const u64 baseOffset, const HcomCollOpInfo *opInfo,
    const std::vector<std::vector<Slice>> multRingsUserMemSlice)
{
    HcclResult ret = HCCL_SUCCESS;
    u32 ringNum = multRingsSliceZero.size();
    innerStreamInfo_t *streamInfo = hcclImpl_->GetStreamInfo(tag);
    CHK_PRT_RET(streamInfo == nullptr,
        HCCL_ERROR("[GetStreamInfo]errNo[0x%016llx] tag[%s] can't find in stream info",
            HCCL_ERROR_CODE(HCCL_E_NOT_FOUND), tag.c_str()), HCCL_E_PARA);

    CommInfo *currComm;
    ret = hcclImpl_->GetCommInfo(currComm, tag);

    std::vector<std::vector<u32>> ringNics;
    CHK_RET(hcclImpl_->GetRingNics(tag, ringNics));
    // 拿到ring环映射关系
    u32 ranksSize = currComm->commOuter[COMM_INDEX_0]->RankSize();
    std::vector<std::vector<u32>> multiRingsOrder = GetRingsOrderByTopoType(ranksSize, topoType_, nicList_);

    u64 reduceAttr = GetReduceAttr(inputMem, outputMem, dataType, reductionOp);

    // 空拷贝用于后续操作附着
    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    for (u32 ringIndex = 0; ringIndex < ringNum; ringIndex++) {
        std::vector<Slice> singleRingSliceZero = multRingsSliceZero[ringIndex];
        CHK_PRT_RET(singleRingSliceZero.empty(),
            HCCL_ERROR("[CommonOperator][MultiRingReduceScatter]singleRingSliceZero is empty"), HCCL_E_INTERNAL);

        // 生成userMemIn_上对应的slices
        std::vector<Slice> userMemInputSlices;
        if (multRingsUserMemSlice.size() == 0) {
            CHK_RET(CalUserMemSlices(dataType, opInfo, singleRingSliceZero, ringIndex, multiRingsOrder,
                userMemInputSlices));
        } else {
            userMemInputSlices = multRingsUserMemSlice[ringIndex];
        }
        std::vector<u32> rankOrder;
        CHK_RET(GetRankOrder(multiRingsOrder, ringIndex, rankOrder));
        std::unique_ptr<CommBase> &commRing = currComm->commOuter[ringIndex];
        CHK_SMART_PTR_NULL(commRing);

        u32 rankSize = commRing->RankSize();
        u32 ringIndexOp = ringIndex;

        std::vector<Stream>       subStreamsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> mainSignalsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> subSignalsInOneRing;
        if (opInfo != nullptr) {
            CHK_RET(GetSubStreamInfoOnOneRing(*streamInfo, ringIndex, subStreamsInOneRing, mainSignalsInOneRing,
                                              subSignalsInOneRing));
        }

        if (ringIndex != (ringNum - 1)) {  // 0~ringNum-2的环
            if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) { // offline
                ret = StreamActiveManager::GetInstance(deviceLogicId_).StreamActive(
                    streamInfo->ringStreams[ringIndex].ptr(), stream.ptr());
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CommonOperator][MultiRingReduceScatter]active stream[%u], failed", ringIndex), ret);
            }

            if (!GetExternalInputHcclEnableFfts() &&
                GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                /* 更新线程参数 */
                if (opInfo != nullptr) {
                    streamInfo->ringThreadsManage[ringIndex]->Prepare(
                        inputMem, inputMem, outputMem, count, dataType, streamInfo->ringStreams[ringIndex], reductionOp,
                        OUTER_BRIDGE_RANK_ID, singleRingSliceZero, baseOffset, ringNics[ringIndex], tag, profStage,
                        commRing.get(), streamInfo->ringSignalAux[ringIndex], streamInfo->ringSignal[ringIndex],
                        ringIndex, ExecutorType::REDUCE_SCATTER_RING_DIRECT, reduceAttr, opInfo,
                        subStreamsInOneRing, mainSignalsInOneRing, subSignalsInOneRing, rankOrder,
                        userMemInputSlices);
                } else {
                    streamInfo->ringThreadsManage[ringIndex]->Prepare(inputMem, inputMem, outputMem, count, dataType,
                        streamInfo->ringStreams[ringIndex], reductionOp, OUTER_BRIDGE_RANK_ID, singleRingSliceZero,
                        baseOffset, ringNics[ringIndex], tag, profStage, commRing.get(),
                        streamInfo->ringSignalAux[ringIndex], streamInfo->ringSignal[ringIndex], ringIndex,
                        ExecutorType::REDUCE_SCATTER_RING, reduceAttr);
                }

                streamInfo->ringThreadsManage[ringIndex]->NotifyStart(); // 给线程发通知启动线程执行
            } else {
                std::unique_ptr<ExecutorBase> executor;
                if (opInfo != nullptr) {
                    executor.reset(new (std::nothrow) ReduceScatterRingConcurrentDirect(
                        dispatcher_, reduceAttr, opInfo, commRing->UserRank(), subStreamsInOneRing,
                        mainSignalsInOneRing, subSignalsInOneRing, rankOrder, userMemInputSlices));
                } else {
                    executor.reset(new (std::nothrow) ReduceScatterRing(dispatcher_, reduceAttr));
                }
                CHK_SMART_PTR_NULL(executor);

                ret = LocalNotify::Wait(streamInfo->ringStreams[ringIndex], dispatcher_,
                    streamInfo->ringSignalAux[ringIndex], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CommonOperator][MultiRingReduceScatter]stream[%u] wait failed", ringIndex), ret);
                ret = executor->Prepare(inputMem, inputMem, outputMem, count, dataType,
                    streamInfo->ringStreams[ringIndex], reductionOp, OUTER_BRIDGE_RANK_ID,
                    singleRingSliceZero, baseOffset, ringNics[ringIndex]);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CommonOperator][MultiRingReduceScatter]stream[%u],reduce scatter(ring) "\
                    "prepare failed,return[%d]", ringIndex, ret), ret);
                ret = executor->RegisterProfiler(
                    ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                    (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + commRing->Rank(),
                    profStage, HCCL_EXEC_STEP_NOT_SET, streamInfo->ringStreams[ringIndex]);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CommonOperator][MultiRingReduceScatter]stream[%u],reduce scatter(ring) "\
                    "register Profiler failed,return[%d]", ringIndex, ret), ret);

                ret = commRing->RunExecutor(executor);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CommonOperator][MultiRingReduceScatter]stream[%u],reduce scatter(ring) run "\
                    "failed,return[%d]", ringIndex, ret), ret);

                ret = LocalNotify::Post(streamInfo->ringStreams[ringIndex], dispatcher_,
                    streamInfo->ringSignal[ringIndex], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CommonOperator][MultiRingReduceScatter]stream[%u] record failed", ringIndex), ret);
            }
            /* 主环record启动从环 */
            ret = LocalNotify::Post(stream, dispatcher_, streamInfo->ringSignalAux[ringIndex], profStage);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CommonOperator][MultiRingReduceScatter]stream[%u] record failed", ringIndex), ret);
        } else { // 主环 最后一个环
            std::unique_ptr<ExecutorBase> executor;
            if (opInfo != nullptr) {
                executor.reset(new (std::nothrow) ReduceScatterRingConcurrentDirect(
                    dispatcher_, reduceAttr, opInfo, commRing->UserRank(), subStreamsInOneRing, mainSignalsInOneRing,
                    subSignalsInOneRing, rankOrder, userMemInputSlices));
            } else {
                executor.reset(new (std::nothrow) ReduceScatterRing(dispatcher_, reduceAttr));
            }
            CHK_SMART_PTR_NULL(executor);
            ret = executor->Prepare(inputMem, inputMem, outputMem, count, dataType, stream,
                reductionOp, OUTER_BRIDGE_RANK_ID, singleRingSliceZero, baseOffset, ringNics[ringIndex]);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CommonOperator][MultiRingReduceScatter]stream[%u],reduce scatter(ring) prepare "\
                "failed,return[%d]", ringIndex, ret), ret);

            ret = executor->RegisterProfiler(
                ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + commRing->Rank(),
                profStage, HCCL_EXEC_STEP_NOT_SET, stream);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CommonOperator][MultiRingReduceScatter]stream[%u],reduce scatter(ring) register "\
                "Profiler failed,return[%d]", ringIndex, ret), ret);

            ret = commRing->RunExecutor(executor);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CommonOperator][MultiRingReduceScatter]stream[%u],reduce scatter(ring) run "\
                "failed,return[%d]", ringIndex, ret), ret);
            for (u32 ring = 0; ring < (ringNum - 1); ring++) {
                if (!GetExternalInputHcclEnableFfts() &&
                    GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                    streamInfo->ringThreadsManage[ring]->WaitDone();
                }
                /* 等待executor执行完毕 */
                ret = LocalNotify::Wait(stream, dispatcher_, streamInfo->ringSignal[ring], profStage);

                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CommonOperator][MultiRingReduceScatter]stream[%u] wait failed", ring), ret);
            }
        }
    }
    // 添加空task,保证子图执行时不乱序
    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}

HcclResult CommonOperator::MultiRingScatter(const std::string &tag, DeviceMem inputMem, DeviceMem outputMem,
    const u64 count, const HcclDataType dataType, const std::vector<std::vector<Slice> > multRingsSliceZero,
    u32 root, Stream stream, const HcomCollOpInfo *opInfo)
{
    HcclResult ret = HCCL_SUCCESS;
    u32 ringNum = multRingsSliceZero.size();

    innerStreamInfo_t *streamInfo = hcclImpl_->GetStreamInfo(tag);
    CHK_PRT_RET(streamInfo == nullptr,
        HCCL_ERROR("[GetStreamInfo]errNo[0x%016llx] tag[%s] can't find in stream info",
            HCCL_ERROR_CODE(HCCL_E_NOT_FOUND), tag.c_str()), HCCL_E_PARA);

    CommInfo *currComm;
    ret = hcclImpl_->GetCommInfo(currComm, tag);

    std::vector<std::vector<u32>> ringNics;
    CHK_RET(hcclImpl_->GetRingNics(tag, ringNics));

    // 拿到ring环映射关系
    u32 ranksSize = currComm->commOuter[COMM_INDEX_0]->RankSize();
    std::vector<std::vector<u32>> multiRingsOrder = GetRingsOrderByTopoType(ranksSize, topoType_, nicList_);

    // 空拷贝用于后续操作附着
    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    for (u32 ringIndex = 0; ringIndex < ringNum; ringIndex++) {
        std::vector<Slice> singleRingSliceZero = multRingsSliceZero[ringIndex];
        CHK_PRT_RET(singleRingSliceZero.empty(),
            HCCL_ERROR("[CommonOperator][MultiRingScatter]singleRingSliceZero is empty"), HCCL_E_INTERNAL);

        // 生成userMemIn_上对应的slices
        std::vector<Slice> userMemInputSlices;
        CHK_RET(
            CalUserMemSlices(dataType, opInfo, singleRingSliceZero, ringIndex, multiRingsOrder, userMemInputSlices));
        std::vector<u32> rankOrder;
        CHK_RET(GetRankOrder(multiRingsOrder, ringIndex, rankOrder));
        std::unique_ptr<CommBase> &commRing = currComm->commOuter[ringIndex];
        CHK_SMART_PTR_NULL(commRing);
        u32 rankSize = commRing->RankSize();

        std::vector<Stream>       subStreamsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> mainSignalsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> subSignalsInOneRing;
        std::unique_ptr<ExecutorBase> executor;
        if (opInfo != nullptr) {
            CHK_RET(GetSubStreamInfoOnOneRing(*streamInfo, ringIndex, subStreamsInOneRing, mainSignalsInOneRing,
                                              subSignalsInOneRing));
            executor.reset(new (std::nothrow) ScatterRingConcurrentDirect(
                dispatcher_, opInfo, commRing->UserRank(), subStreamsInOneRing,
                mainSignalsInOneRing, subSignalsInOneRing, rankOrder, userMemInputSlices));
        } else {
            executor.reset(new (std::nothrow) ScatterRing(dispatcher_));
        }
        CHK_SMART_PTR_NULL(executor);

        if (ringIndex != (ringNum - 1)) {
            if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) { // offline
                ret = StreamActiveManager::GetInstance(deviceLogicId_).StreamActive(
                    streamInfo->ringStreams[ringIndex].ptr(), stream.ptr());
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CommonOperator][MultiRingScatter]stream[%u],active stream failed", ringIndex), ret);
            }
        }

        u32 rootRank = 0;
        ret = commRing->GetRankByUserRank(root, rootRank);
        CHK_PRT_RET(ret == HCCL_E_PARA,
            HCCL_ERROR("[CommonOperator][MultiRingScatter]invalid root [%u] to get userrank", root), ret);

        if (ret == HCCL_SUCCESS) {
            if (ringIndex != (ringNum - 1)) {  // 0~ringNum-2的环
                ret = LocalNotify::Wait(streamInfo->ringStreams[ringIndex], dispatcher_,
                    streamInfo->ringSignalAux[ringIndex], PROF_STAGE_0);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CommonOperator][MultiRingScatter]in stream[%u] wait failed", ringIndex), ret);

                ret = executor->Prepare(inputMem, inputMem, outputMem, count, dataType,
                    streamInfo->ringStreams[ringIndex], HCCL_REDUCE_RESERVED, rootRank, singleRingSliceZero, 0,
                    ringNics[ringIndex]);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CommonOperator][MultiRingScatter]stream[%u],scatter(ring) prepare failed, "\
                    "return[%d]", ringIndex, ret), ret);

                ret = executor->RegisterProfiler(((ringIndex + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                    (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + commRing ->Rank(),
                    PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, streamInfo->ringStreams[ringIndex]);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CommonOperator][MultiRingScatter]stream[%u], scatter(ring) register profiler "\
                    "failed,return[%d]", ringIndex, ret), ret);

                ret = commRing->RunExecutor(executor);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CommonOperator][MultiRingScatter]stream[%u],scatter(ring) run failed, "\
                    "return[%d]", ringIndex, ret), ret);

                ret = LocalNotify::Post(streamInfo->ringStreams[ringIndex], dispatcher_,
                    streamInfo->ringSignal[ringIndex], PROF_STAGE_0);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CommonOperator][MultiRingScatter]stream[%u] record failed", ringIndex), ret);
                /* 主环record启动从环 */
                ret = LocalNotify::Post(stream, dispatcher_, streamInfo->ringSignalAux[ringIndex], PROF_STAGE_0);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CommonOperator][MultiRingScatter]stream[%u] record failed", ringIndex), ret);
            } else {  // 主环
                std::unique_ptr<ExecutorBase> executor;
                if (opInfo != nullptr) {
                    executor.reset(new (std::nothrow) ScatterRingConcurrentDirect(
                        dispatcher_, opInfo, commRing->UserRank(), subStreamsInOneRing, mainSignalsInOneRing,
                        subSignalsInOneRing, rankOrder, userMemInputSlices));
                } else {
                    executor.reset(new (std::nothrow) ScatterRing(dispatcher_));
                }
                CHK_SMART_PTR_NULL(executor);
                ret = executor->Prepare(inputMem, inputMem, outputMem, count, dataType, stream,
                    HCCL_REDUCE_RESERVED, rootRank, singleRingSliceZero, 0, ringNics[ringIndex]);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CommonOperator][MultiRingScatter]stream[%u],scatter(ring) prepare failed, "\
                    "return[%d]", ringIndex, ret), ret);
                ret = executor->RegisterProfiler(((ringIndex + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                    (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + commRing ->Rank(),
                    PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, stream);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CommonOperator][MultiRingScatter]stream[%u], scatter(ring) register profiler "\
                    "failed,return[%d]", ringIndex, ret), ret);

                ret = commRing->RunExecutor(executor);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CommonOperator][MultiRingScatter]stream[%u],scatter(ring) run failed, "\
                    "return[%d]", ringIndex, ret), ret);

                for (u32 ring = 0; ring < (ringNum - 1); ring++) {
                    /* 等待executor执行完毕 , 当前环没有分配数据，跳过此环处理，继续下一个环 */
                    ret = LocalNotify::Wait(stream, dispatcher_, streamInfo->ringSignal[ring], PROF_STAGE_0);
                    CHK_PRT_RET(ret != HCCL_SUCCESS,
                        HCCL_ERROR("[CommonOperator][MultiRingScatter]stream[%u] wait failed", ring), ret);
                }
            }
        }
    }
    // 添加空task,保证子图执行时不乱序
    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}

HcclResult CommonOperator::MultiStreamReduceScatterMeshAtomic(const std::string &tag, DeviceMem &inputMem,
    DeviceMem &outputMem, const u64 count, const HcclDataType dataType, const HcclReduceOp reductionOp,
    const std::vector<Slice> &dataSliceVct, Stream &stream,
    std::vector<std::unique_ptr<CommBase>> &commMeshVec, const u64 baseOffset, HcomCollOpInfo *opInfo)
{
    u32 unitSize = SIZE_TABLE[dataType];
    innerStreamInfo_t *streamInfo = hcclImpl_->GetStreamInfo(tag);
    CHK_PRT_RET(streamInfo == nullptr,
        HCCL_ERROR("[GetStreamInfo]errNo[0x%016llx] tag[%s] can't find in stream info",
            HCCL_ERROR_CODE(HCCL_E_NOT_FOUND), tag.c_str()), HCCL_E_PARA);

    u64 reduceAttr = GetReduceAttr(inputMem, outputMem, dataType, reductionOp);
    std::unique_ptr<ExecutorBase> executor;
    DeviceMem deviceOutputMem = inputMem;
    if (isSingleMeshAggregation_ && (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) &&
        (reduceAttr & INLINE_REDUCE_BITMASK) && (opInfo != nullptr)) {
        if (((opInfo -> count) * unitSize <= HCCL_SMALL_COUNT_32_KB) && (deviceNumPerAggregation_ == DEVICE_EIGHT)) {
            deviceOutputMem = outputMem;
            executor.reset(new (std::nothrow) ReduceScatterHDStage(dispatcher_, reduceAttr, streamInfo->ringStreams,
                streamInfo->ringSignal, streamInfo->ringSignalAux, userRank_, opInfo));
        } else {
            executor.reset(new (std::nothrow) ReduceScatterMeshDirect(dispatcher_, reduceAttr,
                streamInfo->ringStreams, streamInfo->ringSignal, streamInfo->ringSignalAux, userRank_, opInfo));
        }
    } else {
        executor.reset(
            new (std::nothrow) ReduceScatterMeshAtomic(dispatcher_, reduceAttr,
            streamInfo->ringStreams, streamInfo->ringSignal, streamInfo->ringSignalAux,
            userRank_)
        );
    }
    CHK_SMART_PTR_NULL(executor);
    std::unique_ptr<CommBase> &commPtr = commMeshVec[COMM_INDEX_0];
    CHK_SMART_PTR_NULL(commPtr);
    u32 rankSize = commPtr->RankSize();

    CHK_RET(executor->Prepare(inputMem, deviceOutputMem, outputMem, count, dataType, stream, reductionOp,
        OUTER_BRIDGE_RANK_ID, dataSliceVct, baseOffset));
    
    CHK_RET(executor->RegisterProfiler(
        (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + commPtr->Rank(),
        PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, stream));

    CHK_RET(commPtr->RunExecutor(executor));

    return HCCL_SUCCESS;
}

HcclResult CommonOperator::MultiStreamReduceScatterMesh(const std::string &tag, DeviceMem inputMem, DeviceMem outputMem,
    const u64 count, const HcclDataType dataType, const HcclReduceOp reductionOp,
    const std::vector<std::vector<Slice>>& multStreamsSlice, Stream stream,
    std::vector<std::unique_ptr<CommBase>> &commMeshVec, const u64 baseOffset)
{
    HcclResult ret = HCCL_SUCCESS;
    u64 streamNum = multStreamsSlice.size();
    HCCL_INFO("MultiStreamReduceScatterMesh streamNum[%u]", streamNum);
    innerStreamInfo_t *streamInfo = hcclImpl_->GetStreamInfo(tag);
    CHK_PRT_RET(streamInfo == nullptr,
        HCCL_ERROR("[GetStreamInfo]errNo[0x%016llx] tag[%s] can't find in stream info",
            HCCL_ERROR_CODE(HCCL_E_NOT_FOUND), tag.c_str()), HCCL_E_PARA);

    u64 reduceAttr = GetReduceAttr(inputMem, outputMem, dataType, reductionOp);

    for (u32 streamIndex = 0; streamIndex < streamNum; streamIndex++) {
        std::vector<Slice> singleStreamSlice = multStreamsSlice[streamIndex];
        CHK_PRT_RET(singleStreamSlice.size() <= 0,
            HCCL_ERROR("[CommonOperator][MultiStreamReduceScatterMesh]singleStreamSlice is empty"), HCCL_E_INTERNAL);

        std::unique_ptr<CommBase> &commMesh = commMeshVec[streamIndex];
        CHK_SMART_PTR_NULL(commMesh);

        u32 commIndex = commMesh->Rank();
        CHK_PRT_RET(commIndex >= singleStreamSlice.size(), \
            HCCL_ERROR("[CommonOperator][MultiStreamReduceScatterMesh]commIndex[%u] => singleStreamSlice size[%llu]",
                commIndex, singleStreamSlice.size()), HCCL_E_INTERNAL);

        u32 rankSize = commMesh->RankSize();
        u32 ringIndexOp = streamIndex;
        std::unique_ptr<ExecutorBase> executor;

        executor.reset(new (std::nothrow) ReduceScatterMesh(dispatcher_, reduceAttr, streamIndex));
        CHK_SMART_PTR_NULL(executor);

        if (streamIndex != (streamNum - 1)) {  // 0~ringNum-2的环
            HCCL_INFO("MultiStreamReduceScatterMesh step into subStream");
            ret = LocalNotify::Wait(streamInfo->ringStreams[streamIndex], dispatcher_,
                streamInfo->ringSignalAux[streamIndex], PROF_STAGE_0);
            // 等待executor执行完毕
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CommonOperator][MultiStreamReduceScatterMesh]stream[%u] wait failed", streamIndex), ret);

            ret = executor->Prepare(inputMem, inputMem, outputMem, count, dataType,
                streamInfo->ringStreams[streamIndex], reductionOp,
                OUTER_BRIDGE_RANK_ID, singleStreamSlice, baseOffset);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CommonOperator][MultiStreamReduceScatterMesh]stream[%u],reduce scatter(mesh) "\
                "prepare failed,return[%d]", streamIndex, ret), ret);

            ret = executor->RegisterProfiler(
                ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + \
                commMeshVec[0]->Rank(), PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET,
                streamInfo->ringStreams[streamIndex]);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CommonOperator][MultiStreamReduceScatterMesh]stream[%u],reduce scatter(mesh) "\
                "register Profiler failed,return[%d]", streamIndex, ret), ret);

            ret = commMesh->RunExecutor(executor);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CommonOperator][MultiStreamReduceScatterMesh]stream[%u],reduce scatter(mesh) run "\
                "failed,return[%d]", streamIndex, ret), ret);

            ret  = LocalNotify::Post(streamInfo->ringStreams[streamIndex], dispatcher_,
                streamInfo->ringSignal[streamIndex], PROF_STAGE_0);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CommonOperator][MultiStreamReduceScatterMesh]stream[%u] record failed", streamIndex), ret);

            ret = LocalNotify::Post(stream, dispatcher_, streamInfo->ringSignalAux[streamIndex], PROF_STAGE_0);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CommonOperator][MultiStreamReduceScatterMesh]stream[%u] record failed", streamIndex), ret);
        } else { // 主环
            HCCL_INFO("MultiStreamReduceScatterMesh step into mainStream");
            executor.reset(new (std::nothrow) ReduceScatterMesh(dispatcher_, reduceAttr, streamIndex));
            CHK_SMART_PTR_NULL(executor);

            ret = executor->Prepare(inputMem, inputMem, outputMem, count, dataType, stream,
                reductionOp, OUTER_BRIDGE_RANK_ID, singleStreamSlice, baseOffset);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CommonOperator][MultiStreamReduceScatterMesh]stream[%u],reduce scatter(mesh) prepare "\
                    "failed,return[%d]", streamIndex, ret), ret);

            ret = executor->RegisterProfiler(
                ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + \
                commMeshVec[0]->Rank(), PROF_STAGE_0,
                HCCL_EXEC_STEP_NOT_SET, stream);
            CHK_PRT_RET(ret != HCCL_SUCCESS,\
                HCCL_ERROR("[CommonOperator][MultiStreamReduceScatterMesh]stream[%u],reduce scatter(mesh) "\
                "register Profiler failed,return[%d]", streamIndex, ret), ret);

            ret = commMesh->RunExecutor(executor);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CommonOperator][MultiStreamReduceScatterMesh]stream[%u],reduce scatter(mesh) run failed, "\
                    "return[%d]", streamIndex, ret), ret);

            for (u32 streamIndex = 0; streamIndex < (streamNum - 1); streamIndex++) {
                //  等待executor执行完毕
                ret = LocalNotify::Wait(stream, dispatcher_, streamInfo->ringSignal[streamIndex], PROF_STAGE_0);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CommonOperator][MultiStreamReduceScatterMesh]stream[%u] wait failed",
                        streamIndex), ret);
            }
        }
    }
    // 添加空task,保证子图执行时不乱序
    CHK_RET(ExecutorBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return ret;
}

HcclResult CommonOperator::PrepareReduceScatterSliceData(u64 dataCount, u32 unitSize, u32 sliceNum,
    std::vector<Slice> &dataSlice)
{
    CHK_PRT_RET((sliceNum == 0), HCCL_ERROR("[Prepare][ReduceScatterSliceData]sliceNum is zero."), HCCL_E_PARA);

    dataSlice.resize(sliceNum);
    u64 sliceSize = dataCount * unitSize;
    for (u32 i = 0; i < sliceNum; i++) {
        dataSlice[i].size = sliceSize;
        dataSlice[i].offset = (i * sliceSize);
    }
    return HCCL_SUCCESS;
}

HcclResult CommonOperator::PrepareScatterRingSliceData(u64 dataCount, u32 unitSize, u32 sliceNum,
    std::vector<Slice> &dataSlice, u32 &outputOffset)
{
    CHK_PRT_RET((sliceNum == 0), HCCL_ERROR("[Prepare][PrepareScatterRingSliceData]sliceNum is zero."), HCCL_E_PARA);

    // 根据数据量算每个环上数据的偏移和大小
    CHK_RET(PrepareReduceScatterSliceData(dataCount, unitSize, sliceNum, dataSlice));

    const u32 rankNum8p = 8;
    if (topoType_ == TopoType::TOPO_TYPE_8P_RING) {
        std::vector<u32> tmpRing0 = { 0, 1, 2, 6, 5, 4, 7, 3 };
        std::vector<Slice> tempDataSegsSlice(rankNum8p);
        for (u32 i = 0; i < rankNum8p; i++) {
            tempDataSegsSlice[i].size = dataSlice[tmpRing0[i]].size;
            tempDataSegsSlice[i].offset = dataSlice[tmpRing0[i]].offset;
        }
        outputOffset = tmpRing0[outputOffset];
        dataSlice = tempDataSegsSlice;
    }
    return HCCL_SUCCESS;
}

std::vector<std::vector<Slice> > CommonOperator::PrepareMultiRingSlice(const std::vector<Slice> &dataSegsSlice,
    const std::string &tag, bool avoidCceRewrite, std::vector<u32> nicList)
{
    // get ranksSize
    CommInfo *currComm;
    hcclImpl_->GetCommInfo(currComm, tag);
    u32 ranksSize = currComm->commOuter[COMM_INDEX_0]->RankSize();
    // 获取每个ring上设备的排布顺序，顺序均为deviceID
    sort(nicList.begin(), nicList.end());
    std::vector<std::vector<u32> > multiRingsOrder = GetRingsOrderByTopoType(ranksSize, topoType_, nicList);
    std::vector<std::vector<Slice> > mutliRingsSlices;
    std::vector<std::vector<Slice> > mutliSegsSlices;
    u32 ringCount = multiRingsOrder.size();
    // 单环场景不应该走入此流程，需要在函数外校验
    CHK_PRT_RET(ringCount <= 1, HCCL_ERROR("[CommonOperator][PrepareMultiRingSlice] ringCount[%llu] <= 1",
        ringCount), mutliRingsSlices);

    u32 ringRanks = multiRingsOrder[0].size(); // 获取单个 ring 上设备的数量

    // 将数每块据切分为 ringCount 份
    HcclResult ret;
    mutliSegsSlices.reserve(dataSegsSlice.size());
    if (avoidCceRewrite) {
        ret = MutliSegSlicePrepareAvoidCceRewrite(dataSegsSlice, mutliSegsSlices, ringCount);
    } else {
        ret = MutliSegSlicePrepare(dataSegsSlice, mutliSegsSlices, ringCount);
    }
    if (ret != HCCL_SUCCESS) {
        return mutliRingsSlices;
    }
    u32 chunkSize = ringRanks / nicList.size();
    NicSendSizeCal(mutliSegsSlices, ringCount, chunkSize, nicList, tag);
    std::vector<std::vector<u32>> ringRankList;
    std::vector<Slice> singleRingSlices;
    std::vector<u32> rankList;

    ringRankList.reserve(ringCount);
    singleRingSlices.reserve(ringRanks);
    rankList.reserve(ringRanks);

    for (u32 ringIndex = 0; ringIndex < ringCount; ringIndex++) {
        for (u32 segsIndex = 0; segsIndex < ringRanks; segsIndex++) {
            u32 deviceIdx = multiRingsOrder[ringIndex][segsIndex];
            std::vector<u32>::iterator iterRank = std::find(nicList.begin(), nicList.end(), deviceIdx);
            if (iterRank != nicList.end()) {
                rankList.push_back(segsIndex);
                u32 nicPosition = distance(nicList.begin(), iterRank);
                for (u32 chunkIdx = 0; chunkIdx < chunkSize; chunkIdx++) {
                    Slice tempSlice = mutliSegsSlices[nicPosition * chunkSize + chunkIdx][ringIndex];
                    singleRingSlices.push_back(tempSlice);
                }
            }
        }
        mutliRingsSlices.push_back(singleRingSlices);
        ringRankList.push_back(rankList);
        singleRingSlices.clear();
        rankList.clear();
    }

    ret = hcclImpl_->SetRingNics(tag, ringRankList);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[Prepare][MultiRingSlice]tag[%s] set nics in ring failed, ret[%u]", tag.c_str(), ret);
        std::vector<std::vector<Slice> > emptySlice;
        return emptySlice;
    }
    return mutliRingsSlices;
}

HcclResult CommonOperator::MutliSegSlicePrepareAvoidCceRewrite(const std::vector<Slice> &dataSegsSlice,
    std::vector<std::vector<Slice> >& mutliSegsSlices, u32 ringCount) const
{
    for (u32 rankId = 0; rankId < dataSegsSlice.size(); rankId++) {
        Slice rankSliceTemp;
        std::vector<Slice> singleSegSlices;
        for (u32 ringIndex = 0; ringIndex < ringCount; ringIndex++) {
            if (ringIndex < ringCount - 1) {
                rankSliceTemp.size = 0;
                rankSliceTemp.offset = 0;
            } else {
                rankSliceTemp.size = dataSegsSlice[rankId].size;
                rankSliceTemp.offset = dataSegsSlice[rankId].offset;
            }
            singleSegSlices.push_back(rankSliceTemp);
        }
        mutliSegsSlices.push_back(singleSegSlices); // rings_slice 判断大小不为 8 则异常
    }
    return HCCL_SUCCESS;
}

HcclResult CommonOperator::MutliSegSlicePrepare(const std::vector<Slice> &dataSegsSlice,
    std::vector<std::vector<Slice> >& mutliSegsSlices, u32 ringCount)
{
    std::vector<Slice> singleSegSlices;
    singleSegSlices.reserve(ringCount);
    for (u32 rankId = 0; rankId < dataSegsSlice.size(); rankId++) {
        Slice rankSliceTemp;
        u64 rankDataSize = dataSegsSlice[rankId].size;
        u32 ringIndex = 0;
        u64 offsetStart = dataSegsSlice[rankId].offset;
        if (rankDataSize > 0) {
            u64 sizeTemp = (rankDataSize + ringCount - 1) / ringCount; /* 1是为了向上取整 */
            u64 sizePerRing = ExecutorBase::RoundUpWithDivisor(sizeTemp, HCCL_MIN_SLICE_ALIGN);
            u64 residueSize = rankDataSize;

            while (residueSize > 0) {
                u64 singleRingSize = sizePerRing < residueSize ? sizePerRing : residueSize;
                rankSliceTemp.size = singleRingSize;
                rankSliceTemp.offset = offsetStart + rankDataSize - residueSize;
                ringIndex++;
                if (singleRingSize == 0) {
                    HCCL_ERROR("[CommonOperator][MutliSegSlicePrepare]Multrings slices prepare: singleRingSize[%llu]",
                        singleRingSize);
                    return HCCL_E_INTERNAL;
                }
                residueSize -= singleRingSize;
                singleSegSlices.push_back(rankSliceTemp);
            }
        }
        while (ringIndex < ringCount) {
            rankSliceTemp.size = 0;
            rankSliceTemp.offset = offsetStart;
            ringIndex++;
            singleSegSlices.push_back(rankSliceTemp);
        }
        mutliSegsSlices.push_back(singleSegSlices); // rings_slice 判断大小不为 8 则异常
        singleSegSlices.clear();
    }
    return HCCL_SUCCESS;
}

void CommonOperator::NicSendSizeCal(const std::vector<std::vector<Slice>> &mutliSegsSlices, u32 ringCount,
    u32 chunkSize, const std::vector<u32> &nicList, const std::string &tag)
{
    // 计算每个网口最终会发送的数据量大小
    std::vector<u64> sizeList;
    sizeList.reserve(nicList.size());
    for (u32 nicIdx = 0; nicIdx < nicList.size(); nicIdx++) {
        u64 tempSize = 0;
        for (u32 chunkIdx = 0; chunkIdx < chunkSize; chunkIdx++) {
            for (u32 ringIdx = 0; ringIdx < ringCount; ringIdx++) {
                tempSize += mutliSegsSlices[nicIdx * chunkSize + chunkIdx][ringIdx].size;
            }
        }
        sizeList.push_back(tempSize);
    }
    hcclImpl_->SetNicSendSize(tag, sizeList);
}

HcclResult CommonOperator::RunExecutor(std::unique_ptr<CommBase> &commCombine, std::unique_ptr<ExecutorBase> &executor,
    DeviceMem &inputMem, DeviceMem &outputMem, u64 count, HcclDataType dataType,
    HcclReduceOp op, u32 root, Stream &stream) const
{
    CHK_SMART_PTR_NULL(executor);
    CHK_SMART_PTR_NULL(commCombine);

    CHK_RET(executor->Prepare(inputMem, outputMem, outputMem, count, dataType, stream, op, root));

    CHK_RET(commCombine->RunExecutor(executor));
    return HCCL_SUCCESS;
}

bool CommonOperator::Is2U2PInfer()
{
    return ((deviceNumPerAggregation_ == HCCL_DEVICE_NUM_TWO) && (serverNum_ == 1) &&
            (deviceType_ == DevType::DEV_TYPE_910B) && (meshAggregationRankSize_ == HCCL_DEVICE_NUM_TWO) &&
            (pairLinkCounter_[static_cast<u32>(LinkTypeInServer::HCCS_TYPE)] == 0));
}

bool CommonOperator::Is910BSingleMesh()
{
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_NP_MESH || topoType_ == TopoType::TOPO_TYPE_4P_MESH ||
                      topoType_ == TopoType::TOPO_TYPE_2P_MESH || topoType_ == TopoType::TOPO_TYPE_1P_MESH;

    bool isSingleMesh =
        (deviceType_ == DevType::DEV_TYPE_910B) && (isMeshTopo || Is2U2PInfer()) && (userRankSize_ != 1);
    return isSingleMesh;
}

bool CommonOperator::NeedCreateSingleMeshPlane(const bool isInlineReduce)
{
    // 910B 图模式非确定计算，inlineReduce使能，MESH拓扑场景下，创建一个mesh平面
    bool meshSinglePlane = Is910BSingleMesh() && topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_CONFIG_DISABLE &&
        isInlineReduce && (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);

    return meshSinglePlane;
}

bool CommonOperator::SingleMeshInlineReduce(void *inputPtr, void *outputPtr, HcclDataType dataType, HcclReduceOp op)
{
    bool isInlineReduce = IsSupportSDMAReduce(inputPtr, outputPtr, dataType, op);
    bool singleMeshInlineReduce = Is910BSingleMesh() && isInlineReduce && isSingleMeshAggregation_;
    return singleMeshInlineReduce;
}

bool CommonOperator::IsMultiMeshInlineReduce(void *inputPtr, void *outputPtr, HcclDataType dataType, HcclReduceOp op)
{
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_NP_MESH || topoType_ == TopoType::TOPO_TYPE_4P_MESH ||
                      topoType_ == TopoType::TOPO_TYPE_2P_MESH || topoType_ == TopoType::TOPO_TYPE_1P_MESH;

    bool isInlineReduce = IsSupportSDMAReduce(inputPtr, outputPtr, dataType, op);
    bool isRdmaReduce = IsSupportRDMAReduce(dataType, op);
    bool multiMeshInlineReduce = (deviceType_ == DevType::DEV_TYPE_910B) &&
                                 isMeshTopo && isInlineReduce && isRdmaReduce && (!isSingleMeshAggregation_);
    return multiMeshInlineReduce;
}

u64 CommonOperator::GetReduceAttr(DeviceMem &inputMem, DeviceMem &outputMem, HcclDataType dataType, HcclReduceOp op)
{
    u64 reduceAttr = 0;
    bool isInlineReduce = IsSupportSDMAReduce(inputMem.ptr(), outputMem.ptr(), dataType, op);
    if (isInlineReduce && inlineReduceSwitchOn_) {
        SalSetBitOne(reduceAttr, ATTR_POS_INLINE_REDUCE);
    }

    bool isRdmaReduce = IsOverFlowInfNanMode() && IsSupportRDMAReduce(dataType, op);
    if (isRdmaReduce) {
        SalSetBitOne(reduceAttr, ATTR_POS_SUPPORT_RDMA_REDUCE);
    }

    return reduceAttr;
}

u32 CommonOperator::RefreshCommIdx(u32 commIndex, std::vector<u32> nicList, u32 devicePhyId)
{
    if (GetExternalInputEnableRdmaSdmaConcurrent() && CheckRankNeighbors(nicList)) {
        std::vector<u32>::iterator iterRank = std::find(nicList.begin(), nicList.end(), devicePhyId);
        // 按照实际topo寻找对应的rankID,即commIndex
        if (iterRank != nicList.end()) {
            u32 nicPosition = distance(nicList.begin(), iterRank);
            if (commIndex != nicPosition) {
                HCCL_DEBUG(
                    "[RefreshCommIdx] old commIndex %u, new commIndex %u", commIndex, nicPosition);
                commIndex = nicPosition;
            }
        }
    }
    return commIndex;
}

}