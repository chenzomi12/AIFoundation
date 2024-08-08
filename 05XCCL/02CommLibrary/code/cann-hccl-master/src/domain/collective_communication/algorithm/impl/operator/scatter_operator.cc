/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "scatter_operator.h"
#include "device_capacity.h"
#include "rank_consistent.h"
#include "executor_impl.h"
#include "hccl_alg.h"


namespace hccl {

ScatterOperator::ScatterOperator(std::unique_ptr<hcclImpl> &pImpl, std::unique_ptr<TopoMatcher> &topoMatcher)
    : CommonOperator(pImpl, topoMatcher, HcclCMDType::HCCL_CMD_SCATTER)
{
    // 由于scatter只支持server间ring、nb和nhr，其他算法需要重定向到ring
    if (!UseInterServerNHRAlgo(algType_) && !UseInterServerNBAlgo(algType_) && !UseInterServerRingAlgo(algType_)) {
        HCCL_INFO("[ScatterOperator][ScatterOperator] algType[%s] is not supported, reset algType=ring",
            HcclAlg::AlgTypeToStr(algType_).c_str());
        SetInterServerRingAlgo(algType_);
    }
}

ScatterOperator::~ScatterOperator()
{
}

HcclResult ScatterOperator::RunScatter(const std::string &tag, DeviceMem& inputMem, DeviceMem& outputMem,
    u64 count, HcclDataType dataType, u32 root, Stream &stream)
{
    HcclResult ret;
    switch (topoType_) {
        case TopoType::TOPO_TYPE_NP_MESH:
        case TopoType::TOPO_TYPE_4P_MESH:
        case TopoType::TOPO_TYPE_2P_MESH:
        case TopoType::TOPO_TYPE_1P_MESH:
            ret = ScatterMeshExecutor(tag, inputMem, outputMem, count, dataType, root, stream);
                break;
        case TopoType::TOPO_TYPE_8P_RING:
        case TopoType::TOPO_TYPE_NP_SINGLE_RING:
        case TopoType::TOPO_TYPE_NP_DOUBLE_RING: // 当前double ring算法不支持，与single ring保持一致
            ret = ScatterRingExecutor(tag, inputMem, outputMem, count, dataType, root, stream);
            break;
        default:
            ret = ScatterComm(tag, inputMem, outputMem, count, dataType, root, stream);
            break;
    }
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][Scatter]tag[%s], scatter failed, retrun[%d]",
        tag.c_str(), ret), ret);

    return ret;
}

HcclResult ScatterOperator::Scatter(const std::string &tag, void *inputPtr, void *outputPtr, u64 recvCount,
    HcclDataType dataType, u32 root, Stream stream)
{
    HcclResult ret = HCCL_SUCCESS;
    /* ------------集合通信资源准备------------ */
    u32 perDataSize = SIZE_TABLE[dataType];
    u64 sendSize = userRankSize_ * recvCount * perDataSize;
    DeviceMem inputMem(inputPtr, sendSize);
    DeviceMem outputMem(outputPtr, recvCount * perDataSize);

    CHK_RET(hcclImpl_->PrepareCommRes(tag, inputMem, inputMem, algType_, stream, root, false, false));

    HCCL_PROFILER_ADD_STREAM(stream.ptr(), tag, 0, algType_);
    // 添加从流profiling, 用于维护planID
    CHK_RET(hcclImpl_->AddSubStreamToProfiling(tag, HcclCMDType::HCCL_CMD_SCATTER));

    /*  ------------执行算法-------------- */
    HcclUs startut = TIME_NOW();
    ret = RunScatter(tag, inputMem, outputMem, recvCount, dataType, root, stream);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[ScatterOperator][Scatter]errNo[0x%016llx] tag[%s],scatter run failed",
            HCCL_ERROR_CODE(ret), tag.c_str()), ret);

    HCCL_INFO("tag[%s],scatter run success,take time [%lld]us.", tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}


HcclResult ScatterOperator::ScatterOutPlaceForOneRankSize(const std::string &tag, void *inputPtr, void *outputPtr, u64 recvCount,
        HcclDataType dataType, u32 root, Stream stream,
        const std::unique_ptr<HcclOpBaseAtraceInfo> &opBaseAtraceInfo )
{
    bool hugeData = (recvCount * SIZE_TABLE[dataType]) > SDMA_SEND_MAX_SIZE;
    if (inputPtr == outputPtr) {
        // 通过CopyPattern字段区分不同的子图
        auto opMeta = HcclOpMetaInfo::GetOneForScatter(root, hugeData);
        CHK_RET(InitTask(dispatcher_, stream, opMeta.isEnableCache, opMeta.GetCacheKey()));
    } else {
        auto opMeta = HcclOpMetaInfo::GetOneForScatter(root, hugeData);
        CHK_RET(InitTask(dispatcher_, stream, opMeta.isEnableCache, opMeta.GetCacheKey()));
        DeviceMem srcMem(inputPtr, recvCount*SIZE_TABLE[dataType]);
        DeviceMem dstMem(outputPtr, recvCount*SIZE_TABLE[dataType]);
        HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, stream); // ranksize = 1; intput、output地址不同，input->output
    }
    CHK_RET(LaunchTask(dispatcher_, stream));
    return HCCL_SUCCESS;
}

HcclResult ScatterOperator::ScatterOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 recvCount,
    HcclDataType dataType, u32 root, Stream stream, const std::unique_ptr<HcclOpBaseAtraceInfo> &opBaseAtraceInfo)
{
    HcclResult ret;

    u8 *curInputPtr = static_cast<u8 *>(inputPtr);
    u8 *curOutputPtr = static_cast<u8 *>(outputPtr);

    if (userRankSize_ == 1 ) {
        CHK_RET(ScatterOutPlaceForOneRankSize(tag,inputPtr,outputPtr,recvCount,dataType,root,stream,opBaseAtraceInfo)) ;
        return HCCL_SUCCESS;
    }
    auto inCCLbuffer = cclBufferManager_.GetInCCLbuffer();
    auto outCCLbuffer = cclBufferManager_.GetOutCCLbuffer();

    u32 unitSize = SIZE_TABLE[dataType];
    u64 maxCountPerLoop = inCCLbuffer.size() / (userRankSize_ * unitSize); // 中转内存单次最多能够接受的output count
    u64 curCount = 0;

    std::string newTag = tag;
    if (UseInterServerHDAlgo(algType_)) {
        u32 part1Size = 2 * (moduleNum_ - (1 << static_cast<u32>(log2(moduleNum_))));
        u32 root_id = root / deviceNumPerAggregation_;
        std::string appendTag = std::to_string((root_id >= part1Size) || ((root_id % 2) == 0));
        newTag = tag + '_' + appendTag;
        if (opBaseAtraceInfo != nullptr) {
            CHK_RET(opBaseAtraceInfo->SavealgtypeTraceInfo(appendTag, tag));
        }
    }

    HCCL_PROFILER_ADD_TAG(newTag, identifier_, GetWorkflowMode());
    HCCL_PROFILER_ADD_STREAM(stream.ptr(), newTag, 0, algType_);

    for (u64 countLeft = recvCount, inputOffset = 0, outputOffset = 0; countLeft > 0; countLeft -= curCount) {
        curInputPtr += inputOffset;
        curOutputPtr += outputOffset;
        HCCL_INFO("-OP_BASE-ScatterLoop:inputOffset[%llu], outputOffset[%llu].", inputOffset, outputOffset);
        // 判断剩余数据量对应的input size是否大于中转input size
        curCount = ((countLeft * unitSize * userRankSize_) > inCCLbuffer.size()) ? maxCountPerLoop : countLeft;
        CHK_PRT_RET((curCount == 0), HCCL_ERROR("[Loop][Scatter]In OP_BASE curCount is zero"), HCCL_E_PARA);
        u64 curSize = curCount * unitSize; // 单位：字节

        bool hugeData = (curSize * userRankSize_ / HCCL_INTERNODE_MAX_DATA_RATE > RDMA_SEND_MAX_SIZE) ||
                        (curSize > SDMA_SEND_MAX_SIZE);
        auto meta = HcclOpMetaInfo::GetOneForScatter(root, hugeData);
        CHK_RET(InitTask(dispatcher_, stream, meta.isEnableCache, meta.GetCacheKey()));

        DeviceMem dstMem;
        DeviceMem srcMem;
        if (userRank_ == root) {
            // 本rank为root节点，非root节点不需要拷贝到中转内存
            for (u32 i = 0; i < userRankSize_; i++) {
                // 拷贝input上每个slice的数据到中转内存，源端每个slice的size固定为output的size
                dstMem = inCCLbuffer.range(curSize * i, curSize);
                srcMem = DeviceMem::create(curInputPtr + recvCount * unitSize * i, curSize);
                CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, stream));
            }
        }

        /* 记录指令信息用于一致性校验 */
        CHK_RET(RankConsistent::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_SCATTER, newTag.c_str(),
            curCount, dataType, root, inCCLbuffer.size(), outCCLbuffer.size()));
        HCCL_INFO("ScatterOutPlace:curInputPtr[%p], curCount[%llu], curSize[%llu].", curInputPtr, curCount, curSize);

        /* 入参的正确性由HCCL确保 */
        ret = Scatter(newTag, inCCLbuffer.ptr(), outCCLbuffer.ptr(), curCount, dataType, root, stream);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Loop][Scatter]errNo[0x%016llx] OP_BASE hcclComm scatter error, tag[%s], "
                    "input_ptr[%p], output_ptr[%p], recvCount[%llu], data_type[%s], root[%u]",
            HCCL_ERROR_CODE(ret), newTag.c_str(), inCCLbuffer.ptr(), outCCLbuffer.ptr(), curCount,
            GetDataTypeEnumStr(dataType).c_str(), root),
            ret);

        CHK_RET(RankConsistent::GetInstance().DelOpPara(newTag));

        srcMem = outCCLbuffer.range(0, curSize);
        dstMem = DeviceMem::create(curOutputPtr, curSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, stream));

        inputOffset = curSize;
        outputOffset = curSize;
        CHK_RET(LaunchTask(dispatcher_, stream));
    }
    HCCL_PROFILER_DEL_STREAM(stream.ptr());
    HCCL_PROFILER_DEL_TAG(newTag);

    return HCCL_SUCCESS;
}

HcclResult ScatterOperator::ScatterComm(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
    u64 count, HcclDataType dataType, u32 root, Stream &stream)
{
    CommInfo *currComm;
    hcclImpl_->GetCommInfo(currComm, tag);
    bool bRet = currComm->commInner.size() <= 0;
    CHK_PRT_RET(bRet, HCCL_ERROR("[ScatterOperator][ScatterComm]tag[%s],scatter op comm is empty",
        tag.c_str()), HCCL_E_INTERNAL);

    // 统一走server间（根据算法区分走NB或者是ring）
    std::unique_ptr<CommBase> &commCombine = currComm->commInner[COMM_INDEX_0];
    CHK_SMART_PTR_NULL(commCombine);
    std::unique_ptr<ExecutorBase> executor;
    // 获取root
    u32 rootRank = 0;
    CHK_RET(commCombine->GetRankByUserRank(root, rootRank));

    if (UseInterServerNBAlgo(algType_)) {
        executor.reset(new (std::nothrow) ScatterNB(dispatcher_));
        HCCL_INFO("scatter comm: using NB algo inter-server.");
        CHK_SMART_PTR_NULL(executor);
        CHK_RET(executor->Prepare(inputMem, inputMem, inputMem, count * userRankSize_, dataType, stream,
        HCCL_REDUCE_RESERVED, rootRank));
    } else if (UseInterServerNHRAlgo(algType_)) {
        executor.reset(new (std::nothrow) ScatterNHR(dispatcher_));
        HCCL_INFO("scatter comm: using NHR algo inter-server.");
        CHK_SMART_PTR_NULL(executor);
        CHK_RET(executor->Prepare(inputMem, inputMem, inputMem, count * userRankSize_, dataType, stream,
            HCCL_REDUCE_RESERVED, rootRank));
    } else {
        executor.reset(new (std::nothrow) ScatterRing(dispatcher_));
        HCCL_INFO("scatter comm: using ring algo inter-server.");
        CHK_SMART_PTR_NULL(executor);
        CHK_RET(executor->Prepare(inputMem, inputMem, inputMem, count * userRankSize_, dataType, stream,
        HCCL_REDUCE_RESERVED, rootRank));
    }

    HcclResult ret = commCombine->RunExecutor(executor);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[ScatterOperator][ScatterComm]scatter(ring) run failed, return[%d]", ret), ret);

    u8 *inputMemPtr = static_cast<u8 *>(inputMem.ptr());
    DeviceMem resultMem(inputMemPtr + outputMem.size() * commCombine->Rank(), outputMem.size());
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, outputMem, resultMem, stream));

    return HCCL_SUCCESS;
}

HcclResult ScatterOperator::SetScatterMeshPrepareMem(DeviceMem& inputMem, u64 count,
    HcclDataType dataType, u32 &commIndex, u32 root, u32 &subRoot, u32 &innerRankSize,
    CommInfo* &currComm, Stream& stream)
{
    if (innerRankSize > 1 && subRoot == userRank_) {
        CHK_SMART_PTR_NULL(currComm->commInner[commIndex]);
        std::unique_ptr<CommBase> &commInner = currComm->commInner[commIndex];
        u32 rootRankInner = 0;
        CHK_RET(commInner->GetRankByUserRank(root, rootRankInner));

        std::unique_ptr<ExecutorBase> innerExecutor;
        if (UseInterServerNBAlgo(algType_)) {
            // server间NB算法走NB
            innerExecutor.reset(new (std::nothrow) ScatterNB(dispatcher_));
            CHK_SMART_PTR_NULL(innerExecutor);
            HCCL_INFO("scatter mesh: using NB algo inter-server.");
            // 申请临时内存作为scratch内存
            CHK_RET(innerExecutor->Prepare(inputMem, inputMem, inputMem, count * userRankSize_,
                dataType, stream, HCCL_REDUCE_RESERVED, rootRankInner, std::vector<Slice>(0)));
        } else if (UseInterServerNHRAlgo(algType_)) {
            innerExecutor.reset(new (std::nothrow) ScatterNHR(dispatcher_));
            CHK_SMART_PTR_NULL(innerExecutor);
            HCCL_INFO("scatter mesh: using NHR algo inter-server.");
            CHK_RET(innerExecutor->Prepare(inputMem, inputMem, inputMem, count * userRankSize_,
                dataType, stream, HCCL_REDUCE_RESERVED, rootRankInner, std::vector<Slice>(0)));
        } else {
            innerExecutor.reset(new (std::nothrow) ScatterRing(dispatcher_));
            CHK_SMART_PTR_NULL(innerExecutor);
            HCCL_INFO("scatter mesh: using ring algo inter-server.");
            CHK_RET(innerExecutor->Prepare(inputMem, inputMem, inputMem, count * userRankSize_,
                dataType, stream, HCCL_REDUCE_RESERVED, rootRankInner, std::vector<Slice>(0))); // count是output的数据个数
        }
        CHK_RET(innerExecutor->RegisterProfiler(
            (innerRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + currComm->commInner[commIndex]->Rank(),
            PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, stream));
        CHK_RET(currComm->commInner[commIndex]->RunExecutor(innerExecutor));
    }

    return HCCL_SUCCESS;
}

HcclResult ScatterOperator::ScatterMeshExecutor(const std::string &tag, DeviceMem& inputMem, DeviceMem& outputMem,
    u64 count, HcclDataType dataType, u32 root, Stream& stream)
{
    u32 perDataSize = SIZE_TABLE[dataType];

    CommInfo *currComm;
    hcclImpl_->GetCommInfo(currComm, tag);

    bool bRet = currComm->commOuter.size() == 0;
    CHK_PRT_RET(bRet, HCCL_ERROR("[ScatterOperator][ScatterMeshExecutor]tag[%s],comm outer is empty", tag.c_str()),
        HCCL_E_INTERNAL);

    std::unique_ptr<CommBase> &commOuter = currComm->commOuter[COMM_INDEX_0];
    CHK_SMART_PTR_NULL(commOuter);

    /* ******************第一步: 节点间scatter *******************************/
    u32 commIndex = commOuter->Rank(); // 找到rank所在的节点间平面
    bRet = commIndex >= currComm->commInner.size();
    CHK_PRT_RET(bRet, HCCL_ERROR("[ScatterOperator][ScatterMeshExecutor]commIndex[%u] >=(tag[%s])comm size[%llu]", \
        commIndex, tag.c_str(), currComm->commInner.size()), HCCL_E_INTERNAL);

    CHK_SMART_PTR_NULL(currComm->commInner[commIndex]);

    u32 innerRankSize = currComm->commInner[commIndex]->RankSize();
    u32 subRoot = hcclImpl_->GetSubRootForScatter(root);
    CHK_PRT_RET(subRoot == INVALID_VALUE_RANKID, \
        HCCL_ERROR("[ScatterOperator][ScatterMeshExecutor]GetSubRootForScatter failed, ",\
        "userRank[%u], root[%u], subRoot[%u]", userRank_, root, subRoot), HCCL_E_INTERNAL);
    HCCL_DEBUG("[ScatterOperator][ScatterMeshExecutor]GetSubRootForScatter, userRank[%u], root[%u], subRoot[%u]",
        userRank_, root, subRoot);
    CHK_RET(SetScatterMeshPrepareMem(inputMem, count, dataType, commIndex, root,
        subRoot, innerRankSize, currComm, stream));

    /* *******************第二步: 节点内scatter ******************************************/
    std::vector<Slice> dataSegsSlice;
    u32 sliceNum = currComm->commOuter[COMM_INDEX_0]->RankSize();
    // 根据数据量算每个环上数据的偏移和大小
    CHK_RET(PrepareReduceScatterSliceData(count, perDataSize, sliceNum, dataSegsSlice));

    // 每个server分配的slice大小
    u64 serverSliceSize = inputMem.size() / innerRankSize;
    // 每个服务器对应的偏移
    u64 serverSliceOffset = serverSliceSize * currComm->commInner[commIndex]->Rank();
    DeviceMem scatterMeshInput = inputMem.range(serverSliceOffset, serverSliceSize);
    CHK_SMART_PTR_NULL(scatterMeshInput);
    DeviceMem scatterMeshOutput = inputMem.range(serverSliceOffset, serverSliceSize);
    CHK_SMART_PTR_NULL(scatterMeshOutput);

    std::unique_ptr<ExecutorBase> outerExecutor;
    outerExecutor.reset(
        new (std::nothrow) ScatterMesh(dispatcher_, commOuter->Rank(), commOuter->RankSize()));
    CHK_SMART_PTR_NULL(outerExecutor);

    // 偏移需要带入prepare
    u32 rootRankOuter = 0;
    HcclResult ret = commOuter->GetRankByUserRank(subRoot, rootRankOuter);

    CHK_PRT_RET(rootRankOuter == INVALID_VALUE_RANKID,
        HCCL_ERROR("[ScatterOperator][ScatterMeshExecutor]rootRankOuter[%u] is invalid, userRank[%u], subRoot[%u]",
        rootRankOuter, userRank_, subRoot), HCCL_E_INTERNAL);
    CHK_RET(outerExecutor->Prepare(scatterMeshInput, scatterMeshOutput, inputMem, count, dataType, stream,
        HCCL_REDUCE_RESERVED, rootRankOuter, dataSegsSlice, serverSliceOffset));

    ret = commOuter->RunExecutor(outerExecutor);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[ScatterOperator][ScatterMeshExecutor]scatter(mesh) run failed,return[%d]", ret), ret);

    // 将scratchMem赋值给outputMem
    u8 *scatterMeshOutputPtr = static_cast<u8 *>(scatterMeshOutput.ptr());
    DeviceMem resultMem(scatterMeshOutputPtr + outputMem.size() * commOuter->Rank(), outputMem.size());
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, outputMem, resultMem, stream));

    return HCCL_SUCCESS;
}

HcclResult ScatterOperator::SetScatterRingPrepareMem(DeviceMem& inputMem, u64 count,
    HcclDataType dataType, u32 &perDataSize, u32 &innerRankSize, u32 &commIndex, u32 root, u32 &subRoot,
    CommInfo* &currComm, Stream& stream)
{
    if (innerRankSize > 1 && subRoot == userRank_) {
        CHK_SMART_PTR_NULL(currComm->commInner[commIndex]);
        std::unique_ptr<CommBase> &commInner = currComm->commInner[commIndex];
        u32 rootRankInner = 0;
        CHK_RET(commInner->GetRankByUserRank(root, rootRankInner));

        std::unique_ptr<ExecutorBase> innerExecutor;
        if (UseInterServerNBAlgo(algType_)) {
            // server间NB算法走NB
            innerExecutor.reset(new (std::nothrow) ScatterNB(dispatcher_));
            CHK_SMART_PTR_NULL(innerExecutor);
            HCCL_INFO("scatter ring: using NB algo inter-server.");
            // 申请临时内存作为scratch内存
            CHK_RET(innerExecutor->Prepare(inputMem, inputMem, inputMem, count * userRankSize_,
                dataType, stream, HCCL_REDUCE_RESERVED, rootRankInner, std::vector<Slice>(0)));
        } else if (UseInterServerNHRAlgo(algType_)) {
            innerExecutor.reset(new (std::nothrow) ScatterNHR(dispatcher_));
            CHK_SMART_PTR_NULL(innerExecutor);
            HCCL_INFO("scatter ring: using NHR algo inter-server.");
            CHK_RET(innerExecutor->Prepare(inputMem, inputMem, inputMem, count * userRankSize_,
                dataType, stream, HCCL_REDUCE_RESERVED, rootRankInner, std::vector<Slice>(0)));
        } else {
            innerExecutor.reset(new (std::nothrow) ScatterRing(dispatcher_));
            CHK_SMART_PTR_NULL(innerExecutor);
            HCCL_INFO("scatter ring: using ring algo inter-server.");
            CHK_RET(innerExecutor->Prepare(inputMem, inputMem, inputMem, count * userRankSize_,
                dataType, stream, HCCL_REDUCE_RESERVED, rootRankInner, std::vector<Slice>(0))); // count是output的数据个数
        }
        CHK_RET(innerExecutor->RegisterProfiler(
            (innerRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + currComm->commInner[commIndex]->Rank(),
            PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, stream));
        CHK_RET(currComm->commInner[commIndex]->RunExecutor(innerExecutor));
    }

    return HCCL_SUCCESS;
}

HcclResult ScatterOperator::ScatterRingExecutor(const std::string &tag, DeviceMem& inputMem, DeviceMem& outputMem,
    u64 count, HcclDataType dataType, u32 root, Stream& stream)
{
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(dataType, perDataSize));
    CommInfo *currComm;
    hcclImpl_->GetCommInfo(currComm, tag);
    bool bRet = currComm->commOuter.size() == 0;
    CHK_PRT_RET(bRet, HCCL_ERROR("[ScatterOperator][ScatterRingExecutor]tag[%s],comm outer is empty", tag.c_str()),
        HCCL_E_INTERNAL);

    /* ***********第一步: 节点间scatter ****************************/
    u32 commIndex = (topoType_ == TopoType::TOPO_TYPE_8P_RING) ? devicePhyId_ : currComm->commOuter[0]->Rank();
    if (topoType_ != TopoType::TOPO_TYPE_8P_RING) {
        commIndex = RefreshCommIdx(commIndex, nicList_, devicePhyId_);
    }
    HCCL_DEBUG("commIndex:%u tagCommInfo_[tag].commInner.size():%llu", commIndex, currComm->commInner.size());
    bRet = commIndex >= currComm->commInner.size();
    CHK_PRT_RET(bRet, HCCL_ERROR("[ScatterOperator][ScatterRingExecutor]commIndex[%u] >=(tag[%s])comm size[%llu]", \
        commIndex, tag.c_str(), currComm->commInner.size()), HCCL_E_INTERNAL);

    CHK_SMART_PTR_NULL(currComm->commInner[commIndex]);

    u32 innerRankSize = currComm->commInner[commIndex]->RankSize();
    u32 subRoot = hcclImpl_->GetSubRootForScatter(root);
    CHK_PRT_RET(subRoot == INVALID_VALUE_RANKID, \
        HCCL_ERROR("[ScatterOperator][ScatterRingExecutor]GetSubRootForScatter failed, ", \
        "userRank[%u], root[%u], subRoot[%u]", userRank_, root, subRoot), HCCL_E_INTERNAL);
    HCCL_DEBUG("[ScatterOperator][ScatterRingExecutor]GetSubRootForScatter, userRank[%u], root[%u], subRoot[%u]",
        userRank_, root, subRoot);
    CHK_RET(SetScatterRingPrepareMem(inputMem, count, dataType, perDataSize, innerRankSize, commIndex,
        root, subRoot, currComm, stream));

    /* ***********第二步: 节点内scatter*****************************/
    std::unique_ptr<CommBase> &commOuter = currComm->commOuter[COMM_INDEX_0];
    CHK_SMART_PTR_NULL(commOuter);
    u32 sliceNum = commOuter->RankSize();
    std::vector<Slice> dataSegsSlice;
    u32 outputOffset = commOuter->Rank();
    CHK_RET(PrepareScatterRingSliceData(count, perDataSize, sliceNum, dataSegsSlice, outputOffset));

    // 每个server分配的slice大小
    u64 serverSliceSize = inputMem.size() / innerRankSize;
    // 每个服务器对应的偏移
    u64 serverSliceOffset = serverSliceSize * currComm->commInner[commIndex]->Rank();
    DeviceMem scatterMeshInput = inputMem.range(serverSliceOffset, serverSliceSize);
    CHK_SMART_PTR_NULL(scatterMeshInput);
    DeviceMem scatterMeshOutput = inputMem.range(serverSliceOffset, serverSliceSize);
    CHK_SMART_PTR_NULL(scatterMeshOutput);

    std::unique_ptr<ExecutorBase> outerExecutor;
    outerExecutor.reset(new (std::nothrow) ScatterRing(dispatcher_));
    CHK_SMART_PTR_NULL(outerExecutor);

    // 偏移需要带入prepare
    u32 rootRankOuter = 0;
    HcclResult ret = commOuter->GetRankByUserRank(subRoot, rootRankOuter);
    CHK_RET(outerExecutor->Prepare(scatterMeshInput, scatterMeshOutput, inputMem, count, dataType, stream,
        HCCL_REDUCE_RESERVED, rootRankOuter, dataSegsSlice, serverSliceOffset));

    ret = commOuter->RunExecutor(outerExecutor);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[ScatterOperator][ScatterRingExecutor]scatter(ring) run failed,return[%d]", ret), ret);

    // 将scratchMem赋值给outputMem
    u8 *scatterMeshOutputPtr = static_cast<u8 *>(scatterMeshOutput.ptr());
    DeviceMem resultMem(scatterMeshOutputPtr + outputMem.size() * outputOffset, outputMem.size());
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, outputMem, resultMem, stream));

    return HCCL_SUCCESS;
}

HcclResult ScatterOperator::SelectAlg(const std::string& tag, const OpParam& param, std::string& algName,
    std::string& newTag)
{
    HcclResult ret = HCCL_SUCCESS;
    newTag = param.tag;
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && UseInterServerHDAlgo(algType_)) {
        u32 part1Size = 2 * (moduleNum_ - (1 << static_cast<u32>(log2(moduleNum_))));
        u32 rootId = param.root / deviceNumPerAggregation_;
        std::string appendTag = std::to_string((rootId >= part1Size) || ((rootId % 2) == 0));
        newTag = newTag + '_' + appendTag;
        if (param.opBaseAtraceInfo != nullptr) {
            CHK_RET(param.opBaseAtraceInfo->SavealgtypeTraceInfo(appendTag, param.tag));
        }
    }

    // 由于scatter只支持server间ring,nb和NHR，如果不是需要重定向到ring
    if (!UseInterServerNHRAlgo(algType_) && !UseInterServerNBAlgo(algType_) && !UseInterServerRingAlgo(algType_)) {
        HCCL_INFO("[ScatterOperator][Scatter] algType[%s] is not supported, reset algType=ring",
            HcclAlg::AlgTypeToStr(algType_).c_str());
        ret = SetInterServerRingAlgo(algType_);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[ScatterOperator][Scatter]errNo[0x%016llx] tag[%s],scatter set inter server "\
                "algo failed", HCCL_ERROR_CODE(ret), newTag.c_str()), ret);
    }

    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_NP_MESH || topoType_ == TopoType::TOPO_TYPE_4P_MESH ||
        topoType_ == TopoType::TOPO_TYPE_2P_MESH || topoType_ == TopoType::TOPO_TYPE_1P_MESH;
    bool isRingTopo = topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING || topoType_ == TopoType::TOPO_TYPE_8P_RING ||
        topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING;

    if (isMeshTopo) {
        algName = "ScatterMeshExecutor";
    } else if (isRingTopo) {
        algName = "ScatterRingExecutor";
    } else {
        algName = "ScatterCommExecutor";
    }
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        newTag = newTag + algName;
        HCCL_INFO("[SelectAlg] Scatter newTag is [%s] algName is [%s]", newTag.c_str(), algName.c_str());
    }
    return HCCL_SUCCESS;
}

REGISTER_OP(HcclCMDType::HCCL_CMD_SCATTER, Scatter, ScatterOperator);
}
