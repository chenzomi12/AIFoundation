/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "reduce_operator.h"
#include "device_capacity.h"
#include "rank_consistent.h"
#include "executor_impl.h"
#include "stream_active_manager.h"


namespace hccl {

ReduceOperator::ReduceOperator(std::unique_ptr<hcclImpl> &pImpl, std::unique_ptr<TopoMatcher> &topoMatcher)
    : CommonOperator(pImpl, topoMatcher, HcclCMDType::HCCL_CMD_REDUCE)
{
    if (UseInterServerNHRAlgo(algType_) || UseInterServerNHRV1Algo(algType_) || UseInterServerNBAlgo(algType_) ||
        UseInterServerPipelineAlgo(algType_)) {
        HCCL_WARNING("[ReduceOperator][ReduceOperator] nonuniform-hierachical-ring and nonuniform-bruck and pipeline " \
        "algorithms do not support Reduce yet, reset algo to halving-doubling");
        SetInterServerHDAlgo(algType_);
    }
}

ReduceOperator::~ReduceOperator()
{
}

HcclResult ReduceOperator::RunReduce(const std::string &tag, DeviceMem& inputMem, DeviceMem& outputMem,
    u64 count, HcclDataType dataType, HcclReduceOp op, u32 root, Stream& stream)
{
    HcclResult ret;

    switch (topoType_) {
        case TopoType::TOPO_TYPE_NP_MESH:
        case TopoType::TOPO_TYPE_4P_MESH:
        case TopoType::TOPO_TYPE_2P_MESH:
        case TopoType::TOPO_TYPE_1P_MESH:
            ret = ReduceMeshExecutor(tag, inputMem, outputMem, count, dataType, op, root, stream);
            break;
        case TopoType::TOPO_TYPE_8P_RING:
        case TopoType::TOPO_TYPE_NP_SINGLE_RING:
            ret = ReduceRingPlusHd(tag, inputMem, outputMem, count, dataType, op, root, stream);
            break;
        case TopoType::TOPO_TYPE_NP_DOUBLE_RING: // 当前double ring算法不支持，与single ring保持一致
            ret = ReduceDoubleRingExecutor(tag, inputMem, outputMem, count, dataType, op, root, stream);
            break;
        default:  // 节点内节点间单环场景
            ret = ReduceComm(tag, inputMem, outputMem, count, dataType, op, root, stream);
            break;
    }
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[ReduceOperator][RunReduce]tag[%s], reduce failed, retrun[%d]",
        tag.c_str(), ret), ret);

    return ret;
}

HcclResult ReduceOperator::Reduce(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
    HcclDataType dataType, HcclReduceOp op, u32 root, Stream stream)
{
    /* ------------集合通信资源准备------------ */
    u32 perDataSize = SIZE_TABLE[dataType];

    DeviceMem inputMem(inputPtr, count * perDataSize);
    DeviceMem outputMem(outputPtr, count * perDataSize);

    bool isInlineReduce = IsSupportSDMAReduce(inputPtr, outputPtr, dataType, op);
    meshSinglePlane_ = NeedCreateSingleMeshPlane(isInlineReduce);

    CHK_RET(hcclImpl_->PrepareCommRes(tag, inputMem, outputMem, algType_, stream, root, false, false, false,
        meshSinglePlane_));
    // 添加从流profiling, 用于维护planID
    CHK_RET(hcclImpl_->AddSubStreamToProfiling(tag, HcclCMDType::HCCL_CMD_REDUCE));
    HCCL_PROFILER_ADD_STREAM(stream.ptr(), tag, 0, algType_);

    /*  ------------执行算法-------------- */
    HcclUs startut = TIME_NOW();
    HcclResult ret = RunReduce(tag, inputMem, outputMem, count, dataType, op, root, stream);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[ReduceOperator][Reduce]errNo[0x%016llx] tag[%s],reduce run failed", HCCL_ERROR_CODE(ret),
            tag.c_str()), ret);

    HCCL_INFO("tag[%s], rank[%u] root[%u] reduce run success,take time [%lld]us.", tag.c_str(), userRank_, root, DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult ReduceOperator::ReduceOutPlaceForOneRankSize(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
        HcclDataType dataType, HcclReduceOp op, u32 root, Stream stream,bool isRootRank,ReduceType reduceType,
        const std::unique_ptr<HcclOpBaseAtraceInfo> &opBaseAtraceInfo )
{
    auto rtStream = stream.ptr();
    HCCL_PROFILER_ADD_TAG(tag, identifier_, GetWorkflowMode());
    HCCL_PROFILER_ADD_STREAM(rtStream, tag, 0, algType_);
    auto originalAlgTypeLevel1 = static_cast<u32>(algType_) >> HCCL_LEVEL_ALGO_WIDTH;
    bool hugeData = (count * SIZE_TABLE[dataType]) > SDMA_SEND_MAX_SIZE;;
    if (inputPtr == outputPtr) {
        auto opMeta = HcclOpMetaInfo::GetOneForReduce(isRootRank, root,originalAlgTypeLevel1, dataType, reduceType, hugeData, CopyPattern::ZCOPY);
        CHK_RET(InitTask(dispatcher_, stream, opMeta.isEnableCache, opMeta.GetCacheKey()));
    } else {
        auto opMeta = HcclOpMetaInfo::GetOneForReduce(isRootRank, root,originalAlgTypeLevel1, dataType, reduceType, hugeData, CopyPattern::BCOPY );
        CHK_RET(InitTask(dispatcher_, stream, opMeta.isEnableCache, opMeta.GetCacheKey()));
        DeviceMem srcMem(inputPtr, count*SIZE_TABLE[dataType]);
        DeviceMem dstMem(outputPtr, count*SIZE_TABLE[dataType]);
        HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, stream); 
    }
    CHK_RET(LaunchTask(dispatcher_, stream));
    HCCL_PROFILER_DEL_STREAM(rtStream);
    HCCL_PROFILER_DEL_TAG(tag);
    return HCCL_SUCCESS;
}


HcclResult ReduceOperator::ReduceOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
    HcclDataType dataType, HcclReduceOp op, u32 root, Stream stream,
    const std::unique_ptr<HcclOpBaseAtraceInfo> &opBaseAtraceInfo)
{
    HcclResult ret;
    auto rtStream = stream.ptr();

    bool isRootRank = root == realUserRank_ ? true : false;
    ReduceType reduceType = ((op != HCCL_REDUCE_PROD) && (dataType != HCCL_DATA_TYPE_INT64)) ?
        ReduceType::INLINE_REDUCE : ReduceType::TBE_REDUCE;

    u32 unitSize = SIZE_TABLE[dataType];
    u8 *curInputPtr = static_cast<u8 *>(inputPtr);
    u8 *curOutputPtr = static_cast<u8 *>(outputPtr);
    CHK_PTR_NULL(curInputPtr);
    CHK_PTR_NULL(curOutputPtr);

    u64 inputOffset = 0;
    u64 outputOffset = 0;
    u64 countLeft = count;

    if (userRankSize_ == 1 ) {
        CHK_RET(ReduceOutPlaceForOneRankSize(tag,inputPtr,outputPtr,count,dataType,op,root,stream,isRootRank,reduceType,opBaseAtraceInfo)) ;
        return HCCL_SUCCESS;
    }

    DeviceMem dstMem;
    DeviceMem srcMem;
    auto inCCLbuffer = cclBufferManager_.GetInCCLbuffer();
    auto outCCLbuffer = cclBufferManager_.GetOutCCLbuffer();
    u64 maxCountPerLoop = inCCLbuffer.size() / unitSize; // 中转内存单次最多能够接受的output count

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

    while (countLeft > 0) {
        curInputPtr += inputOffset;
        curOutputPtr += outputOffset;
        HCCL_DEBUG("ReduceOutPlace:inputOffset[%llu], outputOffset[%llu].", inputOffset, outputOffset);
        // 单次执行操作的数据量
        u64 curCount = countLeft > maxCountPerLoop ? maxCountPerLoop : countLeft;
        u64 curSize = curCount * unitSize;                                                     // 单位 byte
        HCCL_PROFILER_ADD_TAG(newTag, identifier_, GetWorkflowMode());
        HCCL_PROFILER_ADD_STREAM(rtStream, newTag, 0, algType_);
        HCCL_PROFILER_ADD_OPDATA(newTag, count, inputPtr, outputPtr, dataType, INVALID_VALUE_RANKID, identifier_);
        HCCL_PROFILER_ADD_GROUPRANK(identifier_, userRankSize_, userRank_);
        HcomCollOpInfo opInfo = {"", inputPtr, outputPtr, curCount, dataType, 0, op};
        CHK_RET(hcclImpl_->CreateOpBasedResources(HcclCMDType::HCCL_CMD_REDUCE, newTag, opInfo));
        auto originalAlgTypeLevel1 = static_cast<u32>(algType_) >> HCCL_LEVEL_ALGO_WIDTH;
        bool hugeData = (curSize / HCCL_INTERNODE_MAX_DATA_RATE > RDMA_SEND_MAX_SIZE) ||
                        (curSize > SDMA_SEND_MAX_SIZE); // 先用cursize做memcpyAsync,所以不除以deviceNumPerAggregation_
        auto meta =
            HcclOpMetaInfo::GetOneForReduce(isRootRank, root, originalAlgTypeLevel1, dataType, reduceType, hugeData);
        CHK_RET(InitTask(dispatcher_, stream, meta.isEnableCache, meta.GetCacheKey()));
        HCCL_DEBUG("ReduceOutPlace: curInputPtr[%p], curOutputPtr[%p], curSize[%llu], curCount[%llu], dataType[%s], "
            "tag[%s].", curInputPtr, curOutputPtr, curSize, curCount, GetDataTypeEnumStr(dataType).c_str(),
            newTag.c_str());

        srcMem = DeviceMem::create(curInputPtr, curSize);
        dstMem = inCCLbuffer.range(0, curSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, stream));

        /* 记录指令信息用于一致性校验 */
        CHK_RET(RankConsistent::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_REDUCE, newTag, curCount, dataType,
            op, root, inCCLbuffer.size(), outCCLbuffer.size()));

        /* 入参的正确性由HCCL确保 */
        ret = Reduce(newTag, inCCLbuffer.ptr(), outCCLbuffer.ptr(), curCount, dataType, op, root, stream);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Loop][Reduce]errNo[0x%016llx] op_base hcclComm reduce error, tag[%s], input_ptr[%p], "
                       "output_ptr[%p], count[%llu], data_type[%s], op[%s], root[%u]",
                HCCL_ERROR_CODE(ret), newTag.c_str(), inCCLbuffer.ptr(), outCCLbuffer.ptr(), curCount,
                GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str(), root),
            ret);
        ret = RankConsistent::GetInstance().DelOpPara(newTag);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Loop][Reduce]errNo[0x%016llx] delete CMD with parameters error. tag[%s]", HCCL_ERROR_CODE(ret),
            newTag.c_str()),
            ret);

        if (realUserRank_ == root) { // 只root rank需要把数据从中转内存拷贝出去
            dstMem = DeviceMem::create(curOutputPtr, curSize);
            srcMem = outCCLbuffer.range(0, curSize);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, stream));
        }

        countLeft -= curCount;
        inputOffset = curSize;
        outputOffset = curSize;

        CHK_RET(LaunchTask(dispatcher_, stream));
        HCCL_PROFILER_DEL_STREAM(rtStream);
        HCCL_PROFILER_DEL_TAG(newTag);
        HCCL_PROFILER_DEL_OPDATA(newTag);
        HCCL_PROFILER_DEL_GROUPRANK(newTag);
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceOperator::ReduceRingPlusHd(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
    u64 count, HcclDataType dataType, HcclReduceOp op, u32 root, Stream &stream)
{
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(dataType, perDataSize));

    std::vector<Slice> dataSegsSlice; // 数据分成ranksize份，每份的起始偏移和大小
    std::vector<std::vector<Slice> > mulRingSlice; // 数据基于该rank上环0的偏移

    // step1: 节点内的reducescatter
    u32 ringNum = (topoType_ == TopoType::TOPO_TYPE_8P_RING) ? OUTER_PLANE_NUM_IN_8PRING :
        OUTER_PLANE_NUM_IN_NPRING_SINGLE;
    CommInfo *currComm;
    hcclImpl_->GetCommInfo(currComm, tag);

    bool bRet = currComm->commOuter.size() < ringNum;
    CHK_PRT_RET(bRet, HCCL_ERROR("[ReduceOperator][ReduceRingPlusHd]reduce 8Pring comm(tag:[%s]) size is[%llu],not[%u]",
        tag.c_str(), currComm->commOuter.size(), ringNum), HCCL_E_INTERNAL);

    CHK_SMART_PTR_NULL(currComm->commOuter[COMM_INDEX_0]);
    // 按ranksize得到内存切分slice数为8
    u32 sliceNum = currComm->commOuter[COMM_INDEX_0]->RankSize();
    CHK_RET(ExecutorBase::PrepareSliceData(count, perDataSize, sliceNum, 0, dataSegsSlice));

    /* 外层:reducescatter */
    // 将每slice再切分成4份，按各ring的dev顺序排列
    if (ringNum == OUTER_PLANE_NUM_IN_8PRING) {
        // 构造ring algorithm对应的reduce-scatter实例
        mulRingSlice = PrepareMultiRingSlice(dataSegsSlice, tag);
        CHK_PRT_RET(mulRingSlice.size() != ringNum, HCCL_ERROR("[ReduceOperator][ReduceRingPlusHd]ringNum[%u] "\
            "!=mulRingSlice size[%llu]", ringNum, mulRingSlice.size()), HCCL_E_INTERNAL);
    } else {
        mulRingSlice.push_back(dataSegsSlice); // 应该offset全为0，而大小和dataSegsSlice中一样,里面的offset不使用
    }

    CHK_RET(MultiRingReduceScatter(tag, inputMem, outputMem, count, dataType, op, mulRingSlice, stream,
                                   PROF_STAGE_0));

    HCCL_INFO("reduce 8PringHD stage0 run success");

    // step2: 节点间的reduce
    u64 hdSize;
    u32 segmentIdx;
    u32 commIndex;
    CHK_RET(hcclImpl_->PrepareInnerCommInfo(segmentIdx, commIndex, hdSize, currComm->commOuter, mulRingSlice, tag));
    commIndex = RefreshCommIdx(commIndex, nicList_, devicePhyId_);
    u64 hdCount = hdSize / perDataSize;

    HCCL_DEBUG("commIdx:%u TagCommInfo[%s].commInner.size():%llu", commIndex, tag.c_str(),
        currComm->commInner.size());

    bRet = commIndex >= currComm->commInner.size();
    CHK_PRT_RET(bRet, HCCL_ERROR("[ReduceOperator][ReduceRingPlusHd]commIndex[%u] >= (tag:[%s])comm_inner.size[%llu]", \
        commIndex, tag.c_str(), currComm->commInner.size()), HCCL_E_INTERNAL);

    CHK_SMART_PTR_NULL(currComm->commInner[commIndex]);
    DeviceMem reduceInput = inputMem.range(dataSegsSlice[segmentIdx].offset, hdSize);
    CHK_SMART_PTR_NULL(reduceInput);
    DeviceMem reduceOutput = outputMem.range(dataSegsSlice[segmentIdx].offset, hdSize);
    CHK_SMART_PTR_NULL(reduceOutput);

    u64 reduceAttr = GetReduceAttr(reduceInput, reduceOutput, dataType, op);

    std::unique_ptr<ExecutorBase> innerExecutor;
    if (UseInterServerRingAlgo(algType_)) {
        innerExecutor.reset(new (std::nothrow) ReduceRing(dispatcher_, reduceAttr));
    } else {
        innerExecutor.reset(new (std::nothrow) ReduceRecursiveHalvingDoubling(dispatcher_, reduceAttr));
    }
    CHK_SMART_PTR_NULL(innerExecutor);

    u32 subUserrankRoot = hcclImpl_->GetSubRootUserRank(userRank_, root);
    CHK_PRT_RET(subUserrankRoot == INVALID_VALUE_RANKID,
        HCCL_ERROR("[ReduceOperator][ReduceRingPlusHd]subUserrankRoot[%u] is invalid,userRank[%u],root[%u]",
            subUserrankRoot, userRank_, root), HCCL_E_INTERNAL);
    std::unique_ptr<CommBase> &commInner = currComm->commInner[commIndex];
    u32 planeRoot = 0;
    CHK_RET(commInner->GetRankByUserRank(subUserrankRoot, planeRoot));

    u32 ranksize = commInner->RankSize();
    // 节点间的hd 使用环0来记录
    CHK_RET(innerExecutor->Prepare(reduceInput, reduceOutput, reduceOutput, hdCount, dataType, stream,
        op, planeRoot, std::vector<Slice>(0), dataSegsSlice[segmentIdx].offset));

    CHK_RET(innerExecutor->RegisterProfiler((ranksize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + commInner->Rank(), \
        PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, stream));

    CHK_RET(commInner->RunExecutor(innerExecutor));

    HCCL_INFO("reduce 8PringHD stage1 run success");

    // step3: 节点内的gatherring，只有在root所在server内进行gather操作
    u32 rootRank = 0;
    std::unique_ptr<CommBase> &commOuter = currComm->commOuter[COMM_INDEX_0];
    CHK_SMART_PTR_NULL(commOuter);
    HcclResult ret = commOuter->GetRankByUserRank(root, rootRank);
    CHK_PRT_RET(ret == HCCL_E_PARA,
        HCCL_ERROR("[ReduceOperator][ReduceRingPlusHd]invalid root rank[%u] to get user rank", root), ret);
    if (ret == HCCL_SUCCESS) {
        CHK_RET(MultiRingGather(tag, outputMem, outputMem, hdCount, dataType, mulRingSlice, op, root, stream,
            PROF_STAGE_2));
    }
    HCCL_INFO("reduce 8PringHD stage2 run success");
    return HCCL_SUCCESS;
}

HcclResult ReduceOperator::ReduceComm(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem, u64 count,
    HcclDataType dataType, HcclReduceOp op, u32 root, Stream &stream)
{
    CommInfo *currComm;
    hcclImpl_->GetCommInfo(currComm, tag);
    if (currComm->commInner.size() <= 0) {
        HCCL_ERROR("[ReduceOperator][ReduceComm]errNo[0x%016llx] tag[%s],reduce op inner comm is empty",
            HCCL_ERROR_CODE(HCCL_E_INTERNAL), tag.c_str());
        return HCCL_E_INTERNAL;
    }

    // 获取ring algorithm所需的通信连接
    std::unique_ptr<CommBase> &commCombine = currComm->commInner[COMM_INDEX_0];
    CHK_SMART_PTR_NULL(commCombine);

    u64 reduceAttr = GetReduceAttr(inputMem, outputMem, dataType, op);

    // 构造ring algorithm对应的reduce实例
    std::unique_ptr<ExecutorBase> executor(new (std::nothrow) ReduceRing(dispatcher_, reduceAttr));
    CHK_SMART_PTR_NULL(executor);

    // 获取root
    u32 rootRank = 0;
    CHK_RET(commCombine->GetRankByUserRank(root, rootRank));

    CHK_RET(RunExecutor(commCombine, executor, inputMem, outputMem, count, dataType,
                        op, rootRank, stream));

    return HCCL_SUCCESS;
}

HcclResult ReduceOperator::ReduceMeshExecutor(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
    u64 count, HcclDataType dataType, HcclReduceOp op, u32 root, Stream &stream, HcomCollOpInfo *opInfo)
{
    CommInfo *currComm;
    hcclImpl_->GetCommInfo(currComm, tag);

    bool bRet = currComm->commOuter.size() == 0;
    CHK_PRT_RET(bRet, HCCL_ERROR("[ReduceOperator][ReduceMeshExecutor]tag[%s], comm outer is empty", tag.c_str()),
        HCCL_E_INTERNAL);

    innerStreamInfo_t *streamInfo = hcclImpl_->GetStreamInfo(tag);
    CHK_PRT_RET(streamInfo == nullptr,
        HCCL_ERROR("[GetStreamInfo]errNo[0x%016llx] tag[%s] can't find in stream info",
            HCCL_ERROR_CODE(HCCL_E_NOT_FOUND), tag.c_str()), HCCL_E_PARA);

    // step1：节点内reducescatter mesh
    CHK_SMART_PTR_NULL(currComm->commOuter[COMM_INDEX_0]);
    u32 sliceNum = currComm->commOuter[COMM_INDEX_0]->RankSize();
    std::vector<Slice> dataSegsSlice;   // 数据分成ranksize份，每份的起始偏移和大小
    u32 perDataSize = SIZE_TABLE[dataType];
    // 根据数据量算每个环上数据的偏移和大小
    CHK_RET(ExecutorBase::PrepareSliceData(count, perDataSize, sliceNum, 0, dataSegsSlice));

    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) { // offline
        for (u32 streamIndex = 0; streamIndex < streamInfo->ringStreams.size(); streamIndex++) {
            CHK_RET(StreamActiveManager::GetInstance(deviceLogicId_).StreamActive(
                streamInfo->ringStreams[streamIndex].ptr(), stream.ptr()));
        }
    }
    std::vector<std::unique_ptr<CommBase>> &commMeshVec = currComm->commOuter;
    if (topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_CONFIG_DISABLE && (dataType != HCCL_DATA_TYPE_INT64) &&
        (deviceType_ == DevType::DEV_TYPE_910B && op != HCCL_REDUCE_PROD)) {
        CHK_RET(MultiStreamReduceScatterMeshAtomic(tag, inputMem, outputMem, count, dataType, op,
            dataSegsSlice, stream, commMeshVec));
    } else {
        std::vector<std::vector<Slice>> multiStreamSlice; // 每个stream使用的数据基于用户buffer的偏移
        // mesh算法stream数量为rank数减1
        CHK_RET(ExecutorBase::PrepareSliceMeshStreams(dataSegsSlice, sliceNum - 1, multiStreamSlice));
        CHK_RET(MultiStreamReduceScatterMesh(tag, inputMem, outputMem, count, dataType, op, multiStreamSlice,
            stream, commMeshVec));
    }
    HCCL_INFO("reduce mesh stage0 run success");

    // step2: 节点间的reduce
    u32 commIndex = currComm->commOuter[COMM_INDEX_0]->Rank();
    bRet = commIndex >= dataSegsSlice.size();
    CHK_PRT_RET(bRet, HCCL_ERROR("[ReduceOperator][ReduceMeshExecutor]commIndex[%u] >=  dataSegsSlice size[%llu]",
        commIndex, dataSegsSlice.size()), HCCL_E_INTERNAL);

    u64 hdCount = dataSegsSlice[commIndex].size / perDataSize;

    HCCL_DEBUG("commIdx:%u TagCommInfo[%s].commInner.size():%llu", commIndex, tag.c_str(),
        currComm->commInner.size());

    bRet = commIndex >= currComm->commInner.size();
    CHK_PRT_RET(bRet, HCCL_ERROR("[ReduceOperator][ReduceMeshExecutor]commIndex[%u] >= (tag:[%s])comm_inner.size[%llu]",
        commIndex, tag.c_str(), currComm->commInner.size()), HCCL_E_INTERNAL);

    DeviceMem reduceInput = inputMem.range(dataSegsSlice[commIndex].offset, dataSegsSlice[commIndex].size);
    CHK_SMART_PTR_NULL(reduceInput);
    DeviceMem reduceOutput = outputMem.range(dataSegsSlice[commIndex].offset, dataSegsSlice[commIndex].size);
    CHK_SMART_PTR_NULL(reduceOutput);

    CHK_SMART_PTR_NULL(currComm->commInner[commIndex]);
    std::unique_ptr<CommBase> &commInner = currComm->commInner[commIndex];
    if (commInner->RankSize() > 1) {
        u64 reduceAttr = GetReduceAttr(reduceInput, reduceOutput, dataType, op);

        u32 subUserrankRoot = hcclImpl_->GetSubRootUserRank(userRank_, root);
        CHK_PRT_RET(subUserrankRoot == INVALID_VALUE_RANKID,
            HCCL_ERROR("[ReduceOperator][ReduceMeshExecutor]subUserrankRoot[%u] is invalid,userRank[%u],root[%u]",
                subUserrankRoot, userRank_, root), HCCL_E_INTERNAL);
        u32 planeRoot = 0;
        CHK_RET(commInner->GetRankByUserRank(subUserrankRoot, planeRoot));

        std::unique_ptr<ExecutorBase> innerExecutor;
        if (UseInterServerRingAlgo(algType_)) {
            innerExecutor.reset(new (std::nothrow) ReduceRing(dispatcher_, reduceAttr));
        } else {
            innerExecutor.reset(new (std::nothrow) ReduceRecursiveHalvingDoubling(dispatcher_, reduceAttr));
        }
        CHK_SMART_PTR_NULL(innerExecutor);
        // 节点间的hd 使用环0来记录
        CHK_RET(innerExecutor->Prepare(reduceInput, reduceOutput, reduceOutput, hdCount, dataType, stream,
            op, planeRoot, std::vector<Slice>(0), dataSegsSlice[commIndex].offset));

        u32 ranksize = commInner->RankSize();
        CHK_RET(innerExecutor->RegisterProfiler((ranksize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + commInner->Rank(),
            PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, stream));

        CHK_RET(commInner->RunExecutor(innerExecutor));
    } else {
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, reduceOutput, reduceInput, stream));
    }

    HCCL_INFO("reduce mesh stage1 run success");

    // step3: 节点内的gathermesh
    std::unique_ptr<CommBase> &commOuter = currComm->commOuter[COMM_INDEX_0];
    CHK_SMART_PTR_NULL(commOuter);
    std::unique_ptr<ExecutorBase> outerExecutor;
    outerExecutor.reset(new (std::nothrow) GatherMesh(dispatcher_, streamInfo->ringStreams,
            streamInfo->ringSignal, streamInfo->ringSignalAux, userRank_));
    CHK_SMART_PTR_NULL(outerExecutor);
    u32 rootRank = 0;
    HcclResult ret = commOuter->GetRankByUserRank(root, rootRank);
    CHK_PRT_RET(ret == HCCL_E_PARA,
        HCCL_ERROR("[ReduceOperator][ReduceMeshExecutor]invalid root rank[%u] to get user rank", root), ret);
    if (ret == HCCL_SUCCESS) {
        CHK_RET(outerExecutor->Prepare(outputMem, outputMem, inputMem, count, dataType, stream, op, rootRank,
            dataSegsSlice));
        u32 rankSize = commOuter->RankSize();
        CHK_RET(outerExecutor->RegisterProfiler(
            (0 << PROF_RINGINDEX_OFFSET_OF_PLANEID)+(rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) +
            commOuter->Rank(), PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, stream));
        CHK_RET(commOuter->RunExecutor(outerExecutor));
    }
    HCCL_INFO("reduce mesh stage2 run success");
    return HCCL_SUCCESS;
}

HcclResult ReduceOperator::ReduceDoubleRingExecutor(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
    u64 count, HcclDataType dataType, HcclReduceOp op, u32 root, Stream &stream, HcomCollOpInfo *opInfo)
{
    u32 ringNum = OUTER_PLANE_NUM_IN_NPRING_DOUBLE;
    CommInfo *currComm;
    hcclImpl_->GetCommInfo(currComm, tag);
    bool bRet = currComm->commOuter.size() < ringNum;
    CHK_PRT_RET(bRet, HCCL_ERROR("[ReduceDoubleRingExecutor]ringNum[%u] > ",
        "(tag[%s])comm outer count[%llu]", ringNum, tag.c_str(), currComm->commOuter.size()), HCCL_E_INTERNAL);
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(dataType, perDataSize));
    CHK_SMART_PTR_NULL(currComm->commOuter[COMM_INDEX_0]);
    u32 sliceNum = currComm->commOuter[COMM_INDEX_0]->RankSize();
    std::vector<Slice> dataSegsSlice;   // 数据分成ranksize份，每份的起始偏移和大小
    // 根据数据量算每个环上数据的偏移和大小
    CHK_RET(ExecutorBase::PrepareSliceData(count, perDataSize, sliceNum, 0, dataSegsSlice));
    /* STAGE0: 节点内 reduce-scatter */
    // 数据基于该rank上环0的偏移
    std::vector<std::vector<Slice>> multiRingsSliceZero = PrepareMultiRingSlice(dataSegsSlice, tag, false, nicList_);
    CHK_PRT_RET(multiRingsSliceZero.size() != ringNum, HCCL_ERROR("[ReduceDoubleRingExecutor]ringNum[%u] "\
        "!=multiRingsSliceZero size[%llu]", ringNum, multiRingsSliceZero.size()), HCCL_E_INTERNAL);
    CHK_RET(MultiRingReduceScatter(tag, inputMem, outputMem, count, dataType, op, multiRingsSliceZero, stream,
        PROF_STAGE_0));
    HCCL_INFO("[ReduceDoubleRingExecutor]stage0 run success");
    u32 commIndex = 0;
    u64 level1Size = 0;
    u32 segmentIdx = 0;
    CHK_RET(hcclImpl_->PrepareInnerCommInfo(segmentIdx, commIndex, level1Size, currComm->commOuter, multiRingsSliceZero, tag));
    u64 level1Count = level1Size / perDataSize;
    if (devNumInLevel2_ <= 1) {
        bRet = commIndex >= currComm->commInner.size();
        CHK_PRT_RET(bRet, HCCL_ERROR("[ReduceDoubleRingExecutor]commIndex[%u] >= (tag:[%s])comm_inner.size[%llu]",
            commIndex, tag.c_str(), currComm->commInner.size()), HCCL_E_INTERNAL);
        DeviceMem reduceInput = inputMem.range(dataSegsSlice[segmentIdx].offset, level1Size);
        CHK_SMART_PTR_NULL(reduceInput);
        DeviceMem reduceOutput = outputMem.range(dataSegsSlice[segmentIdx].offset, level1Size);
        CHK_SMART_PTR_NULL(reduceOutput);
        u64 reduceAttr = GetReduceAttr(reduceInput, reduceOutput, dataType, op);
        std::unique_ptr<ExecutorBase> innerExecutor;
        if (UseInterServerRingAlgo(algType_)) {
            innerExecutor.reset(new (std::nothrow) ReduceRing(dispatcher_, reduceAttr));
            HCCL_INFO("[ReduceDoubleRingExecutor]using ring algo inter-server.");
        } else {
            innerExecutor.reset(new (std::nothrow) ReduceRecursiveHalvingDoubling(dispatcher_, reduceAttr));
            HCCL_INFO("[ReduceDoubleRingExecutor]using Recursive halving-doubling algo inter-server.");
        }
        CHK_SMART_PTR_NULL(currComm->commInner[commIndex]);
        u32 rankSize = (currComm->commInner[commIndex]->RankSize());
        u32 subUserrankRoot = hcclImpl_->GetSubRootUserRank(userRank_, root);
        CHK_PRT_RET(subUserrankRoot == INVALID_VALUE_RANKID,
            HCCL_ERROR("[ReduceDoubleRingExecutor]subUserrankRoot[%u] is invalid,userRank[%u],root[%u]",
            subUserrankRoot, userRank_, root), HCCL_E_INTERNAL);
        std::unique_ptr<CommBase> &commInner = currComm->commInner[commIndex];
        u32 planeRoot = 0;
        CHK_RET(commInner->GetRankByUserRank(subUserrankRoot, planeRoot));
        // 节点间的hd 使用环0来记录
        CHK_SMART_PTR_NULL(innerExecutor);
        CHK_RET(innerExecutor->Prepare(
            reduceInput, reduceOutput, reduceOutput, level1Count, dataType, stream, op, planeRoot,
            std::vector<Slice>(0), dataSegsSlice[segmentIdx].offset));
        CHK_RET(innerExecutor->RegisterProfiler(
            (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + commInner->Rank(),
            PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, stream));
        CHK_RET(commInner->RunExecutor(innerExecutor));
    } else {
        //节点间 reduce scatter
        CHK_RET(ExecutorBase::PrepareSliceData(level1Count, perDataSize, sliceNum, 0, dataSegsSlice));
        bRet = commIndex >= currComm->commInner.size();
        CHK_PRT_RET(bRet, HCCL_ERROR("[ReduceDoubleRingExecutor] commIndex[%u] >= ",\
            "(tag[%s])comm size[%llu]", commIndex, tag.c_str(), currComm->commInner.size()), HCCL_E_INTERNAL);
        CHK_SMART_PTR_NULL(currComm->commInner[commIndex]);
        DeviceMem reducescatterInput = inputMem.range(dataSegsSlice[segmentIdx].offset, level1Size);
        DeviceMem reducescatterOutput = outputMem.range(dataSegsSlice[segmentIdx].offset, level1Size);
        u64 reduceAttr = GetReduceAttr(reducescatterInput, reducescatterOutput, dataType, op);
        std::unique_ptr<ExecutorBase> level1RSExecutor;
        u32 subUserrankRoot = hcclImpl_->GetSubRootUserRank(userRank_, root);
        u32 planeRoot = 0;
        std::unique_ptr<CommBase> &commInner = currComm->commInner[commIndex];
        CHK_RET(commInner->GetRankByUserRank(subUserrankRoot, planeRoot));
        if (UseInterServerRingAlgo(algType_)) {
            level1RSExecutor.reset(new (std::nothrow) ReduceScatterRing(dispatcher_, reduceAttr));
            CHK_SMART_PTR_NULL(level1RSExecutor);
            CHK_RET(level1RSExecutor->Prepare(
                reducescatterInput, reducescatterInput, reducescatterOutput, level1Count, dataType, stream, op,
                planeRoot, dataSegsSlice, dataSegsSlice[segmentIdx].offset));
            HCCL_INFO("[ReduceDoubleRingExecutor]reducescatter ring: using ring algo inter-server.");
        } else {
            level1RSExecutor.reset(new (std::nothrow) ReduceScatterRecursiveHalvingDoubling(dispatcher_, reduceAttr));
            CHK_SMART_PTR_NULL(level1RSExecutor);
            CHK_RET(level1RSExecutor->Prepare(
                reducescatterInput, reducescatterOutput, reducescatterOutput, level1Count, dataType, stream, op,
                planeRoot, dataSegsSlice, dataSegsSlice[segmentIdx].offset));
            HCCL_INFO("[ReduceDoubleRingExecutor]reducescatter ring: using halving-doubling algo inter-server.");
        }
        CHK_RET(level1RSExecutor->RegisterProfiler(
            (sliceNum << PROF_RANKSIZE_OFFSET_OF_PLANEID) + currComm->commInner[commIndex]->Rank(),
            PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, stream));
        CHK_RET(currComm->commInner[commIndex]->RunExecutor(level1RSExecutor));
        HCCL_INFO("[ReduceDoubleRingExecutor]reduce double ring [superpod] level1 reduce-scatter run success");
        // 超节点 reduce
        u64 rSize;
        std::vector<std::vector<Slice>> rdSlice;
        rdSlice.push_back(dataSegsSlice);
        CHK_RET(hcclImpl_->PrepareInnerCommInfo(segmentIdx, commIndex, rSize, currComm->commInner, rdSlice, tag));
        u64 arCount = rSize / perDataSize;
        bRet = commIndex >= currComm->commLevel2.size();
        CHK_PRT_RET(bRet, HCCL_ERROR("[ReduceDoubleRingExecutor] commIndex[%u] >= ",\
            "(tag[%s])comm size[%llu]", commIndex, tag.c_str(), currComm->commLevel2.size()), HCCL_E_INTERNAL);
        std::unique_ptr<CommBase> &commSuperpod = currComm->commLevel2[commIndex];
        CHK_PTR_NULL(commSuperpod);
        u32 subUserrankRootSupperPod = hcclImpl_->GetSubRootUserRankWithSuperPod(userRank_, root);
        u32 planeRootSupperPod = 0;
        CHK_RET(commSuperpod->GetRankByUserRank(subUserrankRootSupperPod,planeRootSupperPod));
        u32 rankSize = currComm->commLevel2[COMM_INDEX_0]->RankSize();
        DeviceMem reduceInput = inputMem.range(dataSegsSlice[segmentIdx].offset, rSize);
        DeviceMem reduceOutput = outputMem.range(dataSegsSlice[segmentIdx].offset, rSize);
        reduceAttr = GetReduceAttr(reduceInput, reduceOutput, dataType, op);
        std::unique_ptr<ExecutorBase> level2RExecutor;
        if (UseLevel2RingAlgo(algType_)) {
            level2RExecutor.reset(new (std::nothrow) ReduceRing(dispatcher_, reduceAttr));
            HCCL_INFO("[ReduceDoubleRingExecutor]reducescatter ring: using ring algo inter-server.");
        } else {
            level2RExecutor.reset(new (std::nothrow) ReduceRecursiveHalvingDoubling(dispatcher_, reduceAttr));
            HCCL_INFO("[ReduceDoubleRingExecutor]reducescatter ring: using halving-doubling algo inter-server.");
        }
        CHK_RET(level2RExecutor->Prepare(
            reduceInput, reduceOutput, reduceOutput, arCount, dataType, stream, op, planeRootSupperPod,
            std::vector<Slice>(0), dataSegsSlice[segmentIdx].offset));
        CHK_SMART_PTR_NULL(level2RExecutor);
        CHK_RET(level2RExecutor->RegisterProfiler(
            (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + currComm->commLevel2[commIndex]->Rank(),
            PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, stream));
        CHK_RET(currComm->commLevel2[commIndex]->RunExecutor(level2RExecutor));
        HCCL_INFO("[ReduceDoubleRingExecutor]reduce double ring [superpod] level2 reduce run success");
        // 节点间 gather
        std::unique_ptr<ExecutorBase> level1GExecutor;
        DeviceMem gatherInput = outputMem.range(dataSegsSlice[segmentIdx].offset, rSize);
        DeviceMem gatherOutput = outputMem.range(dataSegsSlice[segmentIdx].offset, rSize*sliceNum);
        level1GExecutor.reset(new (std::nothrow) GatherRing(dispatcher_));
        CHK_SMART_PTR_NULL(level1GExecutor);
        CHK_RET(level1GExecutor->Prepare(gatherOutput, gatherOutput, gatherOutput, arCount, dataType, stream,
            HcclReduceOp::HCCL_REDUCE_RESERVED, planeRoot, dataSegsSlice,
            dataSegsSlice[segmentIdx].offset));
        CHK_RET(level1GExecutor->RegisterProfiler(
            (sliceNum << PROF_RANKSIZE_OFFSET_OF_PLANEID) + currComm->commInner[commIndex]->Rank(),
            PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, stream));
        CHK_RET(currComm->commInner[commIndex]->RunExecutor(level1GExecutor));
        HCCL_INFO("[ReduceDoubleRingExecutor]reduce double ring [superpod] level1 gather run success");
    }
    HCCL_INFO("[ReduceDoubleRingExecutor]stage1 run success");
    u32 rootRank = 0;
    std::unique_ptr<CommBase> &commOuter = currComm->commOuter[COMM_INDEX_0];
    CHK_SMART_PTR_NULL(commOuter);
    HcclResult ret = commOuter->GetRankByUserRank(root, rootRank);
    CHK_PRT_RET(ret == HCCL_E_PARA,
        HCCL_ERROR("[ReduceDoubleRingExecutor]invalid root rank[%u] to get user rank", root), ret);
    if (ret == HCCL_SUCCESS) {
        CHK_RET(MultiRingGather(tag, inputMem, outputMem, level1Count, dataType, multiRingsSliceZero, op, root, stream,
            PROF_STAGE_2));
    }
    HCCL_INFO("[ReduceDoubleRingExecutor]reduce double ring stage2 run success");
    return HCCL_SUCCESS;
}
 
HcclResult ReduceOperator::SelectAlg(const std::string &tag, const OpParam &param, std::string &algName,
    std::string &newTag)
{
    HcclResult ret = HCCL_SUCCESS;

    if (userRankSize_ == 1 && GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        algName = "ReduceSingleExecutor";
        return HCCL_SUCCESS;
    }
 
    if (deviceType_ == DevType::DEV_TYPE_910) {
        ret = SelectAlgfor910A(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910B) {
        ret = SelectAlgfor910B(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910_73) {
        ret = SelectAlgfor91073(param, algName);
    } else {
        HCCL_ERROR("[SelectAlg] device type[%d] is out of range for selector.", deviceType_);
        return HCCL_E_NOT_SUPPORT;
    }
 
    AlgTypeLevel1 algType1 = GetLevel1AlgType(algType_);
        auto level1Iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType1);
        newTag = tag + level1Iter->second + algName;
    
    HCCL_INFO("[SelectAlg] reduce newTag is [%s]", newTag.c_str());
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[ReduceSelector][SelectAlg]tag[%s], reduce failed, return[%d]", tag.c_str(), ret), ret);
    return ret;
}
 
HcclResult ReduceOperator::SelectAlgfor910A(const OpParam& param, std::string& algName)
{
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_4P_MESH || topoType_ == TopoType::TOPO_TYPE_2P_MESH;
    bool isRingTopo = topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING || topoType_ == TopoType::TOPO_TYPE_8P_RING;
 
    if (isMeshTopo) {
        algName = "ReduceMeshExecutor";
    } else if (isRingTopo) {
        algName = "ReduceRingPlusHd";
    } else {
        algName = "ReduceComm";
    }

    HCCL_INFO("[SelectAlgfor910A] reduce SelectAlgfor910A is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}
 
HcclResult ReduceOperator::SelectAlgfor910B(const OpParam& param, std::string& algName)
{
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_NP_MESH || topoType_ == TopoType::TOPO_TYPE_4P_MESH ||
        topoType_ == TopoType::TOPO_TYPE_2P_MESH || topoType_ == TopoType::TOPO_TYPE_1P_MESH;
    bool isRingTopo = topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING;
 
    if (isMeshTopo) {
        algName = "ReduceMeshExecutor";
    } else if (isRingTopo) {
        algName = "ReduceRingPlusHd";
    } else {
        algName = "ReduceComm";
    }

    HCCL_INFO("[SelectAlgfor910B] reduce SelectAlgfor910B is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}
 
HcclResult ReduceOperator::SelectAlgfor91073(const OpParam& param, std::string& algName)
{
    // 当前double ring算法不支持，与single ring保持一致
    if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
        algName = "ReduceDoubleRingExecutor";
    } else {
        algName = "ReduceComm";
    }
    HCCL_INFO("[SelectAlgfor91073] areduce SelectAlgfor91073 is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

REGISTER_OP(HcclCMDType::HCCL_CMD_REDUCE, Reduce, ReduceOperator);

}