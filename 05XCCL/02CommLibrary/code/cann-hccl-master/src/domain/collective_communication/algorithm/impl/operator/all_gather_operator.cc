/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_gather_operator.h"
#include "device_capacity.h"
#include "rank_consistent.h"
#include "executor_impl.h"
#include "coll_alg_op_registry.h"

namespace hccl {
AllGatherOperator::AllGatherOperator(std::unique_ptr<hcclImpl> &pImpl, std::unique_ptr<TopoMatcher> &topoMatcher)
    : CommonOperator(pImpl, topoMatcher, HcclCMDType::HCCL_CMD_ALLGATHER)
{
}

AllGatherOperator::~AllGatherOperator()
{
}

HcclResult AllGatherOperator::AllGather(const std::string &tag, void *inputPtr, void *outputPtr, u64 inputCount,
    HcclDataType dataType, Stream stream, HcomCollOpInfo *opInfo)
{
    u32 perDataSize = SIZE_TABLE[dataType];

    DeviceMem inputMem(inputPtr, inputCount * perDataSize);
    DeviceMem outputMem(outputPtr, inputCount * perDataSize * userRankSize_);
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_NP_MESH || topoType_ == TopoType::TOPO_TYPE_4P_MESH ||
        topoType_ == TopoType::TOPO_TYPE_2P_MESH || topoType_ == TopoType::TOPO_TYPE_1P_MESH;

    // 910_73超节点只支持server间ring,NB和NHR，默认需继续使用NHR
    if (!(UseInterServerRingAlgo(algType_) || UseInterServerNBAlgo(algType_)) &&
        deviceType_ == DevType::DEV_TYPE_910_73) {
        HcclResult ret = SetInterServerNHRAlgo(algType_);
        HCCL_WARNING("[AllGatherOperator][AllGather] only support ring, NB and NHR in AlgoLevel1 yet, "\
            "default is algType=NHR.");
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[AllGatherOperator][AllGather]errNo[0x%016llx] tag[%s],AllGather set inter server "\
                "nhr algo failed", HCCL_ERROR_CODE(ret), tag.c_str()), ret);
    }
    /* 屏蔽pytorch子图+静态图场景 */
    if (UseInterServerPipelineAlgo(algType_) &&
        ((GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) || !isMeshTopo)) {
        HcclResult ret = SetInterServerHDAlgo(algType_);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[AllGatherOperator][AllGather]errNo[0x%016llx] tag[%s], AllGather set inter server "\
                "halving-doubling algo failed", HCCL_ERROR_CODE(ret), tag.c_str()), ret);
        HCCL_WARNING("[AllGatherOperator][AllGather] pipeline not support subgraphmode yet, reset algo to HD");
    }

    meshSinglePlane_ = NeedCreateSingleMeshPlane(true);

    CHK_RET(hcclImpl_->PrepareCommRes(tag, inputMem, outputMem, algType_, stream, INVALID_VALUE_RANKID, false, false,
        false, meshSinglePlane_));

    HCCL_PROFILER_ADD_STREAM(stream.ptr(), tag, 0, algType_);

    // 添加从流profiling, 用于维护planID
    CHK_RET(hcclImpl_->AddSubStreamToProfiling(tag, HcclCMDType::HCCL_CMD_ALLGATHER));

    /*  ------------执行算法-------------- */
    HcclUs startut = TIME_NOW();

    HcclResult ret = RunAllGather(tag, inputMem, outputMem, inputCount, dataType,
        HcclReduceOp::HCCL_REDUCE_RESERVED, stream, opInfo);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[AllGatherOperator][AllGather]errNo[0x%016llx] tag[%s],all gather run failed",
            HCCL_ERROR_CODE(ret), tag.c_str()), ret);

    HCCL_INFO("tag[%s],all gather run success,take time [%lld]us.", tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult AllGatherOperator::GetAllGatherOutPlaceSplitLoop(void* commOutputPtr, bool isMeshTopo, const u32 unitSize,
    const u64 inputCount, u64 &maxCountPerLoop)
{
    u64 commOutputSize = 0;
    CHK_RET(cclBufferManager_.GetOutCCLbuffer(commOutputPtr, commOutputSize));
    if (deviceType_ == DevType::DEV_TYPE_910B && isMeshTopo &&
        GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
        (UseInterServerPipelineAlgo(algType_) || isSingleMeshAggregation_)) {
        // 910B场景+单算子模式，使用SDMA消减
        // 中转内存单次最多能够接受的input count
        // SDMA消减情况注意16k地址对齐
        maxCountPerLoop = (commOutputSize - HCCL_MIN_SLICE_ALIGN_910B) / unitSize;
        HCCL_DEBUG("[AllGatherOperator][GetAllGatherOutPlaceSplitLoop]userRank_[%llu] maxCountPerLoop is [%llu]",
            userRank_, maxCountPerLoop);
    } else {
        maxCountPerLoop = commOutputSize / (userRankSize_ * unitSize);
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherOperator::SetAllGatherOutPlaceHugedata(bool isMeshTopo, const u64 curSize, bool &hugeData)
{
    if (deviceType_ == DevType::DEV_TYPE_910B && isMeshTopo &&
        GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
        (UseInterServerPipelineAlgo(algType_) || isSingleMeshAggregation_)) {
        // 910B场景+单算子模式，使用SDMA消减
        if (isSingleMeshAggregation_) {
            hugeData = curSize > SDMA_SEND_MAX_SIZE;
        } else {
            hugeData = curSize > RDMA_SEND_MAX_SIZE || curSize > SDMA_SEND_MAX_SIZE;
        }
    } else {
        hugeData = curSize * userRankSize_ / HCCL_INTERNODE_MAX_DATA_RATE > RDMA_SEND_MAX_SIZE ||
            curSize > SDMA_SEND_MAX_SIZE;
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherOperator::AllGatherOutPlaceFor310PForOneRankSize(const std::string &tag, void *inputPtr, void *outputPtr, u64 inputCount,
    HcclDataType dataType, Stream stream)
{
    DeviceMem srcMem(inputPtr, inputCount*SIZE_TABLE[dataType]);
    DeviceMem dstMem(outputPtr, inputCount*SIZE_TABLE[dataType]);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, stream));
    return HCCL_SUCCESS;
}

HcclResult AllGatherOperator::AllGatherOutPlaceFor310P(const std::string &tag, void *inputPtr,
    void *outputPtr, u64 inputCount, HcclDataType dataType, Stream stream)
{
    void *commInputPtr = nullptr;
    void *commOutputPtr = nullptr;
    u64 commInputSize, commOutputSize;

    CHK_RET(cclBufferManager_.GetInCCLbuffer(commInputPtr, commInputSize));
    CHK_RET(cclBufferManager_.GetInCCLbuffer(commOutputPtr, commOutputSize));
    if (userRankSize_ == 1 ) {
        CHK_RET(AllGatherOutPlaceFor310PForOneRankSize(tag,inputPtr,outputPtr,inputCount,dataType,stream));
        return HCCL_SUCCESS;
    }

    u32 unitSize = SIZE_TABLE[dataType];
    u8 *curInputPtr = static_cast<u8 *>(inputPtr);
    u8 *curOutputPtr = static_cast<u8 *>(outputPtr);
    CHK_PTR_NULL(curInputPtr);
    CHK_PTR_NULL(curOutputPtr);

    u64 maxCountPerLoop = commOutputSize / (userRankSize_ * unitSize); // 中转内存单次最多能够接受的input count
    u64 curCount = 0;
    DeviceMem dstMem;
    DeviceMem srcMem;
    for (u64 countLeft = inputCount, inputOffset = 0, outputOffset = 0; countLeft > 0; countLeft -= curCount) {
        curInputPtr += inputOffset;
        curOutputPtr += outputOffset;
        HCCL_INFO("-OP_BASE-AllGatherLoop:inputOffset[%llu], outputOffset[%llu]", inputOffset, outputOffset);
        // 判断剩余数据量对应的output size是否大于中转output size
        curCount = ((countLeft * unitSize * userRankSize_) > commOutputSize) ? maxCountPerLoop : countLeft;
        u64 curSize = curCount * unitSize; // 单位：字节

        srcMem = DeviceMem::create(curInputPtr, curSize);
        dstMem = DeviceMem::create(commInputPtr, curSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, stream));
        HcclResult ret =
            AllGather(tag, commInputPtr, commOutputPtr, curCount, dataType, stream);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Loop][AllGather]errNo[0x%016llx] op_base hcclComm all_gather error, tag[%s], input_ptr[%p],"
                " output_ptr[%p], count[%llu], data_type[%s]",
                HCCL_ERROR_CODE(ret),
                tag.c_str(),
                commInputPtr,
                commOutputPtr,
                curCount,
                GetDataTypeEnumStr(dataType).c_str()),
            ret);
        for (u32 i = 0; i < userRankSize_; i++) {
            // 拷贝中转output上每个slice的数据到output内存，目的端中每个slice的size固定为output的size
            dstMem = DeviceMem::create(curOutputPtr + inputCount * unitSize * i, curSize);
            srcMem = DeviceMem::create(static_cast<char *>(commOutputPtr) + curSize * i, curSize);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, stream));
        }
        inputOffset = curSize;
        outputOffset = curSize;
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherOperator::AllGatherOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 inputCount,
    HcclDataType dataType, Stream stream, const std::unique_ptr<HcclOpBaseAtraceInfo> &opBaseAtraceInfo)
{
    if (Is310P3Common()) {
        CHK_RET(AllGatherOutPlaceFor310P(tag, inputPtr, outputPtr, inputCount, dataType, stream));
        return HCCL_SUCCESS;
    }

    auto rtStream = stream.ptr();

    u32 unitSize = SIZE_TABLE[dataType];
    u8 *curInputPtr = static_cast<u8 *>(inputPtr);
    u8 *curOutputPtr = static_cast<u8 *>(outputPtr);
    CHK_PTR_NULL(curInputPtr);
    CHK_PTR_NULL(curOutputPtr);

    auto inCCLbuffer = cclBufferManager_.GetInCCLbuffer();
    auto outCCLbuffer = cclBufferManager_.GetOutCCLbuffer();

    // 判断是否使用mesh算法，避免mesh物理链路下使用非mesh算法勿入SDMA消减流程
    // isSingleMeshAggregation_只是指示了物理链路为mesh，而SDMA消减只在mesh算法下使用
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_NP_MESH || topoType_ == TopoType::TOPO_TYPE_4P_MESH ||
            topoType_ == TopoType::TOPO_TYPE_2P_MESH || topoType_ == TopoType::TOPO_TYPE_1P_MESH;
    if (userRankSize_ == 1 ) {
        HCCL_PROFILER_ADD_TAG(tag, identifier_, GetWorkflowMode());
        HCCL_PROFILER_ADD_STREAM(rtStream, tag, 0, algType_);
        HCCL_PROFILER_ADD_OPDATA(tag, inputCount, inputPtr, outputPtr, dataType, INVALID_VALUE_RANKID, identifier_);
        HCCL_PROFILER_ADD_GROUPRANK(identifier_, userRankSize_, userRank_);
        auto originalAlgTypeLevel1 = static_cast<u32>(algType_) >> HCCL_LEVEL_ALGO_WIDTH;
        bool hugeData = (inputCount * unitSize) > SDMA_SEND_MAX_SIZE;
        if (inputPtr == outputPtr) {
            auto opMeta = HcclOpMetaInfo::GetOneForAllGather(originalAlgTypeLevel1, hugeData, CopyPattern::ZCOPY);
            CHK_RET(InitTask(dispatcher_, stream, opMeta.isEnableCache, opMeta.GetCacheKey()));
        } else {
            auto opMeta = HcclOpMetaInfo::GetOneForAllGather(originalAlgTypeLevel1, hugeData, CopyPattern::BCOPY);
            CHK_RET(InitTask(dispatcher_, stream, opMeta.isEnableCache, opMeta.GetCacheKey()));
            DeviceMem srcMem(inputPtr, inputCount*SIZE_TABLE[dataType]);
            DeviceMem dstMem(outputPtr, inputCount*SIZE_TABLE[dataType]);
            HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, stream); 
        }
        CHK_RET(LaunchTask(dispatcher_, stream));
        HCCL_PROFILER_DEL_STREAM(rtStream);
        HCCL_PROFILER_DEL_TAG(tag);
        HCCL_PROFILER_DEL_OPDATA(tag);
        HCCL_PROFILER_DEL_GROUPRANK(tag);

        return HCCL_SUCCESS;
    }
    u64 maxCountPerLoop = 0;
    u64 cclBufferSize = outCCLbuffer.size() / userRankSize_;
    u64 curCount = 0;
    std::string newTag = tag;
    if (!isSingleMeshAggregation_) {
        u64 inputSize = inputCount * unitSize; // 单位：字节
        std::string algTypeLevel1Tag;
        CHK_RET(AutoSelectAlgTypeLevel1(HcclCMDType::HCCL_CMD_ALLGATHER, inputSize, cclBufferSize, algTypeLevel1Tag));
        if (opBaseAtraceInfo != nullptr) {
            CHK_RET(opBaseAtraceInfo->SavealgtypeTraceInfo(algTypeLevel1Tag, tag));
        }
        newTag = GenerateNewTagByAlgTypeLevel1(tag, algTypeLevel1Tag);
    }

    CHK_RET(GetAllGatherOutPlaceSplitLoop(outCCLbuffer.ptr(), isMeshTopo, unitSize, inputCount, maxCountPerLoop));
    HCCL_DEBUG("[AllGatherOperator][AllGatherOutPlace] maxCountPerLoop = %llu", maxCountPerLoop);

    HCCL_PROFILER_ADD_TAG(newTag, identifier_, GetWorkflowMode());
    HCCL_PROFILER_ADD_STREAM(rtStream, newTag, 0, algType_);
    HCCL_PROFILER_ADD_OPDATA(newTag, inputCount, inputPtr, outputPtr, dataType, INVALID_VALUE_RANKID, identifier_);
    HCCL_PROFILER_ADD_GROUPRANK(identifier_, userRankSize_, userRank_);

    if (!isSingleMeshAggregation_) {
        HcomCollOpInfo opInfo = {"", inputPtr, outputPtr, inputCount, dataType, 0, HCCL_REDUCE_RESERVED};
        CHK_RET(hcclImpl_->CreateOpBasedResources(HcclCMDType::HCCL_CMD_ALLGATHER, newTag, opInfo));
    }

    DeviceMem dstMem;
    DeviceMem srcMem;
    for (u64 countLeft = inputCount, inputOffset = 0, outputOffset = 0; countLeft > 0; countLeft -= curCount) {
        curInputPtr += inputOffset;
        curOutputPtr += outputOffset;
        HCCL_INFO("-OP_BASE-AllGatherLoop:inputOffset[%llu], outputOffset[%llu].", inputOffset, outputOffset);
        // 判断剩余数据量对应的output size是否大于中转output size
        curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        u64 curSize = curCount * unitSize; // 单位：字节
        auto autoSelectedAlgTypeLevel1 = static_cast<u32>(algType_) >> HCCL_LEVEL_ALGO_WIDTH;

        bool hugeData = false;
        u32 dataSplit = 0;
        u64 dataValue = curCount * unitSize * userRankSize_;
        if ((serverNum_ > 1) && ((dataValue / serverNum_) <= HCCL_SDMA_RDMA_SPLIT_SIZE)) {
            dataSplit = 1;
        } else if (dataValue <= HCCL_SDMA_RDMA_SPLIT_SIZE) {
            dataSplit = HCCL_SPLIT_FLAG;
        }
        CHK_RET(SetAllGatherOutPlaceHugedata(isMeshTopo, curSize, hugeData));
        auto meta = HcclOpMetaInfo::GetOneForAllGather(autoSelectedAlgTypeLevel1, hugeData);
        meta.dataSplit = dataSplit;
        CHK_RET(InitTask(dispatcher_, stream, meta.isEnableCache, meta.GetCacheKey()));
        HCCL_DEBUG("AllGatherOutPlace: sendBuf[%p], recvBuf[%p], sendCount[%llu], dataType[%s], tag[%s]",
            curInputPtr, curOutputPtr, curCount, GetDataTypeEnumStr(dataType).c_str(), newTag.c_str());

        bool isPipeLine = (deviceType_ == DevType::DEV_TYPE_910B && isMeshTopo &&
            GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
            (UseInterServerPipelineAlgo(algType_) || isSingleMeshAggregation_));
        // 当前allgather 的DMA削减只支持sever内
        bool isDMAreduceOn91073 = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE
                                  && (deviceType_ == DevType::DEV_TYPE_910_73) && !isMeshTopo);
        bool isUseDMA = !GetExternalInputEnableRdmaSdmaConcurrent();
        if (isUseDMA && (isPipeLine || isDMAreduceOn91073)) {
            CHK_RET(RankConsistent::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_ALLGATHER, newTag, curCount,
                dataType, inCCLbuffer.size(), outCCLbuffer.size(), HCCL_WORLD_GROUP));
            HcomCollOpInfo singleServerOpInfo;
            singleServerOpInfo.inputAddr = curInputPtr;
            singleServerOpInfo.outputAddr = curOutputPtr;
            singleServerOpInfo.count = inputCount;
            singleServerOpInfo.dataType = dataType;

            HcclResult ret =
                AllGather(newTag, inCCLbuffer.ptr(), outCCLbuffer.ptr(), curCount, dataType, stream,
                    &singleServerOpInfo);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Loop][AllGather]errNo[0x%016llx] op_base hcclComm all_gather error with SDMA direct,"
                    "tag[%s], input_ptr[%p], output_ptr[%p], count[%llu], data_type[%s]", HCCL_ERROR_CODE(ret),
                    newTag.c_str(), inCCLbuffer.ptr(), outCCLbuffer.ptr(), curCount,
                    GetDataTypeEnumStr(dataType).c_str()), ret);
            CHK_RET(RankConsistent::GetInstance().DelOpPara(newTag));
        } else {
            srcMem = DeviceMem::create(curInputPtr, curSize);
            dstMem = inCCLbuffer.range(0, curSize);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, stream));

            CHK_RET(RankConsistent::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_ALLGATHER, newTag, curCount,
                dataType, inCCLbuffer.size(), outCCLbuffer.size(), HCCL_WORLD_GROUP));

            HcclResult ret = AllGather(newTag, inCCLbuffer.ptr(), outCCLbuffer.ptr(), curCount, dataType, stream);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR(
                "[Loop][AllGather]errNo[0x%016llx] op_base hcclComm all_gather error, tag[%s],"
                "input_ptr[%p], output_ptr[%p], count[%llu], data_type[%s]",
                HCCL_ERROR_CODE(ret),
                newTag.c_str(),
                inCCLbuffer.ptr(),
                outCCLbuffer.ptr(),
                curCount,
                GetDataTypeEnumStr(dataType).c_str()),
                ret);

            CHK_RET(RankConsistent::GetInstance().DelOpPara(newTag));

            for (u32 i = 0; i < userRankSize_; i++) {
                // 拷贝中转output上每个slice的数据到output内存，目的端中每个slice的size固定为output的size
                dstMem = DeviceMem::create(curOutputPtr + inputCount * unitSize * i, curSize);
                srcMem = outCCLbuffer.range(curSize * i, curSize);
                CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, stream));
            }
        }

        CHK_PRT_RET((curCount == 0), HCCL_ERROR("[Loop][AllGather]In OP_BASE curCount is zero"), HCCL_E_PARA);
        inputOffset = curSize;
        outputOffset = curSize;

        CHK_RET(LaunchTask(dispatcher_, stream));
    }
    HCCL_PROFILER_DEL_STREAM(rtStream);
    HCCL_PROFILER_DEL_TAG(newTag);
    HCCL_PROFILER_DEL_OPDATA(newTag);
    HCCL_PROFILER_DEL_GROUPRANK(newTag);
    return HCCL_SUCCESS;
}

HcclResult AllGatherOperator::AllGatherCommFor310P(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
    u64 count, HcclDataType dataType, HcclReduceOp op, Stream &stream)
{
    CommInfo *currComm;
    hcclImpl_->GetCommInfo(currComm, tag);
    std::unique_ptr<CommBase> &commCombined = currComm->commIntraServer;

    std::unique_ptr<ExecutorBase> executor;
    executor.reset(new (std::nothrow) AllGatherRing(dispatcher_));
    CHK_SMART_PTR_NULL(executor);

    CHK_RET(executor->Prepare(inputMem, outputMem, outputMem, count, dataType, stream, op));

    u32 rankSize = commCombined->RankSize();
    CHK_RET(executor->RegisterProfiler(
        (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + commCombined->Rank(),
        PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, stream));

    CHK_RET(commCombined->RunExecutor(executor));
    HCCL_INFO("allgather for 310P run success");

    return HCCL_SUCCESS;
}

HcclResult AllGatherOperator::RunAllGather(const std::string& tag, DeviceMem &inputMem, DeviceMem &outputMem, u64 count,
    HcclDataType dataType, HcclReduceOp op, Stream &stream, HcomCollOpInfo *opInfo)
{
    HcclResult ret;

    if (Is310P3Common()) {
        ret = AllGatherCommFor310P(tag, inputMem, outputMem, count, dataType, op, stream);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Run][AllGather]tag[%s], all_gather_run failed, return[%d]", tag.c_str(), ret), ret);
        return HCCL_SUCCESS;
    }

    switch (topoType_) {
        case TopoType::TOPO_TYPE_NP_MESH:
        case TopoType::TOPO_TYPE_4P_MESH:
        case TopoType::TOPO_TYPE_2P_MESH:
        case TopoType::TOPO_TYPE_1P_MESH:
            if (deviceType_ == DevType::DEV_TYPE_910B && GetWorkflowMode() ==
                HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && opInfo != nullptr && (isSingleMeshAggregation_ ||
                UseInterServerPipelineAlgo(algType_))) {
                if (isSingleMeshAggregation_) {
                    ret = AllGatherMeshOpbaseExecutor(tag, inputMem, outputMem, count, dataType, op, stream, opInfo);
                    break;
                } else { // pipeline allgather
                    ret = AllGatherMeshOpbasePipelineExecutor(tag, inputMem, outputMem, count,
                        dataType, op, stream, opInfo);
                    break;
                }
            }  else {
                ret = AllGatherMeshExecutor(tag, inputMem, outputMem, count, dataType, op, stream);
                break;
            }
        case TopoType::TOPO_TYPE_8P_RING:
        case TopoType::TOPO_TYPE_NP_SINGLE_RING:
        case TopoType::TOPO_TYPE_NP_DOUBLE_RING: // 只存在于910_73场景下
            if (deviceType_ == DevType::DEV_TYPE_910_73) {
                if (GetExternalInputEnableRdmaSdmaConcurrent()) {
                    ret = AllGatherDoubleRingConcurrentExecutor(tag, inputMem, outputMem, count,
                                                            dataType, op, stream, opInfo);
                } else {
                    ret = AllGatherDoubleRingExecutor(tag, inputMem, outputMem, count, dataType, op, stream, opInfo);
                }
                break;
            } else {
                ret = AllGatherRingExecutor(tag, inputMem, outputMem, count, dataType, op, stream, opInfo);
                break;
            }
        default:
            ret = AllGatherComm(tag, inputMem, outputMem, count, dataType, op, stream);
            break;
    }
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[AllGatherExector][RunAllGather]tag[%s], all_gather failed, return[%d]", tag.c_str(), ret), ret);

    return ret;
}

HcclResult AllGatherOperator::AllGatherComm(const std::string &tag, DeviceMem &inputMem,
    DeviceMem &outputMem, u64 count, HcclDataType dataType, HcclReduceOp op, Stream &stream)
{
    (void)op;
    CommInfo *currComm;
    hcclImpl_->GetCommInfo(currComm, tag);
    CHK_PRT_RET(currComm->commInner.size() == 0,
        HCCL_ERROR("[AllGatherExector][AllGatherComm]errNo[0x%016llx] tag[%s], all gather op inner comm is empty",
            HCCL_ERROR_CODE(HCCL_E_INTERNAL), tag.c_str()), HCCL_E_INTERNAL);

    std::unique_ptr<CommBase> &commCombine = currComm->commInner[COMM_INDEX_0];
    CHK_SMART_PTR_NULL(commCombine);

    // 构造ring algorithm对应的all_gather实例
    std::unique_ptr<ExecutorBase> executor;
    if (UseInterServerNHRAlgo(algType_)) {
        executor.reset(new (std::nothrow) AllGatherNHR(dispatcher_));
        HCCL_INFO("algather comm: using nhr algo inter-server.");
    } else if (UseInterServerNHRV1Algo(algType_)) {
        executor.reset(new (std::nothrow) AllGatherNHRV1(dispatcher_));
        HCCL_INFO("algather comm: using nhr_v1 algo inter-server.");
    } else if (UseInterServerNBAlgo(algType_)) {
        executor.reset(new (std::nothrow) AllGatherNB(dispatcher_));
        HCCL_INFO("algather comm: using nonuniform-bruck algo inter-server.");
    } else {
        executor.reset(new (std::nothrow) AllGatherRing(dispatcher_));
        HCCL_INFO("algather comm: ring algo inter-server.");
    }
    CHK_SMART_PTR_NULL(executor);

    HcclResult ret = RunExecutor(commCombine, executor, inputMem, outputMem, count, dataType,
        HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, stream);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[AllGatherExector][AllGatherComm]tag[%s],allgather run failed,return[%d]", tag.c_str(), ret), ret);

    return HCCL_SUCCESS;
}
HcclResult AllGatherOperator::AllGatherMeshOpbaseExecutor(const std::string &tag, DeviceMem &inputMem,
    DeviceMem &outputMem, u64 count, HcclDataType dataType, HcclReduceOp op, Stream &stream, HcomCollOpInfo *opInfo)
{
    u8 *curInputPtr = static_cast<u8 *>(opInfo->inputAddr);
    u8 *curOutputPtr = static_cast<u8 *>(opInfo->outputAddr);
    CHK_PTR_NULL(curInputPtr);
    CHK_PTR_NULL(curOutputPtr);
    CommInfo *currComm;
    hcclImpl_->GetCommInfo(currComm, tag);
    CHK_PRT_RET(currComm->commOuter.empty(),
        HCCL_ERROR("[AllGatherOperator][AllGatherMeshOpbaseExecutor]errNo[0x%016llx] comm outer is empty",
        HCCL_ERROR_CODE(HCCL_E_PARA)),
        HCCL_E_PARA);

    CHK_SMART_PTR_NULL(currComm->commOuter[COMM_INDEX_0]);

    u64 inputMemSize = inputMem.size();
    u64 baseOffset = 0;
    std::vector<Slice> dataSegsSlice;                 // 数据分成ranksize份，每份的起始偏移和大小
    bool bRet = currComm->commOuter.size() == 0;
    CHK_PRT_RET(bRet, HCCL_ERROR("[AllGatherExector][AllGatherMeshOpbaseExecutor]tag[%s],comm outer is empty",
        tag.c_str()), HCCL_E_INTERNAL);
    std::unique_ptr<CommBase> &commOuter = currComm->commOuter[COMM_INDEX_0];
    CHK_SMART_PTR_NULL(commOuter);
    CHK_RET(hcclImpl_->ActiveRingStreams(tag, stream));
    innerStreamInfo_t *streamInfo = hcclImpl_->GetStreamInfo(tag);
    CHK_PRT_RET(streamInfo == nullptr,
        HCCL_ERROR("[GetStreamInfo]errNo[0x%016llx] tag[%s] can't find in stream info",
            HCCL_ERROR_CODE(HCCL_E_NOT_FOUND), tag.c_str()), HCCL_E_PARA);

    //  抽取当前用于多环all gather 的output内存数据
    DeviceMem currentOutputMem = outputMem.range(baseOffset, inputMemSize); // 减少dma out大小
    CHK_SMART_PTR_NULL(currentOutputMem);

    std::unique_ptr<ExecutorBase> outerExecutor;
    outerExecutor.reset(
        new (std::nothrow) AllgatherMeshDirect(dispatcher_, streamInfo->ringStreams,
        streamInfo->ringSignal, streamInfo->ringSignalAux, commOuter->Rank(), commOuter->RankSize(),
        commOuter->UserRank(), opInfo));
    CHK_SMART_PTR_NULL(outerExecutor);
    CHK_RET(outerExecutor->Prepare(currentOutputMem, currentOutputMem, inputMem, count, dataType,
        stream, op, OUTER_BRIDGE_RANK_ID, dataSegsSlice, baseOffset));

    u32 rankSize = commOuter->RankSize();
    CHK_RET(outerExecutor->RegisterProfiler((rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + commOuter->Rank(),
        PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, stream));

    CHK_RET(commOuter->RunExecutor(outerExecutor));

    HCCL_INFO("all gather mesh outer run success");
    return HCCL_SUCCESS;
}

HcclResult AllGatherOperator::AllGatherMeshOpbasePipelineExecutor(const std::string &tag,
    DeviceMem &inputMem, DeviceMem &outputMem,
    u64 count, HcclDataType dataType, HcclReduceOp op, Stream &stream, HcomCollOpInfo *opInfo)
{
    HCCL_INFO("[AllGatherExector][AllGatherMeshOpbasePipelineExecutor] AllGatherMeshOpbasePipelineExecutor begins.");
    // step 1 先获取 comm inner \ comm outer 的value
    CommInfo *currComm;
    hcclImpl_->GetCommInfo(currComm, tag);

    CHK_PRT_RET(currComm->commOuter.empty(),
        HCCL_ERROR("[AllGatherExector][AllGatherMeshOpbasePipelineExecutor]errNo[0x%016llx] comm outer is empty",
        HCCL_ERROR_CODE(HCCL_E_PARA)),
        HCCL_E_PARA);
    u32 commIndex = 0;
    u32 serverIndex = 0;

    CHK_SMART_PTR_NULL(currComm->commOuter[COMM_INDEX_0]);
    commIndex = currComm->commOuter[COMM_INDEX_0]->Rank();
    bool bRet = commIndex >= currComm->commInner.size();
    CHK_PRT_RET(bRet,
        HCCL_ERROR("[AllGatherExector][AllGatherMeshOpbasePipelineExecutor]errNo[0x%016llx] commIndex[%u] >= (tag:[%s])"
        "comm_inner.size[%llu]",
        HCCL_ERROR_CODE(HCCL_E_INTERNAL), commIndex, tag.c_str(), currComm->commInner.size()),
        HCCL_E_INTERNAL);
    CHK_SMART_PTR_NULL(currComm->commInner[commIndex]);
    HcclResult ret = currComm->commInner[commIndex]->GetRankByUserRank(userRank_, serverIndex);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[AllGatherExector][AllGatherMeshOpbasePipelineExecutor]Get Rank[%u] by User Rank[%u]"
        " form CommInner[%u] Failed!", serverIndex, userRank_, devicePhyId_), ret);
    bRet = currComm->commOuter.size() == 0;
    CHK_PRT_RET(bRet, HCCL_ERROR("[AllGatherExector][AllGatherMeshOpbasePipelineExecutor]tag[%s],comm outer is empty",
        tag.c_str()), HCCL_E_INTERNAL);
    std::unique_ptr<CommBase> &commOuter = currComm->commOuter[COMM_INDEX_0];
    CHK_SMART_PTR_NULL(commOuter);

    innerStreamInfo_t *streamInfo = hcclImpl_->GetStreamInfo(tag);
    CHK_PRT_RET(streamInfo == nullptr,
        HCCL_ERROR("[GetStreamInfo]errNo[0x%016llx] tag[%s] can't find in stream info",
            HCCL_ERROR_CODE(HCCL_E_NOT_FOUND), tag.c_str()), HCCL_E_PARA);

    bRet = commIndex >= currComm->commInner.size();
    CHK_PRT_RET(bRet,
        HCCL_ERROR("[AllGatherExector][AllGatherMeshOpbasePipelineExecutor]errNo[0x%016llx] commIndex[%u] >= (tag:[%s])"
        "comm_inner.size[%llu]",
        HCCL_ERROR_CODE(HCCL_E_INTERNAL), commIndex, tag.c_str(), currComm->commInner.size()),
        HCCL_E_INTERNAL);
    std::unique_ptr<CommBase> &commInner = currComm->commInner[commIndex];

    std::unique_ptr<AllGatherPipeline> executor;
    executor.reset(new (std::nothrow) AllGatherPipeline(dispatcher_));
    CHK_SMART_PTR_NULL(executor);
    CHK_RET(executor->Prepare(opInfo, userRank_, count, inputMem, outputMem, commOuter, commInner, stream,
        streamInfo->ringStreams, streamInfo->ringSignal, streamInfo->ringSignalAux));
    CHK_RET(executor->RunAsync());
    return HCCL_SUCCESS;
}

HcclResult AllGatherOperator::AllGatherMeshExecutor(const std::string &tag, DeviceMem &inputMem,
    DeviceMem &outputMem, u64 count, HcclDataType dataType, HcclReduceOp op, Stream &stream)
{
    u32 perDataSize = SIZE_TABLE[dataType];

    CommInfo *currComm;
    hcclImpl_->GetCommInfo(currComm, tag);

    CHK_PRT_RET(currComm->commOuter.empty(),
        HCCL_ERROR("[AllGatherOperator][AllGatherMeshExecutor]errNo[0x%016llx] comm outer is empty",
        HCCL_ERROR_CODE(HCCL_E_PARA)),
        HCCL_E_PARA);
    u32 outerRankSize = 0;
    u32 commIndex = 0;
    u32 serverIndex = 0;
    {
        CHK_SMART_PTR_NULL(currComm->commOuter[COMM_INDEX_0]);
        outerRankSize = currComm->commOuter[COMM_INDEX_0]->RankSize();
        commIndex = currComm->commOuter[COMM_INDEX_0]->Rank();
        bool bRet = commIndex >= currComm->commInner.size();
        CHK_PRT_RET(bRet,
            HCCL_ERROR("[AllGatherOperator][AllGatherMeshExecutor]errNo[0x%016llx] commIndex[%u] >= (tag:[%s])"
            "comm_inner.size[%llu]",
            HCCL_ERROR_CODE(HCCL_E_INTERNAL), commIndex, tag.c_str(), currComm->commInner.size()),
            HCCL_E_INTERNAL);
        CHK_SMART_PTR_NULL(currComm->commInner[commIndex]);
        HcclResult ret = currComm->commInner[commIndex]->GetRankByUserRank(userRank_, serverIndex);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[AllGatherOperator][AllGatherMeshExecutor]Get Rank[%u] by User Rank[%u] form CommInner[%u] "
            "Failed!",
            serverIndex, userRank_, devicePhyId_),
            ret);
    }

    u64 inputMemSize = inputMem.size();
    u64 baseOffset = serverIndex * inputMemSize * outerRankSize;
    u64 outerOffset = commIndex * inputMemSize;
    DeviceMem dstMem = outputMem.range(baseOffset + outerOffset, inputMemSize);
    CHK_SMART_PTR_NULL(dstMem);
    //  第一步，将数据从input内存拷贝到output内存的对应位置
    HcclResult ret = HcclD2DMemcpyAsync(dispatcher_, dstMem, inputMem, stream);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[AllGatherOperator][AllGatherMeshExecutor]all gather 4PmeshHD memcpy Failed, Offset[%llu], "
        "Size[%llu]",
        baseOffset + outerOffset, inputMemSize),
        ret);

    // 第二步，各个AI Server 内 multi stream mesh all gather
    std::vector<Slice> dataSegsSlice;                 // 数据分成ranksize份，每份的起始偏移和大小
    std::vector<std::vector<Slice>> multiStreamSlice; // 每个stream使用的数据基于用户buffer的偏移
    bool bRet = currComm->commOuter.size() == 0;
    CHK_PRT_RET(bRet, HCCL_ERROR("[AllGatherOperator][AllGatherMeshExecutor]tag[%s],comm outer is empty", tag.c_str()),
        HCCL_E_INTERNAL);
    std::unique_ptr<CommBase> &commOuter = currComm->commOuter[COMM_INDEX_0];
    CHK_SMART_PTR_NULL(commOuter);
    u32 sliceNum = commOuter->RankSize();
    Slice sliceTemp;

    for (u32 i = 0; i < sliceNum; i++) { // 根据数据量算每个环上数据的偏移和大小
        sliceTemp.size = inputMemSize;
        sliceTemp.offset = inputMemSize * i;
        dataSegsSlice.push_back(sliceTemp);
    }
    // mesh算法stream数量为server内rank数减1
    CHK_RET(ExecutorBase::PrepareSliceMeshStreams(dataSegsSlice, sliceNum - 1, multiStreamSlice));

    CHK_RET(hcclImpl_->ActiveRingStreams(tag, stream));

    innerStreamInfo_t *streamInfo = hcclImpl_->GetStreamInfo(tag);
    CHK_PRT_RET(streamInfo == nullptr,
        HCCL_ERROR("[GetStreamInfo]errNo[0x%016llx] tag[%s] can't find in stream info",
            HCCL_ERROR_CODE(HCCL_E_NOT_FOUND), tag.c_str()), HCCL_E_PARA);

    //  抽取当前用于多环all gather 的output内存数据
    DeviceMem currentOutputMem = outputMem.range(baseOffset, inputMemSize * outerRankSize);
    CHK_SMART_PTR_NULL(currentOutputMem);
    std::unique_ptr<ExecutorBase> outerExecutor;
    if (deviceType_ == DevType::DEV_TYPE_910B) {
        outerExecutor.reset(
            new (std::nothrow) AllGatherMeshAtomic(dispatcher_, streamInfo->ringStreams,
            streamInfo->ringSignal, streamInfo->ringSignalAux, commOuter->Rank(), commOuter->RankSize(),
            commOuter->UserRank()));
    } else {
        outerExecutor.reset(
            new (std::nothrow) AllGatherMesh(dispatcher_, streamInfo->ringStreams, streamInfo->ringSignal,
                streamInfo->ringSignalAux, commOuter->Rank(), commOuter->RankSize(), commOuter->UserRank()));
    }
    CHK_SMART_PTR_NULL(outerExecutor);
    CHK_RET(outerExecutor->Prepare(currentOutputMem, currentOutputMem, inputMem, count * outerRankSize, dataType,
        stream, op, OUTER_BRIDGE_RANK_ID, dataSegsSlice, baseOffset));
    u32 rankSize = commOuter->RankSize();
    CHK_RET(outerExecutor->RegisterProfiler((rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + commOuter->Rank(),
        PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, stream));
    CHK_RET(commOuter->RunExecutor(outerExecutor));

    HCCL_INFO("all gather mesh HD outer run success");

    //  第三步， AI server 间 recursive halving doubling all gather
    u64 hdSize = inputMemSize * outerRankSize;
    u64 hdCount = hdSize / perDataSize;

    {
        std::unique_ptr<ExecutorBase> innerExecutor;
        if (UseInterServerRingAlgo(algType_) || (isDiffDeviceModule_ && serverNum_ == 1)) { // 1-单server-SDMA
            innerExecutor.reset(new (std::nothrow) AllGatherRing(dispatcher_));
            HCCL_INFO("allgather mesh: using ring algo inter-server.");
        } else if (UseInterServerNHRAlgo(algType_)) {
            innerExecutor.reset(new (std::nothrow) AllGatherNHR(dispatcher_));
            HCCL_INFO("allgather mesh: using nhr algo inter-server.");
        } else if (UseInterServerNHRV1Algo(algType_)) {
            innerExecutor.reset(new (std::nothrow) AllGatherNHRV1(dispatcher_));
            HCCL_INFO("allgather mesh: using nhr_v1 algo inter-server.");
        } else if (UseInterServerNBAlgo(algType_)) {
            innerExecutor.reset(new (std::nothrow) AllGatherNB(dispatcher_));
            HCCL_INFO("allgather mesh: using nonuniform-bruck algo inter-server.");
        } else {
            innerExecutor.reset(new (std::nothrow) AllGatherRecursiveHalvingDoubling(dispatcher_));
            HCCL_INFO("allgather mesh: using halving-doubling algo inter-server.");
        }

        CHK_SMART_PTR_NULL(innerExecutor);
        bool bRet = commIndex >= currComm->commInner.size();
        CHK_PRT_RET(bRet,
            HCCL_ERROR("[AllGatherOperator][AllGatherMeshExecutor]errNo[0x%016llx] commIndex[%u] >= (tag:[%s])"
            "comm_inner.size[%llu]",
            HCCL_ERROR_CODE(HCCL_E_INTERNAL), commIndex, tag.c_str(), currComm->commInner.size()),
            HCCL_E_INTERNAL);
        std::unique_ptr<CommBase> &commInner = currComm->commInner[commIndex];
        //  此处虽然带入inputMem作为scratch mem, 但inputMem 不能被使用
        CHK_RET(innerExecutor->Prepare(outputMem, outputMem, inputMem, hdCount, dataType, stream,
            HcclReduceOp::HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, std::vector<Slice>(COMM_INDEX_0), 0));

        rankSize = commInner->RankSize();
        CHK_RET(innerExecutor->RegisterProfiler((rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + commInner->Rank(),
            PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, stream));

        CHK_RET(commInner->RunExecutor(innerExecutor));
    }
    HCCL_INFO("all gather mesh HD inner run success");

    return HCCL_SUCCESS;
}

HcclResult AllGatherOperator::AllGatherDoubleRingExecutor(const std::string &tag, DeviceMem &inputMem,
                                                          DeviceMem &outputMem, u64 count, HcclDataType dataType,
                                                          HcclReduceOp op, Stream &stream, const HcomCollOpInfo *opInfo)
{
    HCCL_INFO("[AllGatherOperator][AllGatherDoubleRingExecutor] The AllGatherDoubleRingExecutor starts.");
    (void)op;
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(dataType, perDataSize));
    CHK_PRT_RET(perDataSize == 0,
        HCCL_ERROR("[AllGatherOperator][AllGatherDoubleRingExecutor]errNo[0x%016llx] datatype[%s] is invalid",
            HCCL_ERROR_CODE(HCCL_E_PARA), GetDataTypeEnumStr(dataType).c_str()), HCCL_E_PARA);
    CommInfo *currComm;
    hcclImpl_->GetCommInfo(currComm, tag);

    CHK_PRT_RET(currComm->commOuter.empty(),
        HCCL_ERROR("[AllGatherOperator][AllGatherDoubleRingExecutor]comm outer is empty"), HCCL_E_PARA);

    u32 commIndex = currComm->commOuter[0]->Rank();
    bool bRet = commIndex >= currComm->commInner.size();
    CHK_PRT_RET(bRet,
        HCCL_ERROR("[AllGatherOperator][AllGatherDoubleRingExecutor]commIndex[%u] >= (tag[%s])comm size[%llu]",
                   commIndex, tag.c_str(), currComm->commInner.size()), HCCL_E_INTERNAL);

    //  第一步，将数据从input内存拷贝到output内存的对应位置
    HcclResult ret;
    CHK_PRT_RET(currComm->commOuter.empty(),
        HCCL_ERROR("[AllGatherOperator][AllGatherDoubleRingExecutor]errNo[0x%016llx] comm outer is empty",
            HCCL_ERROR_CODE(HCCL_E_PARA)), HCCL_E_PARA);
    CHK_SMART_PTR_NULL(currComm->commOuter[COMM_INDEX_0]);
    u32 outerRankSize = currComm->commOuter[COMM_INDEX_0]->RankSize();

    CHK_SMART_PTR_NULL(currComm->commInner[commIndex]);
    u32 level0ServerIndex = 0;
    ret = currComm->commOuter[COMM_INDEX_0]->GetRankByUserRank(userRank_, level0ServerIndex);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AllGatherOperator][AllGatherDoubleRingExecutor]Get Rank[%u] by User"\
        " Rank[%u] from CommOuter[%u] Failed!", level0ServerIndex, userRank_, commIndex), ret);

    u32 level1ServerIndex = 0;
    ret = currComm->commInner[commIndex]->GetRankByUserRank(userRank_, level1ServerIndex);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AllGatherOperator][AllGatherDoubleRingExecutor]Get Rank[%u] by User"\
        " Rank[%u] from CommInner[%u] Failed!", level1ServerIndex, userRank_, commIndex), ret);

    u64 inputMemSize = inputMem.size();
    u64 baseOffset = level1ServerIndex * inputMemSize * outerRankSize;
    u64 outerOffset = commIndex * inputMemSize;
    DeviceMem dstMem = outputMem.range(baseOffset + outerOffset, inputMemSize);
    CHK_SMART_PTR_NULL(dstMem);

    u32 level0RankSize = currComm->commOuter[COMM_INDEX_0]->RankSize();
    u32 level1RankSize = currComm->commInner[commIndex]->RankSize();
    // 图模式opinfo为空，需要将数据从ccl input拷贝到ccl output上
    if (opInfo == nullptr) {
        ret = HcclD2DMemcpyAsync(dispatcher_, dstMem, inputMem, stream);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[AllGatherOperator][AllGatherDoubleRingExecutor]all gather double "
                               "ring memcpy Failed, Offset[%llu], Size[%llu]",
                               baseOffset + outerOffset, inputMemSize), ret);
    } else {
        // 先做server间算法，带有消减拷贝场景数据需要从user input取，拷贝到ccl output上
        if (level1RankSize > 1) {
            DeviceMem srcMem = DeviceMem::create(static_cast<u8 *>(opInfo->inputAddr), inputMemSize);
            ret = HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, stream);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[AllGatherOperator][AllGatherDoubleRingExecutor]all gather double "
                    "ring user memcpy Failed, Offset[%llu], Size[%llu]",
                    baseOffset + outerOffset, inputMemSize), ret);
        }
    }
    if (devNumInLevel2_ > 1) {
        // 超节点间做allgather
        ret = AllGatherLevel2Executor(tag, inputMem, outputMem, count, dataType, op, stream, opInfo);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[AllGatherOperator][AllGatherDoubleRingExecutor]tag[%s], all_gather failed, return[%d]",
                tag.c_str(), ret), ret);
    } else {
        // 无超节点间场景
        if (level1RankSize > 1) {
            u32 commIndex = currComm->commOuter[0]->Rank();
            HCCL_INFO("commIdx:%u TagCommInfo[%s].commInner.size():%u", commIndex, tag.c_str(),
                currComm->commInner.size());

            CHK_PRT_RET(commIndex >= currComm->commInner.size(),
                HCCL_ERROR("[AllGatherOperator][AllGatherDoubleRingExecutor]"\
                    "commIndex[%u] >= (tag:[%s])commLevel2.size[%u]",
                    commIndex, tag.c_str(), currComm->commInner.size()), HCCL_E_INTERNAL);

            std::unique_ptr<ExecutorBase> level1AGExecutor;
            std::unique_ptr<CommBase> &commLevel1 = currComm->commInner[commIndex];
            if (UseInterServerRingAlgo(algType_)) {
                level1AGExecutor.reset(new (std::nothrow) AllGatherRing(dispatcher_));
                HCCL_INFO("allgather ring: using ring algo inter-server.");
            } else if (UseInterServerNBAlgo(algType_)) {
                level1AGExecutor.reset(new (std::nothrow) AllGatherNB(dispatcher_));
                HCCL_INFO("allgather ring: using nonuniform-bruck algo inter-server.");
            } else {
                level1AGExecutor.reset(new (std::nothrow) AllGatherNHR(dispatcher_));
                HCCL_INFO("allgather ring: using nonuniform-hierarchical-ring algo inter-server.");
            }

            // 计算slice, 不同超节点相同slice
            std::vector<Slice> level1DataSegsSlice;
            Slice sliceTemp;
            for (u32 i = 0; i < level1RankSize; i++) {
                sliceTemp.size = inputMemSize;
                sliceTemp.offset = (i * level0RankSize +  level0ServerIndex) * inputMemSize;
                level1DataSegsSlice.push_back(sliceTemp);
            }
            CHK_RET(level1AGExecutor->Prepare(outputMem, outputMem, inputMem, count, dataType, stream,
                HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, level1DataSegsSlice, 0));

            CHK_RET(level1AGExecutor->RegisterProfiler((
                level1RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + commLevel1->Rank(),
                PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, stream));

            CHK_RET(commLevel1->RunExecutor(level1AGExecutor));
            HCCL_INFO("allgather double ring [superpod] level1 allgather run success");
        }
        // 节点内做all gather double ring
        std::vector<Slice> dataSegsSlice;
        std::vector<std::vector<Slice>> multRingsSliceZero; // 数据基于该rank上环0的偏移
        CHK_RET(PrepareAllgatherSlice(level0RankSize, inputMemSize, dataSegsSlice));

        //  多环数据切分
        if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
            multRingsSliceZero = PrepareMultiRingSlice(dataSegsSlice, tag, false, nicList_);
        } else {
            multRingsSliceZero.push_back(dataSegsSlice);
        }
        std::vector<std::vector<Slice>> multRingsSlice;
        CHK_RET(CalculateLevel1AllgatherSlice(inputMemSize, level0RankSize, level1RankSize,
            multRingsSliceZero, multRingsSlice));

        std::vector<std::vector<Slice>> multRingsUserMemSlice;
        if (opInfo == nullptr) {
            multRingsUserMemSlice = multRingsSlice;
        } else {
            for (u32 ringIndex = 0; ringIndex < multRingsSlice.size(); ringIndex++) {
                std::vector<Slice> level1UserMemSlice;
                for (auto &cclSlice : multRingsSlice[ringIndex]) {
                    Slice tmpSlice;
                    tmpSlice.size = cclSlice.size;
                    tmpSlice.offset =
                        (cclSlice.offset / inputMemSize) * opInfo->count * perDataSize +
                        multRingsSliceZero[ringIndex][0].offset;
                    level1UserMemSlice.push_back(tmpSlice);
                    HCCL_DEBUG("rank[%u], ringIndex[%u], tmpSlice.offset=[%llu], size=[%llu]",
                        userRank_, ringIndex, tmpSlice.offset, tmpSlice.size);
                }
                multRingsUserMemSlice.push_back(level1UserMemSlice);
            }
        }
        CHK_RET(hcclImpl_->ActiveRingStreams(tag, stream));
        if (opInfo != nullptr && level1RankSize > 1) {
            // allgather输入放在CCL buffer上，通过设置nullptr指示要从CCL buffer获取输入
            HcomCollOpInfo opInfoByAllGatherDMAreduce = *opInfo;
            opInfoByAllGatherDMAreduce.inputAddr      = nullptr;
            CHK_RET(MultiRingAllGather(tag, inputMem, outputMem, count, dataType,
                multRingsSlice, stream, PROF_STAGE_2, 0, &opInfoByAllGatherDMAreduce, multRingsUserMemSlice));
        } else {
            CHK_RET(MultiRingAllGather(tag, inputMem, outputMem, count, dataType,
                multRingsSlice, stream, PROF_STAGE_2, 0, opInfo, multRingsUserMemSlice));
        }
    }
    HCCL_INFO("all gather double ring inner run success");
    return HCCL_SUCCESS;
}

HcclResult AllGatherOperator::AllGatherDoubleRingConcurrentExecutor(const std::string &tag, DeviceMem &inputMem,
    DeviceMem &outputMem, u64 count, HcclDataType dataType,
    HcclReduceOp op, Stream &stream, const HcomCollOpInfo *opInfo)
{
    HCCL_INFO("[AllGatherOperator][AllGatherDoubleRingConcurrentExecutor] "\
                                     " AllGatherDoubleRingConcurrentExecutor starts.");
    (void)op;
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(dataType, perDataSize));
    CHK_PRT_RET(perDataSize == 0,
        HCCL_ERROR("[AllGatherOperator][AllGatherDoubleRingConcurrentExecutor]errNo[0x%016llx] datatype[%d] is invalid",
            HCCL_ERROR_CODE(HCCL_E_PARA), dataType), HCCL_E_PARA);
    CommInfo *currComm;
    hcclImpl_->GetCommInfo(currComm, tag);

    CHK_PRT_RET(currComm->commOuter.empty(),
        HCCL_ERROR("[AllGatherOperator][AllGatherDoubleRingConcurrentExecutor]comm outer is empty"), HCCL_E_PARA);
    u32 ringNum = OUTER_PLANE_NUM_IN_NPRING_DOUBLE;
    u32 commIndex = currComm->commOuter[0]->Rank();
    bool bRet = commIndex >= currComm->commInner.size();
    CHK_PRT_RET(bRet,
        HCCL_ERROR("[AllGatherOperator][AllGatherDoubleRingConcurrentExecutor]commIndex[%u] >= (tag[%s]) "\
            "comm size[%llu]", commIndex, tag.c_str(), currComm->commInner.size()), HCCL_E_INTERNAL);

    commIndex = RefreshCommIdx(commIndex, nicList_, devicePhyId_);

    //  第一步，将数据从input内存拷贝到output内存的对应位置
    HcclResult ret;
    CHK_PRT_RET(currComm->commOuter.empty(),
        HCCL_ERROR("[AllGatherOperator][AllGatherDoubleRingConcurrentExecutor]errNo[0x%016llx] comm outer is empty",
            HCCL_ERROR_CODE(HCCL_E_PARA)), HCCL_E_PARA);
    CHK_SMART_PTR_NULL(currComm->commOuter[COMM_INDEX_0]);
    u32 outerRankSize = currComm->commOuter[COMM_INDEX_0]->RankSize();
    u32 serverIndex = 0;
    ret = currComm->commInner[commIndex]->GetRankByUserRank(userRank_, serverIndex);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AllGatherOperator][AllGatherDoubleRingConcurrentExecutor]Get "\
        " Rank[%u] by User Rank[%u] from CommInner[%u] Failed!", serverIndex, userRank_, commIndex), ret);

    u64 inputMemSize = inputMem.size();
    u64 baseOffset = serverIndex * inputMemSize * outerRankSize;
    u64 outerOffset = commIndex * inputMemSize;
    DeviceMem dstMem = outputMem.range(baseOffset + outerOffset, inputMemSize);
    CHK_SMART_PTR_NULL(dstMem);
    if (opInfo == nullptr) {
        ret = HcclD2DMemcpyAsync(dispatcher_, dstMem, inputMem, stream);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[AllGatherOperator][AllGatherDoubleRingConcurrentExecutor]all gather double "
                               "ring memcpy Failed, Offset[%llu], Size[%llu]",
                               baseOffset + outerOffset, inputMemSize), ret);
    }

    // 第二步，各个AI Server 内 multi ring all gather
    std::vector<Slice> dataSegsSlice; // 数据分成ranksize份，每份的起始偏移和大小
    std::vector<std::vector<Slice>> multRingsSliceZero; // 数据基于该rank上环0的偏移
    bRet = currComm->commOuter.size() < ringNum;
    CHK_PRT_RET(bRet, HCCL_ERROR("[[AllGatherOperator][AllGatherDoubleRingConcurrentExecutor]][]ringNum[%u] > "\
        " (tag[%s]), comm outer count[%llu]", ringNum, tag.c_str(), currComm->commOuter.size()), HCCL_E_INTERNAL);
    u32 sliceNum = currComm->commOuter[COMM_INDEX_0]->RankSize();

    CHK_RET(PrepareAllgatherSlice(sliceNum, inputMemSize, dataSegsSlice));

    //  多环数据切分
    auto mult2RingsSlice = PrepareMultiRingSlice(dataSegsSlice, tag, false, nicList_);
    std::vector<std::pair<bool, std::vector<Slice>>> mult4RingsSlice;
    // 基于2环数据切分2环SDMA+2环ROH; bool = true表示SDMA;
    u32 syncTrans = BEST_SPLIT_VALUE;
    u64 totalDataSize = inputMemSize * dataSegsSlice.size();
    if (totalDataSize <= HCCL_SDMA_RDMA_SPLIT_SIZE) {
        syncTrans = MAX_SPLIT_VALUE;
    }
    mult4RingsSlice.resize(mult2RingsSlice.size() * SLICES_FACTOR);
    for (u32 ringIndex = 0; ringIndex < mult2RingsSlice.size(); ringIndex++) {
        std::vector<Slice> sdmaSlice;
        std::vector<Slice> rdmaSlice;
        for (u32 segsIndex = 0; segsIndex < mult2RingsSlice[ringIndex].size(); segsIndex++) {
            auto totalSize = mult2RingsSlice[ringIndex][segsIndex].size;
            auto sdmaSliceOffset = mult2RingsSlice[ringIndex][segsIndex].offset;
            auto sdmaSliceSize = (totalSize <= HCCL_MIN_SLICE_ALIGN_910_73) ? totalSize:
                ((syncTrans * totalSize / MAX_SPLIT_VALUE) / HCCL_MIN_SLICE_ALIGN_910_73) * HCCL_MIN_SLICE_ALIGN_910_73;
            Slice sdmaSliceTmp;
            sdmaSliceTmp.offset = sdmaSliceOffset;
            sdmaSliceTmp.size = sdmaSliceSize;
            Slice rdmaSliceTmp;
            rdmaSliceTmp.offset = sdmaSliceOffset + sdmaSliceSize;
            rdmaSliceTmp.size = totalSize - sdmaSliceSize;
            sdmaSlice.push_back(sdmaSliceTmp);
            rdmaSlice.push_back(rdmaSliceTmp);
            HCCL_DEBUG("Ring index:%u, segId:%u, Orignal [offset %llu, size %llu], sdma [offset %llu, size %llu], "\
                "rdma [offset %llu, size %llu]", ringIndex, segsIndex, sdmaSliceOffset, totalSize,
                sdmaSliceTmp.offset, sdmaSliceTmp.size, rdmaSliceTmp.offset, rdmaSliceTmp.size);
        }
        mult4RingsSlice[ringIndex] = std::make_pair(true, sdmaSlice); // true表示使用sdma
        mult4RingsSlice[ringIndex + mult2RingsSlice.size()] = std::make_pair(false, rdmaSlice); // false表示rdma
    }
    if (syncTrans == MAX_SPLIT_VALUE) {
        mult4RingsSlice.erase(mult4RingsSlice.end() - mult2RingsSlice.size(), mult4RingsSlice.end());
    }

    //  抽取当前用于多环all gather 的output内存数据
    DeviceMem currentOutputMem = outputMem.range(baseOffset, inputMemSize * outerRankSize);
    CHK_SMART_PTR_NULL(currentOutputMem);
    CHK_RET(hcclImpl_->ActiveRingStreams(tag, stream));

    CHK_RET(MultiRingAllGatherConcurrent(tag, inputMem, currentOutputMem, count, dataType,
                                         mult4RingsSlice, stream, PROF_STAGE_1, baseOffset, opInfo));

    HCCL_INFO("all gather double ring outer run success");

    //  第三步， AI server 间 recursive halving doubling all gather
    u64 hdSize = 0;
    std::vector<u32>::iterator iterNic = std::find(nicList_.begin(), nicList_.end(), devicePhyId_);
    if (iterNic != nicList_.end()) {
        hdSize = inputMemSize * outerRankSize;
    }

    u64 hdCount = hdSize / perDataSize;
    std::unique_ptr<ExecutorBase> innerExecutor;
    u64 firstCommInnerSize = ((syncTrans * hdSize / MAX_SPLIT_VALUE) / HCCL_MIN_SLICE_ALIGN_910_73) *
                             HCCL_MIN_SLICE_ALIGN_910_73;
    std::vector<u64> sendSize{firstCommInnerSize, hdSize - firstCommInnerSize};
    std::vector<u64> sendOffset{0, firstCommInnerSize};
    innerStreamInfo_t *streamInfo = hcclImpl_->GetStreamInfo(tag);
    for (int innerCommIndex = 0; innerCommIndex < RDMA_PLANE_NUM_IN_NPRING_DOUBLE; ++innerCommIndex) {
        if (sendSize[innerCommIndex] == 0 || (!GetExternalInputEnableRdmaSdmaConcurrent() && innerCommIndex > 0)) {
            continue;
        }
        if (GetExternalInputEnableRdmaSdmaConcurrent() || UseInterServerRingAlgo(algType_)) {
            innerExecutor.reset(new (std::nothrow) AllGatherRing(dispatcher_));
            HCCL_INFO("allgather ring: using ring algo inter-server.");
        } else if (UseInterServerNHRAlgo(algType_)) {
            innerExecutor.reset(new (std::nothrow) AllGatherNHR(dispatcher_));
            HCCL_INFO("allgather ring: using nonuniform-hierarchical-ring algo inter-server.");
        } else if (UseInterServerNBAlgo(algType_)) {
            innerExecutor.reset(new (std::nothrow) AllGatherNB(dispatcher_));
            HCCL_INFO("allgather ring: using nonuniform-bruck algo inter-server.");
        } else {
            innerExecutor.reset(new (std::nothrow) AllGatherRecursiveHalvingDoubling(dispatcher_));
            HCCL_INFO("allgather ring: using halving-doubling algo inter-server.");
        }

        CHK_SMART_PTR_NULL(innerExecutor);
        std::unique_ptr<CommBase> &commInner = (innerCommIndex == 0 ? currComm->commInner[commIndex] :
                                                                      currComm->commInnerRdma[commIndex]);
        CHK_SMART_PTR_NULL(currComm->commInner[commIndex]);
        CHK_SMART_PTR_NULL(currComm->commInnerRdma[commIndex]);

        if (devNumInLevel2_ <= 1) {
            //  此处虽然带入inputMem作为scratch mem, 但inputMem 不能被使用
            u32 rankSize = commInner->RankSize();
            std::vector<Slice> inputSlices(rankSize, Slice());
            for (u32 i = 0; i < rankSize; i++) {
                inputSlices[i].size = sendSize[innerCommIndex];
                inputSlices[i].offset = hdSize * i + sendOffset[innerCommIndex];
            }
            auto &innerCommStream = streamInfo->ringStreams[innerCommIndex];
            auto ret = streamInfo->ringSignalAux[innerCommIndex]->Wait(innerCommStream, dispatcher_, PROF_STAGE_2);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AllGatherOperator][AllGatherDoubleRingConcurrentExecutor] "\
                " inner wait main [%u] failed", innerCommIndex), ret);

            CHK_RET(innerExecutor->Prepare(outputMem, outputMem, inputMem, hdCount, dataType, innerCommStream,
                HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, inputSlices, 0));

            CHK_RET(innerExecutor->RegisterProfiler((rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + commInner->Rank(),
                PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, innerCommStream));

            CHK_RET(commInner->RunExecutor(innerExecutor));

            ret = streamInfo->ringSignal[innerCommIndex]->Post(innerCommStream, dispatcher_, PROF_STAGE_2);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AllGatherOperator][AllGatherDoubleRingConcurrentExecutor] "\
                " inner post mains [%u] failed", innerCommIndex), ret);

            ret = streamInfo->ringSignalAux[innerCommIndex]->Post(stream, dispatcher_, PROF_STAGE_2);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AllGatherOperator][AllGatherDoubleRingConcurrentExecutor] "\
                " main post inner [%u] failed", innerCommIndex), ret);
        } else {
            u32 innerRankSize = currComm->commInner[COMM_INDEX_0]->RankSize();
            u64 innerBaseOffset = baseOffset * innerRankSize;
            DeviceMem innerInputMem = outputMem.range(innerBaseOffset, inputMemSize * outerRankSize);
            DeviceMem innerOutputMem = outputMem.range(innerBaseOffset, inputMemSize * outerRankSize * innerRankSize);

            std::vector<Slice> inputSlices(innerRankSize, Slice());
            for (u32 i = 0; i < innerRankSize; i++) {
                inputSlices[i].size = sendSize[innerCommIndex];
                inputSlices[i].offset = hdSize * i + sendOffset[innerCommIndex];
            }

            auto &innerCommStream = streamInfo->ringStreams[innerCommIndex];
            auto ret = streamInfo->ringSignalAux[innerCommIndex]->Wait(innerCommStream, dispatcher_, PROF_STAGE_2);

        //  此处虽然带入inputMem作为scratch mem, 但inputMem 不能被使用
            CHK_RET(innerExecutor->Prepare(innerInputMem, innerOutputMem, inputMem, hdCount, dataType,
                innerCommStream, HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, inputSlices, 0));

            u32 rankSize = commInner->RankSize();
            CHK_RET(innerExecutor->RegisterProfiler((rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + commInner->Rank(),
                PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, innerCommStream));

            CHK_RET(commInner->RunExecutor(innerExecutor));
            ret = streamInfo->ringSignal[innerCommIndex]->Post(innerCommStream, dispatcher_, PROF_STAGE_2);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AllGatherOperator][AllGatherDoubleRingConcurrentExecutor] "\
                "inner post mains [%u] failed", innerCommIndex), ret);

            ret = streamInfo->ringSignalAux[innerCommIndex]->Post(stream, dispatcher_, PROF_STAGE_2);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AllGatherOperator][AllGatherDoubleRingConcurrentExecutor] "\
                "main post inner [%u] failed", innerCommIndex), ret);

             // 超节点间做allgather
            ret = AllGatherLevel2Executor(tag, inputMem, outputMem, count, dataType, op, stream);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[AllGatherOperator][AllGatherDoubleRingConcurrentExecutor]tag[%s], all_gather failed, "\
                "return[%d]", tag.c_str(), ret), ret);
        }
        if (sendSize[innerCommIndex] == 0 || (!GetExternalInputEnableRdmaSdmaConcurrent() && innerCommIndex > 0)) {
            continue;
        }

        auto ret = streamInfo->ringSignal[innerCommIndex]->Wait(stream, dispatcher_, PROF_STAGE_2);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AllGatherOperator][AllGatherDoubleRingConcurrentExecutor] "\
            "main wait inner [%u] failed", innerCommIndex), ret);
    }
    HCCL_INFO("all gather double ring inner run success");
    return HCCL_SUCCESS;
}

HcclResult AllGatherOperator::AllGatherLevel2Executor(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
    u64 count, HcclDataType dataType, HcclReduceOp op, Stream &stream, const HcomCollOpInfo *opInfo)
{
    CommInfo *currComm;
    hcclImpl_->GetCommInfo(currComm, tag);
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(dataType, perDataSize));
    u32 commIndex = currComm->commOuter[0]->Rank();
    u64 inputMemSize = inputMem.size();
    u32 level0RankSize = currComm->commOuter[COMM_INDEX_0]->RankSize();
    u32 level1RankSize = currComm->commInner[commIndex]->RankSize();
    u32 level2RankSize = currComm->commLevel2[COMM_INDEX_0]->RankSize();
    u32 level0ServerIndex = 0;

    HcclResult ret = currComm->commOuter[COMM_INDEX_0]->GetRankByUserRank(userRank_, level0ServerIndex);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AllGatherOperator][AllGatherLevel2Executor]Get Rank[%u] by User"\
        " Rank[%u] from CommOuter[%u] Failed!", level0ServerIndex, userRank_, commIndex), ret);
    u32 level1ServerIndex = 0;
    ret = currComm->commInner[commIndex]->GetRankByUserRank(userRank_, level1ServerIndex);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AllGatherOperator][AllGatherLevel2Executor]Get Rank[%u] by User"\
        " Rank[%u] from CommInner[%u] Failed!", level1ServerIndex, userRank_, commIndex), ret);
    // 超节点间all gather (ring/R-HD)

    CHK_PRT_RET(commIndex >= currComm->commLevel2.size(),
        HCCL_ERROR("[AllGatherOperator][AllGatherDoubleRingExecutor]"\
            "commIndex[%u] >= (tag:[%s])commLevel2.size[%u]",
            commIndex, tag.c_str(), currComm->commLevel2.size()), HCCL_E_INTERNAL);

    std::unique_ptr<ExecutorBase> level2AGExecutor;
    std::unique_ptr<CommBase> &commLevel2 = currComm->commLevel2[commIndex];
    if (UseLevel2RingAlgo(algType_)) {
        level2AGExecutor.reset(new (std::nothrow) AllGatherRing(dispatcher_));
        HCCL_INFO("allgather ring: using ring algo inter-server.");
    } else {
        level2AGExecutor.reset(new (std::nothrow) AllGatherRecursiveHalvingDoubling(dispatcher_));
        HCCL_INFO("allgather ring: using halving-doubling algo inter-server.");
    }

    // 计算slice, 不同超节点相同slice
    std::vector<Slice> level2DataSegsSlice;
    Slice sliceTemp;
    for (u32 i = 0; i < level2RankSize; i++) {
        sliceTemp.size = inputMemSize;
        sliceTemp.offset = i * level1RankSize * level0RankSize * inputMemSize;
        level2DataSegsSlice.push_back(sliceTemp);
    }
    //  outputMem传整块，通过baseOffset偏移
    u64 level2BaseOffset = (level0ServerIndex + level1ServerIndex * level1RankSize) * inputMemSize;
    CHK_RET(level2AGExecutor->Prepare(outputMem, outputMem, inputMem, count, dataType, stream,
        HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, level2DataSegsSlice, level2BaseOffset));

    CHK_RET(level2AGExecutor->RegisterProfiler((
        level2RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + commLevel2->Rank(),
        PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, stream));

    CHK_RET(commLevel2->RunExecutor(level2AGExecutor));
    HCCL_INFO("allgather double ring [superpod] level2 allgather run success");

    // 第二步，各个AI Server 间 all gather (ring/NHR)
    commIndex = currComm->commOuter[0]->Rank();
    HCCL_INFO("commIdx:%u TagCommInfo[%s].commInner.size():%u", commIndex, tag.c_str(),
        currComm->commInner.size());

    CHK_PRT_RET(commIndex >= currComm->commInner.size(),
        HCCL_ERROR("[AllGatherOperator][AllGatherDoubleRingExecutor]"\
            "commIndex[%u] >= (tag:[%s])commLevel2.size[%u]",
            commIndex, tag.c_str(), currComm->commInner.size()), HCCL_E_INTERNAL);

    std::unique_ptr<ExecutorBase> level1AGExecutor;
    std::unique_ptr<CommBase> &commLevel1 = currComm->commInner[commIndex];
    if (UseInterServerRingAlgo(algType_)) {
        level1AGExecutor.reset(new (std::nothrow) AllGatherRing(dispatcher_));
        HCCL_INFO("allgather ring: using ring algo inter-server.");
    } else {
        level1AGExecutor.reset(new (std::nothrow) AllGatherNHR(dispatcher_));
        HCCL_INFO("allgather ring: using nonuniform-hierarchical-ring algo inter-server.");
    }

    // 计算slice, 不同超节点相同slice
    std::vector<Slice> level1DataSegsSlice;
    for (u32 j = 0; j < level1RankSize; j++) {
        for (u32 i = 0; i < level2RankSize; i++) {
            sliceTemp.size = inputMemSize;
            sliceTemp.offset =
                j * level0RankSize *inputMemSize + i * level1RankSize * level0RankSize * inputMemSize;
            level1DataSegsSlice.push_back(sliceTemp);
        }
    }
    //  outputMem传整块，通过baseOffset偏移?
    u64 level1BaseOffset = level0ServerIndex * inputMemSize;
    CHK_RET(level1AGExecutor->Prepare(outputMem, outputMem, inputMem, count, dataType, stream,
        HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, level1DataSegsSlice, level1BaseOffset));

    CHK_RET(level1AGExecutor->RegisterProfiler((
        level1RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + commLevel2->Rank(),
        PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, stream));

    CHK_RET(commLevel1->RunExecutor(level1AGExecutor));
    HCCL_INFO("allgather double ring [superpod] level1 allgather run success");

    // 节点内做all gather double ring
    std::vector<Slice> dataSegsSlice;
    std::vector<std::vector<Slice>> multRingsSliceZero; // 数据基于该rank上环0的偏移
    CHK_RET(PrepareAllgatherSlice(level0RankSize, inputMemSize, dataSegsSlice));

    //  多环数据切分
    multRingsSliceZero = PrepareMultiRingSlice(dataSegsSlice, tag, false, nicList_);

    // 计算slice
    std::vector<std::vector<Slice>> multRingsSlice;
    for (u32 ringIndex = 0; ringIndex < multRingsSliceZero.size(); ringIndex++) {
        std::vector<Slice> level0DataSlice;
        CHK_RET(CalculateLevel2AllgatherSlice(inputMemSize, level0RankSize, level1RankSize,
            level2RankSize, dataSegsSlice, level0DataSlice));
        multRingsSlice.push_back(level0DataSlice);
    }

    CHK_RET(hcclImpl_->ActiveRingStreams(tag, stream));
    CHK_RET(MultiRingAllGather(tag, inputMem, outputMem, count, dataType,
                               multRingsSliceZero, stream, PROF_STAGE_1, 0, opInfo));
    HCCL_INFO("allgather double ring [superpod] level2 allgather run success");
    return HCCL_SUCCESS;
}

HcclResult AllGatherOperator::AllGatherRingExecutor(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
    u64 count, HcclDataType dataType, HcclReduceOp op, Stream &stream, const HcomCollOpInfo *opInfo)
{
    HCCL_INFO("[AllGatherOperator][AllGatherRingExecutor] The AllGatherRingExecutor starts.");
    (void)op;
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(dataType, perDataSize));
    CHK_PRT_RET(perDataSize == 0,
        HCCL_ERROR("[AllGatherOperator][AllGatherRingExecutor]errNo[0x%016llx] datatype[%s] is invalid",
            HCCL_ERROR_CODE(HCCL_E_PARA), GetDataTypeEnumStr(dataType).c_str()), HCCL_E_PARA);
    CommInfo *currComm;
    hcclImpl_->GetCommInfo(currComm, tag);

    CHK_PRT_RET(currComm->commOuter.empty(), HCCL_ERROR("[AllGatherOperator][AllGatherRingExecutor]"\
        "comm outer is empty"), HCCL_E_PARA);
    u32 ringNum = (topoType_ == TopoType::TOPO_TYPE_8P_RING) ? OUTER_PLANE_NUM_IN_8PRING :
        OUTER_PLANE_NUM_IN_NPRING_SINGLE;
    u32 commIndex = (ringNum == OUTER_PLANE_NUM_IN_8PRING) ? devicePhyId_ : currComm->commOuter[0]->Rank();
    bool bRet = commIndex >= currComm->commInner.size();
    CHK_PRT_RET(bRet,
        HCCL_ERROR("[AllGatherOperator][AllGatherRingExecutor]commIndex[%u] >= (tag[%s])comm size[%llu]",
            commIndex, tag.c_str(), currComm->commInner.size()), HCCL_E_INTERNAL);

    //  第一步，将数据从input内存拷贝到output内存的对应位置
    HcclResult ret;
    CHK_PRT_RET(currComm->commOuter.empty(),
        HCCL_ERROR("[AllGatherOperator][AllGatherRingExecutor]errNo[0x%016llx] comm outer is empty",
            HCCL_ERROR_CODE(HCCL_E_PARA)), HCCL_E_PARA);
    CHK_SMART_PTR_NULL(currComm->commOuter[COMM_INDEX_0]);
    u32 outerRankSize = currComm->commOuter[COMM_INDEX_0]->RankSize();
    u32 serverIndex = 0;
    if (topoType_ == TopoType::TOPO_TYPE_8P_RING && nicList_.size() != DEVICE_EIGHT) {
        serverIndex = hcclImpl_->GetInnerCommRank(commIndex);
        CHK_PRT_RET(serverIndex == INVALID_VALUE_RANKID, HCCL_ERROR("[AllGatherOperator][AllGatherRingExecutor]"\
            "get rank of bridgeRank failed, commIdx[%u]", commIndex), HCCL_E_PARA);
    } else {
        ret = currComm->commInner[commIndex]->GetRankByUserRank(userRank_, serverIndex);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AllGatherOperator][AllGatherRingExecutor]Get Rank[%u] "\
            "by User Rank[%u] from CommInner[%u] Failed!", serverIndex, userRank_, commIndex), ret);
    }

    u64 inputMemSize = inputMem.size();
    u64 baseOffset = serverIndex * inputMemSize * outerRankSize;
    u64 outerOffset = commIndex * inputMemSize;
    DeviceMem dstMem = outputMem.range(baseOffset + outerOffset, inputMemSize);
    CHK_SMART_PTR_NULL(dstMem);
    if (opInfo == nullptr) {
        ret = HcclD2DMemcpyAsync(dispatcher_, dstMem, inputMem, stream);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[AllGatherOperator][AllGatherRingExecutor]all gather 8PringHD memcpy"
                               " Failed, Offset[%llu], Size[%llu]",
                               baseOffset + outerOffset, inputMemSize), ret);
    }

    // 第二步，各个AI Server 内 multi ring all gather
    std::vector<Slice> dataSegsSlice; // 数据分成ranksize份，每份的起始偏移和大小
    std::vector<std::vector<Slice>> multRingsSliceZero; // 数据基于该rank上环0的偏移
    bRet = currComm->commOuter.size() < ringNum;
    CHK_PRT_RET(bRet, HCCL_ERROR("[[AllGatherOperator][AllGatherRingExecutor]]ringNum[%u] > "\
        "(tag[%s])comm outer count[%llu]", ringNum, tag.c_str(), currComm->commOuter.size()), HCCL_E_INTERNAL);
    u32 sliceNum = currComm->commOuter[COMM_INDEX_0]->RankSize();

    CHK_RET(PrepareAllgatherSlice(sliceNum, inputMemSize, dataSegsSlice));

    //  多环数据切分
    if (ringNum == OUTER_PLANE_NUM_IN_8PRING) {
        multRingsSliceZero = PrepareMultiRingSlice(dataSegsSlice, tag);
    } else {
        multRingsSliceZero.push_back(dataSegsSlice);
    }

    CHK_PRT_RET(multRingsSliceZero.size() != ringNum,
        HCCL_ERROR("[AllGatherOperator][AllGatherRingExecutor]ringNum[%u] !=multRingsSliceZero size[%llu]",
            ringNum, multRingsSliceZero.size()), HCCL_E_INTERNAL);
    //  抽取当前用于多环all gather 的output内存数据
    DeviceMem currentOutputMem = outputMem.range(baseOffset, inputMemSize * outerRankSize);
    CHK_SMART_PTR_NULL(currentOutputMem);
    CHK_RET(hcclImpl_->ActiveRingStreams(tag, stream));

    CHK_RET(MultiRingAllGather(tag, inputMem, currentOutputMem, count, dataType,
                               multRingsSliceZero, stream, PROF_STAGE_1, baseOffset, opInfo));

    HCCL_INFO("all gather 8PringHD outer run success");

    //  第三步， AI server 间 recursive halving doubling all gather
    u64 hdSize = 0;
    std::vector<u32>::iterator iterNic = std::find(nicList_.begin(), nicList_.end(), devicePhyId_);
    if (iterNic != nicList_.end()) {
        hdSize = inputMemSize * outerRankSize;
    }

    u64 hdCount = hdSize / perDataSize;

    commIndex = (ringNum == OUTER_PLANE_NUM_IN_8PRING) ? devicePhyId_ : currComm->commOuter[0]->Rank();
    HCCL_INFO("commIdx:%u TagCommInfo[%s].commInner.size():%llu", commIndex, tag.c_str(),
        currComm->commInner.size());
    bool isMultiNic = topoType_ == TopoType::TOPO_TYPE_8P_RING && nicList_.size() != DEVICE_EIGHT;
    bool innRunRet = isMultiNic && (iterNic == nicList_.end());
    if (!innRunRet) { // 满足以下条件, 不做server间通信: 1. 8P ring的拓扑 2. 网口不满配 3. 当前device不出网口
        bRet = commIndex >= currComm->commInner.size();
        CHK_PRT_RET(bRet,
            HCCL_ERROR("[AllGatherOperator][AllGatherRingExecutor]commIndex[%u] >= (tag:[%s])comm_inner.size[%llu]",
                commIndex, tag.c_str(), currComm->commInner.size()), HCCL_E_INTERNAL);

        CHK_SMART_PTR_NULL(currComm->commInner[commIndex]);
        std::unique_ptr<ExecutorBase> innerExecutor;
        if (UseInterServerRingAlgo(algType_)) {
            innerExecutor.reset(new (std::nothrow) AllGatherRing(dispatcher_));
            HCCL_INFO("allgather ring: using ring algo inter-server.");
        } else if (UseInterServerNHRAlgo(algType_)) {
            innerExecutor.reset(new (std::nothrow) AllGatherNHR(dispatcher_));
            HCCL_INFO("allgather ring: using nhr algo inter-server.");
        } else if (UseInterServerNHRV1Algo(algType_)) {
            innerExecutor.reset(new (std::nothrow) AllGatherNHRV1(dispatcher_));
            HCCL_INFO("allgather ring: using nhr_v1 algo inter-server.");
        } else if (UseInterServerNBAlgo(algType_)) {
            innerExecutor.reset(new (std::nothrow) AllGatherNB(dispatcher_));
            HCCL_INFO("allgather ring: using nonuniform-bruck algo inter-server.");
        } else {
            innerExecutor.reset(new (std::nothrow) AllGatherRecursiveHalvingDoubling(dispatcher_));
            HCCL_INFO("allgather ring: using halving-doubling algo inter-server.");
        }

        CHK_SMART_PTR_NULL(innerExecutor);
        CHK_SMART_PTR_NULL(currComm->commInner[commIndex]);
        std::unique_ptr<CommBase> &commInner = currComm->commInner[commIndex];
        //  此处虽然带入inputMem作为scratch mem, 但inputMem 不能被使用
        CHK_RET(innerExecutor->Prepare(outputMem, outputMem, inputMem, hdCount, dataType, stream,
            HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, std::vector<Slice>(COMM_INDEX_0), 0));

        u32 rankSize = commInner->RankSize();
        CHK_RET(innerExecutor->RegisterProfiler((rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + commInner->Rank(),
            PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, stream));

        CHK_RET(commInner->RunExecutor(innerExecutor));
    }
    HCCL_INFO("all gather 8PringHD inner run success");

    //  网口裁剪：AI server 内多网口的allgather
    if (topoType_ == TopoType::TOPO_TYPE_8P_RING && nicList_.size() != DEVICE_EIGHT) {
        CHK_RET(hcclImpl_->ActiveRingStreams(tag, stream));

        u32 perDataSize = 0;
        CHK_RET(SalGetDataTypeSize(dataType, perDataSize));
        u64 tempCount = outputMem.size() / perDataSize;
        CHK_RET(ExecutorBase::PrepareSliceData(tempCount, perDataSize, sliceNum, 0, dataSegsSlice));
        multRingsSliceZero = PrepareMultiRingSlice(dataSegsSlice, tag, false, nicList_);
        CHK_PRT_RET(multRingsSliceZero.size() != ringNum, HCCL_ERROR("[AllGatherOperator][AllGatherRingExecutor]"\
            "ringNum[%u] != multRingsSliceZero size[%llu]", ringNum, multRingsSliceZero.size()), HCCL_E_INTERNAL);

        CHK_RET(MultiRingAllGather(tag, outputMem, outputMem, tempCount / DEVICE_EIGHT, dataType,
                                   multRingsSliceZero, stream, PROF_STAGE_1));

        HCCL_INFO("all gather 8PringHD inner chunk run success");
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherOperator::PrepareAllgatherSlice(u32 sliceNum, u64 inputMemSize,
    std::vector<Slice> &dataSegsSlice) const
{
    Slice sliceTemp;
    for (u32 i = 0; i < sliceNum; i++) { // 根据数据量计算每个环上数据的偏移和大小
        sliceTemp.size = inputMemSize;
        sliceTemp.offset = inputMemSize * i;
        dataSegsSlice.push_back(sliceTemp);
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherOperator::CalculateLevel1AllgatherSlice(u64 inputMemSize, u32 level0RankSize, u32 level1RankSize,
    std::vector<std::vector<Slice>> multRingsSliceZero, std::vector<std::vector<Slice>> &multRingsSlice) const
{
    for (u32 ringIndex = 0; ringIndex < multRingsSliceZero.size(); ringIndex++) {
        std::vector<Slice> level1DataSlice;
        for (u32 level0Idx = 0; level0Idx < level0RankSize; level0Idx++) {
            for (u32 level1Idx = 0; level1Idx < level1RankSize; level1Idx++) {
                Slice tmpSlice;
                tmpSlice.size = multRingsSliceZero[ringIndex][level0Idx].size;
                tmpSlice.offset =
                    multRingsSliceZero[ringIndex][level0Idx].offset + level1Idx * level0RankSize * inputMemSize;
                level1DataSlice.push_back(tmpSlice);
            }
        }
        multRingsSlice.push_back(level1DataSlice);
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherOperator::CalculateLevel2AllgatherSlice(u64 inputMemSize, u32 level0RankSize, u32 level1RankSize,
    u32 level2RankSize, std::vector<Slice> dataSegsSlice, std::vector<Slice> &level0DataSlice) const
{
    for (u32 i = 0; i < level0RankSize; i++) {
        for (u32 j = 0; j < level1RankSize; j++) {
            for (u32 z = 0; z < level2RankSize; z++) {
                Slice rankSliceTemp;
                rankSliceTemp.size = dataSegsSlice[i].size;
                rankSliceTemp.offset = dataSegsSlice[i].offset +
                    (j * level0RankSize * level1RankSize +  z * level1RankSize) * inputMemSize;
                level0DataSlice.push_back(rankSliceTemp);
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherOperator::SelectAlg(const std::string& tag, const OpParam& param, std::string& algName,
                                        std::string& newTag)
{
    if (userRankSize_ == 1 && GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        algName = "AllGatherSingleExecutor";
        return HCCL_SUCCESS;
    }
    HcclResult ret;
    if (deviceType_ == DevType::DEV_TYPE_310P3) {
        ret = SelectAlgfor310P3(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910) {
        ret = SelectAlgfor910A(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910B) {
        ret = SelectAlgfor910B(param, algName);
    } else {
        ret = SelectAlgfor91073(param, algName);
    }
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        newTag = tag;
    } else if (deviceType_ == DevType::DEV_TYPE_310P3) {
        newTag = tag + algName;
    } else {
        AlgTypeLevel1 algType1 = GetLevel1AlgType(algType_);
        auto level1Iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType1);
        newTag = tag + level1Iter->second + algName;
    }
    HCCL_INFO("[SelectAlg] all_gather newTag is [%s]", newTag.c_str());
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[AllGatherSelector][SelectAlg]tag[%s], all_gather failed, retrun[%d]", tag.c_str(), ret), ret);
    return ret;
}

HcclResult AllGatherOperator::SelectAlgfor310P3(const OpParam& param, std::string& algName)
{
    algName = "AllGatherFor310PExecutor";
    HCCL_INFO("[SelectAlgfor310P3] all_gather SelectAlgfor310P3 is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult AllGatherOperator::SelectAlgfor910A(const OpParam& param, std::string& algName)
{
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_4P_MESH || topoType_ == TopoType::TOPO_TYPE_2P_MESH;
    bool isRingTopo = topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING || topoType_ == TopoType::TOPO_TYPE_8P_RING;

    if (isMeshTopo) {
        algName = "AllGatherMeshExecutor";
    } else if (isRingTopo) {
        algName = "AllGatherRingExecutor";
    } else {
        algName = "AllGatherComm";
    }
    HCCL_INFO("[SelectAlgfor910A] all_gather SelectAlgfor910A is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult AllGatherOperator::SelectAlgfor910B(const OpParam& param, std::string& algName)
{
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    u64 dataSize = param.DataDes.count * unitSize; // 单位：字节
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_NP_MESH || topoType_ == TopoType::TOPO_TYPE_4P_MESH ||
        topoType_ == TopoType::TOPO_TYPE_2P_MESH || topoType_ == TopoType::TOPO_TYPE_1P_MESH;
    bool isRingTopo = topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING;

    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && !isSingleMeshAggregation_) {
        u64 cclBufferSize = cclBufferManager_.GetOutCCLbufferSize() / userRankSize_;
        std::string algTypeLevel1Tag;
        CHK_RET(AutoSelectAlgTypeLevel1(HcclCMDType::HCCL_CMD_ALLGATHER, dataSize, cclBufferSize, algTypeLevel1Tag));
        if (param.opBaseAtraceInfo != nullptr) {
            CHK_RET(param.opBaseAtraceInfo->SavealgtypeTraceInfo(algTypeLevel1Tag, param.tag));
        }
    }

    if (isMeshTopo) {
        if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            if (isSingleMeshAggregation_) {
                algName = "AllGatherMeshOpbaseExecutor";
            } else if (UseInterServerPipelineAlgo(algType_)) {
                algName = "AllGatherMeshOpbasePipelineExecutor";
            }
        }
        if (algName.empty()) {
            algName = "AllGatherMeshExecutor";
        }
    } else if (isRingTopo) {
        algName = "AllGatherRingExecutor";
    } else {
        algName = "AllGatherComm";
    }
    HCCL_INFO("[SelectAlgfor910B] all_gather SelectAlgfor910B is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult AllGatherOperator::SelectAlgfor91073(const OpParam& param, std::string& algName)
{
    if (GetExternalInputEnableRdmaSdmaConcurrent() && topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
        if (!UseInterServerRingAlgo(algType_)) {
            HcclResult ret = SetInterServerRingAlgo(algType_);
            HCCL_WARNING("[AllGatherOperator][SelectAlgfor91073] concurrent only support ring in AlgoLevel1 yet, "\
                "default is algType=ring.");
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[AllGatherOperator][SelectAlgfor91073]errNo[0x%016llx] tag[%s], AllGather concurrent "\
                    "set inter server ring algo failed", HCCL_ERROR_CODE(ret), param.tag.c_str()), ret);
        }
        algName = "AllGatherDoubleRingConcurrentExecutor";
    } else {
        if (!(UseInterServerRingAlgo(algType_) || UseInterServerNBAlgo(algType_))) {
            HcclResult ret = SetInterServerNHRAlgo(algType_);
            HCCL_WARNING("[AllGatherOperator][SelectAlgfor91073] only support ring, NB and NHR in AlgoLevel1 yet, "\
                "default is algType=NHR.");
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[AllGatherOperator][SelectAlgfor91073]errNo[0x%016llx] tag[%s], AllGather set inter server "\
                    "nhr algo failed", HCCL_ERROR_CODE(ret), param.tag.c_str()), ret);
        }
        algName = "AllGatherRingFor91073Executor";
    }
    HCCL_INFO("[SelectAlgfor91073] all_gather SelectAlgfor91073 is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

REGISTER_OP(HcclCMDType::HCCL_CMD_ALLGATHER, AllGather, AllGatherOperator);

}