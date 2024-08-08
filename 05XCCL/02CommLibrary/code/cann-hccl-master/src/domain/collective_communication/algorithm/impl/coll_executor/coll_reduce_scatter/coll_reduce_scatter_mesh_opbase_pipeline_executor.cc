/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_reduce_scatter_mesh_opbase_pipeline_executor.h"

namespace hccl {

CollReduceScatterMeshOpbasePipelineExecutor::CollReduceScatterMeshOpbasePipelineExecutor(
    const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollReduceScatterExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = true;
}

void CollReduceScatterMeshOpbasePipelineExecutor::ParseParam(const OpParam& param)
{
    tag_ = param.tag;
}

HcclResult CollReduceScatterMeshOpbasePipelineExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = topoAttr_.deviceNumPerAggregation + 1U;
    streamNum = totalStreamNum - 1U;
    HCCL_INFO("[CollReduceScatterMeshOpbasePipelineExecutor][CalcStreamNum] tag[%s] streamNum[%u]",
        tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterMeshOpbasePipelineExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterMeshOpbasePipelineExecutor::CalcTransportMemType(TransportMemType &inputType,
    TransportMemType &outputType)
{
    inputType = TransportMemType::CCL_INPUT;
    outputType = TransportMemType::CCL_OUTPUT;
    HCCL_INFO("[CollReduceScatterMeshOpbasePipelineExecutor][CalcTransportMemType] tag[%s] inputType[%d],"
        " outputType[%d]", tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterMeshOpbasePipelineExecutor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaInfo(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    commParaInfo.meshSinglePlane = true;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaInfo, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

// PipeLine模式下使用Ring算法
HcclResult CollReduceScatterMeshOpbasePipelineExecutor::CalcLevel1CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaInfo(COMM_LEVEL1, CommType::COMM_TAG_RING_INNER);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaInfo, opTransport[COMM_LEVEL1], inputType, outputType));
    return HCCL_SUCCESS;
}

u64 CollReduceScatterMeshOpbasePipelineExecutor::CalcLoopMaxCount(const u32 unitSize)
{
    // 中转内存单次最多能够接受的output count，放开ranksize限制
    u64 maxCountPerLoop = ((inCCLbufferSize_ / (HCCL_MIN_SLICE_ALIGN_910B * PIPELINE_DEPTH)) \
            * HCCL_MIN_SLICE_ALIGN_910B - HCCL_MIN_SLICE_ALIGN_910B) / unitSize;
    HCCL_INFO("[CollReduceScatterMeshOpbasePipelineExecutor][CalcLoopMaxCount] maxCountPerLoop[%llu]", maxCountPerLoop);
    return maxCountPerLoop;
}

bool CollReduceScatterMeshOpbasePipelineExecutor::IsHugeData(const u64 curSize)
{
    bool hugeData = curSize > RDMA_SEND_MAX_SIZE || curSize > SDMA_SEND_MAX_SIZE;
    return hugeData;
}

HcclResult CollReduceScatterMeshOpbasePipelineExecutor::RunLoop(const OpParam &param, const AlgResourceResponse &algRes)
{
    HCCL_INFO("[CollReduceScatterMeshOpbasePipelineExecutor][RunLoop] begins.");

    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    ReduceType reduceType = ((param.reduceType != HCCL_REDUCE_PROD) &&
        (param.DataDes.dataType != HCCL_DATA_TYPE_INT64)) ?
        ReduceType::INLINE_REDUCE : ReduceType::TBE_REDUCE;

    u8 *curInputPtr = static_cast<u8 *>(param.inputPtr);
    u8 *curOutputPtr = static_cast<u8 *>(param.outputPtr);
    CHK_PTR_NULL(curInputPtr);
    CHK_PTR_NULL(curOutputPtr);

    u64 maxCountPerLoop = CalcLoopMaxCount(unitSize);
    HCCL_DEBUG("[CollReduceScatterMeshOpbasePipelineExecutor][RunLoop]tag[%s], userRankSize is [%llu], maxCountPerLoop "
        "is [%llu].", param.tag.c_str(), topoAttr_.userRankSize, maxCountPerLoop);

    // 先获取 comm inner \ comm outer 的value
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 commIndex = outerCommInfo.localRank;

    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo innerCommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    DeviceMem userInMem = DeviceMem::create(param.inputPtr, param.DataDes.count * unitSize);
    u64 reduceAttr = GetReduceAttr(userInMem, const_cast<DeviceMem&>(algRes.cclInputMem), param.DataDes.dataType,
        param.reduceType); // scratchMem用的cclin
    u64 bufferSize = algRes.cclInputMem.size();

    auto originalAlgTypeLevel1 = static_cast<u32>(algType_) >> HCCL_LEVEL_ALGO_WIDTH;

    for (u64 countLeft = param.DataDes.count, curCount = 0, curOffset = 0, curSize = 0; countLeft > 0;
        countLeft -= curCount) {
        curInputPtr += curSize;
        curOutputPtr += curSize;

        curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        curSize = curCount * unitSize;

        HCCL_DEBUG("[CollReduceScatterMeshOpbasePipelineExecutor][RunLoop]tag[%s], curOffset[%llu], " \
            "curInputPtr[%p], curOutputPtr[%p], curCount[%llu], dataType[%d].",
            param.tag.c_str(), curOffset, curInputPtr, curOutputPtr, curCount, param.DataDes.dataType);

        bool hugeData = IsHugeData(curSize);
        auto meta = HcclOpMetaInfo::GetOneForReduceScatter(originalAlgTypeLevel1, param.DataDes.dataType, reduceType,
            hugeData);
        CHK_RET(InitTask(dispatcher_, const_cast<Stream&>(param.stream), meta.isEnableCache, meta.GetCacheKey()));

        ExecMem execMem;
        execMem.count = curCount;
        execMem.inputMem = algRes.cclInputMem;
        execMem.outputMem = algRes.cclOutputMem;
        execMem.scratchMem = algRes.scratchMem;
        // 使用当前Loop偏移到的地址作为当前的inputPtr和outputPtr
        execMem.inputPtr = curInputPtr;
        execMem.outputPtr = curOutputPtr;

        std::unique_ptr<ReduceScatterPipeline> executor;
        executor.reset(new (std::nothrow) ReduceScatterPipeline(dispatcher_, reduceAttr));
        CHK_SMART_PTR_NULL(executor);

        HcomCollOpInfo opInfo = {"", execMem.inputPtr, execMem.outputPtr, param.DataDes.count, param.DataDes.dataType,
            param.root, param.reduceType};

        CHK_RET(executor->Prepare(&opInfo, execMem.inputMem, curCount, bufferSize, curOffset, outerCommInfo,
            innerCommInfo, const_cast<Stream&>(param.stream), streamInfo_.ringStreams, streamInfo_.ringSignal,
            streamInfo_.ringSignalAux));
        CHK_RET(executor->RunAsync());

        CHK_RET(LaunchTask(dispatcher_, const_cast<Stream&>(param.stream)));

        curOffset += curSize;
    }
    return HCCL_SUCCESS;
}

REGISTER_EXEC("ReduceScatterMeshOpbasePipelineExecutor", ReduceScatterMeshOpbasePipeline,
    CollReduceScatterMeshOpbasePipelineExecutor);

}

