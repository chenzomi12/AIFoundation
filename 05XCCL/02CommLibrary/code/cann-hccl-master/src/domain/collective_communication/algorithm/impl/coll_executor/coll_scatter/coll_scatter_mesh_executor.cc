/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include "coll_scatter_mesh_executor.h"

namespace hccl {
CollScatterMeshExecutor::CollScatterMeshExecutor(const HcclDispatcher dispatcher,
                                std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollScatterExecutor(dispatcher, topoMatcher)
{
}

HcclResult CollScatterMeshExecutor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    HCCL_INFO("[CollScatterMeshExecutor][CalcLevel0CommInfo]tag[%s ]start", tag_.c_str());
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);

    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    HCCL_INFO("[CollScatterMeshExecutor][CalcLevel0CommInfo]tag[%s] Calc meshComm finish", tag_.c_str());
    return HCCL_SUCCESS;
}

HcclResult CollScatterMeshExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = topoAttr_.deviceNumPerAggregation > 1U ? topoAttr_.deviceNumPerAggregation - 1U : 1U;
    streamNum = totalStreamNum - 1U;
    return HCCL_SUCCESS;
}

HcclResult CollScatterMeshExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    Stream& stream = const_cast<Stream&>(param.stream);

    u32 perDataSize = SIZE_TABLE[param.DataDes.dataType];

    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 level0LocalRank = level0CommInfo.localRank;
    u32 level0LocalRankSize = level0CommInfo.localRankSize;

    u32 commIndex = level0LocalRank;

    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);
    u32 level1LocalRank = level1CommInfo.localRank;
    u32 level1LocalRankSize = level1CommInfo.localRankSize;

    bool bRet = level0LocalRankSize == 0;
    CHK_PRT_RET(bRet, HCCL_ERROR("[CollScatterMeshExecutor][KernelRun]tag[%s],comm outer is empty", tag_.c_str()),
        HCCL_E_INTERNAL);

    /* ***********第一步: 节点间scatter ****************************/
    u32 subRoot = topoMatcher_->GetSubRootForScatter(param.root);
    CHK_PRT_RET(subRoot == INVALID_VALUE_RANKID,
        HCCL_ERROR("[CollScatterMeshExecutor][KernelRun]GetSubRootForScatter failed, ",
        "userRank[%u], root[%u], subRoot[%u]", topoAttr_.userRank, param.root, subRoot), HCCL_E_INTERNAL);
    HCCL_DEBUG("[CollScatterMeshExecutor][KernelRun]GetSubRootForScatter, userRank[%u], root[%u], subRoot[%u]",
        topoAttr_.userRank, param.root, subRoot);
    CHK_RET(KernelRunInner(execMem.inputMem, execMem.count, param.DataDes.dataType, commIndex,
        param.root, subRoot, COMM_LEVEL1, stream));

    /* ***********第二步: 节点内scatter*****************************/
    // 根据数据量算每个环上数据的偏移和大小
    u32 sliceNum = level0LocalRankSize;
    std::vector<Slice> dataSegsSlice;
    CHK_RET(PrepareDataSlice(execMem.count, perDataSize, sliceNum, dataSegsSlice));

    // 每个server分配的slice大小
    u64 serverSliceSize = execMem.inputMem.size() / level1LocalRankSize;
    // 每个服务器对应的偏移
    u64 serverSliceOffset = serverSliceSize * level1LocalRank;
    DeviceMem scatterMeshInput = execMem.inputMem.range(serverSliceOffset, serverSliceSize);
    CHK_SMART_PTR_NULL(scatterMeshInput);
    DeviceMem scatterMeshOutput = execMem.inputMem.range(serverSliceOffset, serverSliceSize);
    CHK_SMART_PTR_NULL(scatterMeshOutput);

    std::unique_ptr<ExecutorBase> outerExecutor;
    outerExecutor.reset(
        new (std::nothrow) ScatterMesh(dispatcher_, level0LocalRank, level0LocalRankSize));
    CHK_SMART_PTR_NULL(outerExecutor);

    // 偏移需要带入prepare
    u32 rootRankOuter = 0;
    CHK_RET(GetRankByUserRank(COMM_LEVEL0, COMM_INDEX_0, param.root, rootRankOuter));
    CHK_PRT_RET(rootRankOuter == INVALID_VALUE_RANKID,
        HCCL_ERROR("[CollScatterMeshExecutor][KernelRun]rootRankOuter[%u] is invalid, userRank[%u], subRoot[%u]",
        rootRankOuter, topoAttr_.userRank, subRoot), HCCL_E_INTERNAL);

    CHK_RET(outerExecutor->Prepare(scatterMeshInput, scatterMeshOutput, execMem.inputMem, execMem.count,
        param.DataDes.dataType, stream, HCCL_REDUCE_RESERVED, rootRankOuter, dataSegsSlice, serverSliceOffset));

    HcclResult ret = RunTemplate(outerExecutor, level0CommInfo);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollScatterMeshExecutor][KernelRun]scatter(mesh) RunTemplate failed,return[%d]", ret), ret);

    // 将scratchMem赋值给outputMem
    u8 *scatterMeshOutputPtr = static_cast<u8 *>(scatterMeshOutput.ptr());
    DeviceMem resultMem(scatterMeshOutputPtr + execMem.outputMem.size() * level0LocalRank, execMem.outputMem.size());
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, execMem.outputMem, resultMem, stream));
    return HCCL_SUCCESS;
}

REGISTER_EXEC("ScatterMeshExecutor", ScatterMesh, CollScatterMeshExecutor);

}

