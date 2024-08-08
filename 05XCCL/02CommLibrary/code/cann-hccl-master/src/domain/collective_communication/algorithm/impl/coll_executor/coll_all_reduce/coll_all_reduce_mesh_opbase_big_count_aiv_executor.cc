/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_reduce_mesh_opbase_big_count_aiv_executor.h"

namespace hccl {

CollAllReduceMeshOpbaseBigCountAivExecutor::CollAllReduceMeshOpbaseBigCountAivExecutor(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher): CollAllReduceExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = false;
}

HcclResult CollAllReduceMeshOpbaseBigCountAivExecutor::CalcStreamNum(u32& streamNum)
{
    streamNum = 0; // AIV通信不需要申请从流
    HCCL_INFO("[CollAllReduceMeshOpbaseBigCountAivExecutor][CalcStreamNum] tag[%s] streamNum[%u]",
        tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMeshOpbaseBigCountAivExecutor::GetIfNeedAivBuffer(bool &needAivBuffer)
{
    // AIV通信需要AIV buffer
    needAivBuffer = true;
    HCCL_INFO("[CollAllReduceMeshOpbaseBigCountAivExecutor][GetIfNeedAivBuffer]tag[%s] needAivBuffer is [%u].",
        tag_.c_str(), needAivBuffer);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMeshOpbaseBigCountAivExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMeshOpbaseBigCountAivExecutor::CalcTransportMemType(TransportMemType &inputType,
    TransportMemType &outputType)
{
    inputType = TransportMemType::CCL_INPUT;
    outputType = TransportMemType::AIV_OUTPUT;
    HCCL_INFO("[CollAllReduceMeshOpbaseBigCountAivExecutor][CalcTransportMemType] tag[%s] inputType[%d],"
        " outputType[%d].", tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMeshOpbaseBigCountAivExecutor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMeshOpbaseBigCountAivExecutor::Orchestrate(const OpParam& param,
    const AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    tag_ = param.tag;
    GetStreamInfo(algRes);
    algResResp_ = &algRes;

    HcclResult ret = HCCL_SUCCESS;
    ExecMem execMem;
    execMem.count = param.DataDes.count;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;

    // 单算子大数据量
    execMem.inputMem = algRes.cclInputMem; // 仍然使用CCLIN
    execMem.outputMem = algRes.aivOutputMem;
    execMem.scratchMem = algRes.scratchMem; // 不需要
    ret = KernelRun(param, execMem);

    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAllReduceMeshOpbaseBigCountAivExecutor][Orchestrate]errNo[0x%016llx] tag[%s] excutor kernel "
            "run failed", HCCL_ERROR_CODE(ret), param.tag.c_str()), ret);

    HCCL_INFO("tag[%s], AllReduce executor orchestrate success, take time [%lld]us.",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMeshOpbaseBigCountAivExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_INFO("[CollAllReduceMeshOpbaseBigCountAivExecutor][KernelRun]allreduce aiv enter");

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    void *buffersIn[MAX_RANK_SIZE];
    void *buffersOut[MAX_RANK_SIZE];

    u32 localRank = outerCommInfo.localRank;
    u32 localRankSize = outerCommInfo.localRankSize;
    HCCL_DEBUG("[CollAllReduceMeshOpbaseBigCountAivExecutor][KernelRun] userRank [%d] localRank [%d]",
        topoAttr_.userRank, localRank);

    for (u32 i = 0; i < localRankSize; i++) {
        if (i != localRank) {
            CHK_RET(outerCommInfo.links[i]->GetRemoteMem(UserMemType::INPUT_MEM, &(buffersIn[i])));
            CHK_RET(outerCommInfo.links[i]->GetRemoteMem(UserMemType::OUTPUT_MEM, &(buffersOut[i])));
        } else {
            buffersIn[i] = execMem.inputMem.ptr();
            buffersOut[i] = execMem.outputMem.ptr();
        }
    }

    bool isOpbase = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    HcclResult ret = ExecuteKernelLaunch(HcclCMDType::HCCL_CMD_ALLREDUCE, execMem.inputPtr, execMem.outputPtr,
        execMem.count, param.DataDes.dataType, param.reduceType, localRank, localRankSize, 0,
        buffersIn, buffersOut, param.stream.ptr(), isOpbase, execMem.inputMem.size());
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAllReduceMeshOpbaseBigCountAivExecutor][KernelRun]allreduce aiv failed, return[%d]", ret),
        ret);

    HCCL_INFO("[CollAllReduceMeshOpbaseBigCountAivExecutor][KernelRun]allreduce aiv run success");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllReduceMeshOpbaseBigCountAivExecutor", AllReduceMeshOpbaseBigCountAiv,
    CollAllReduceMeshOpbaseBigCountAivExecutor);

} // namespace hccl