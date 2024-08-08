/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gather_operator.h"
#include "executor_impl.h"

namespace hccl {
GatherOperator::GatherOperator(std::unique_ptr<hcclImpl> &pImpl, std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAlgOperator(pImpl, topoMatcher, HcclCMDType::HCCL_CMD_GATHER)
{
}

GatherOperator::~GatherOperator()
{
}

HcclResult GatherOperator::Gather(const std::string &tag, void *inputPtr, void *outputPtr, u32 rootRank, u64 inputCount,
    HcclDataType dataType, Stream stream)
{
    /* ------------集合通信资源准备------------ */
    DeviceMem inputMem = DeviceMem::create(inputPtr, inputCount * SIZE_TABLE[dataType]);
    DeviceMem outputMem;

    if (userRank_ == rootRank) {
        RPT_INPUT_ERR(
            outputPtr == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
            std::vector<std::string>({"HcomGather", "outputPtr", "nullptr", "please check outputPtr"}));
        CHK_PTR_NULL(outputPtr);
    }

    outputMem = DeviceMem::create(outputPtr, userRankSize_ * inputCount * SIZE_TABLE[dataType]);

    if (isHaveCpuRank_) {
        algType_ = AlgType::ALG_NP_STAR;
    }
    CHK_RET(hcclImpl_->PrepareCommRes(tag, inputMem, outputMem, algType_, stream, rootRank, false, isHaveCpuRank_));

    // 添加从流profiling, 用于维护planID
    CHK_RET(hcclImpl_->AddSubStreamToProfiling(tag, HcclCMDType::HCCL_CMD_GATHER));

    /*  ------------执行算法-------------- */
    HcclResult ret = GatherStarExecutor(tag, inputMem, outputMem, inputCount, dataType,
        HcclReduceOp::HCCL_REDUCE_RESERVED, rootRank, stream);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[GatherOperator][Gather]errNo[0x%016llx] tag[%s],gather run failed",
            HCCL_ERROR_CODE(ret), tag.c_str()), ret);

    HCCL_INFO("tag[%s],gather run success", tag.c_str());
    return HCCL_SUCCESS;
}

HcclResult GatherOperator::GatherStarExecutor(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
    u64 count, HcclDataType dataType, HcclReduceOp op, u32 root, Stream &stream)
{
    std::unique_ptr<ExecutorBase> gstarExecutor(new (std::nothrow) GatherStar(dispatcher_, userRank_));
    CHK_SMART_PTR_NULL(gstarExecutor);

    std::vector<u32> nicRankList{0, 1};
    CHK_RET(gstarExecutor->Prepare(inputMem, outputMem, inputMem, count, dataType, stream, op, root,
        std::vector<Slice>(0), 0, nicRankList));

    CommInfo *currComm;
    hcclImpl_->GetCommInfo(currComm, tag);
    CHK_PRT_RET(currComm->commOuter.size() == 0, HCCL_ERROR("commOuter size is zero"), HCCL_E_PARA);
    std::unique_ptr<CommBase> &commOuter = currComm->commOuter[COMM_INDEX_0];
    CHK_SMART_PTR_NULL(commOuter);
    CHK_RET(commOuter->RunExecutor(gstarExecutor));
    return HCCL_SUCCESS;
}
}