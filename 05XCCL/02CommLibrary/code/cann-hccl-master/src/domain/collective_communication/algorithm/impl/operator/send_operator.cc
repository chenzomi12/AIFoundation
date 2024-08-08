/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#include "send_operator.h"
#include "rank_consistent.h"
#include "executor_impl.h"
 
namespace hccl {
SendOperator::SendOperator(std::unique_ptr<hcclImpl> &pImpl, std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAlgOperator(pImpl, topoMatcher, HcclCMDType::HCCL_CMD_SEND)
{
}
 
SendOperator::~SendOperator()
{
}
 
HcclResult SendOperator::SelectAlg(const std::string& tag, const OpParam& param, std::string& algName,
                                        std::string& newTag)
{
    algName = "SendExecutor";
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        newTag = tag;
    } else {
        newTag = tag + algName;
    }
    HCCL_INFO("[SelectAlg] send newTag is [%s]", newTag.c_str());
    return HCCL_SUCCESS;
}

REGISTER_OP(HcclCMDType::HCCL_CMD_SEND, Send, SendOperator);
}