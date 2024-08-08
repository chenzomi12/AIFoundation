/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_alg_exec_registry.h"

namespace hccl {

CollAlgExecRegistry *CollAlgExecRegistry::Instance()
{
    static CollAlgExecRegistry *globalExecRegistry = new CollAlgExecRegistry;
    return globalExecRegistry;
}

HcclResult CollAlgExecRegistry::Register(const std::string &tag, const CollExecCreator &collExecCreator)
{
    const std::lock_guard<std::mutex> lock(mu_);
    if (execCreators_.find(tag) != execCreators_.end()) {
        HCCL_ERROR("[CollAlgExecRegistry]Exec tag[%s] already registered.", tag.c_str());
        return HcclResult::HCCL_E_INTERNAL;
    }
    execCreators_.emplace(tag, collExecCreator);
    return HcclResult::HCCL_SUCCESS;
}

std::unique_ptr<CollExecutorBase> CollAlgExecRegistry::GetAlgExec(
    const std::string &tag, const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher)
{
    if (execCreators_.find(tag) == execCreators_.end()) {
        HCCL_DEBUG("[CollAlgExecRegistry]Creator for executor tag[%s] has not registered.", tag.c_str());
        return nullptr;
    }
    HCCL_DEBUG("[CollAlgExecRegistry][GetAlgExec]get executor by algName[%s]", tag.c_str());
    return std::unique_ptr<CollExecutorBase>(execCreators_[tag](dispatcher, topoMatcher));
}

} // namespace Hccl