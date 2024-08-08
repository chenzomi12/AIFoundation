/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_alg_op_registry.h"

namespace hccl {

CollAlgOpRegistry *CollAlgOpRegistry::Instance()
{
    static CollAlgOpRegistry *globalOpRegistry = new CollAlgOpRegistry;
    return globalOpRegistry;
}

HcclResult CollAlgOpRegistry::Register(const HcclCMDType &opType, const CollAlgOpCreator &collAlgOpCreator)
{
    const std::lock_guard<std::mutex> lock(mu_);
    if (opCreators_.find(opType) != opCreators_.end()) {
        HCCL_ERROR("[CollAlgOpRegistry]Op Type[%d] already registered.", opType);
        return HcclResult::HCCL_E_INTERNAL;
    }
    opCreators_.emplace(opType, collAlgOpCreator);
    return HcclResult::HCCL_SUCCESS;
}

std::unique_ptr<CollAlgOperator> CollAlgOpRegistry::GetAlgOp(
    const HcclCMDType &opType, std::unique_ptr<hcclImpl> &pImpl, std::unique_ptr<TopoMatcher> &topoMatcher)
{
    if (opCreators_.find(opType) == opCreators_.end()) {
        HCCL_ERROR("[CollAlgOpRegistry]Creator for op type[%d] has not registered.", opType);
        return nullptr;
    }
    return std::unique_ptr<CollAlgOperator>(opCreators_[opType](pImpl, topoMatcher));
}

} // namespace Hccl