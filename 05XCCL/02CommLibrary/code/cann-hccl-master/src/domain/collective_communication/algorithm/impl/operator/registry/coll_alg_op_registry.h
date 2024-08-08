/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_ALG_OP_REGISTRY_H
#define COLL_ALG_OP_REGISTRY_H

#include <map>
#include <functional>
#include <memory>

#include "coll_alg_operator.h"

namespace hccl {

using CollAlgOpCreator = std::function<CollAlgOperator *(std::unique_ptr<hcclImpl> &, std::unique_ptr<TopoMatcher> &)>;

template <typename P> static CollAlgOperator *DefaultOpCreator(std::unique_ptr<hcclImpl> &pImpl,
                                                               std::unique_ptr<TopoMatcher> &topoMatcher)
{
    static_assert(std::is_base_of<CollAlgOperator, P>::value, "CollAlgOp type must derived from Hccl::CollAlgOperator");
    return new (std::nothrow) P(pImpl, topoMatcher);
}

class CollAlgOpRegistry {
public:
    static CollAlgOpRegistry *Instance();
    HcclResult Register(const HcclCMDType &opType, const CollAlgOpCreator &collAlgOpCreator);
    std::unique_ptr<CollAlgOperator> GetAlgOp(const HcclCMDType &opType, std::unique_ptr<hcclImpl> &pImpl,
                                              std::unique_ptr<TopoMatcher> &topoMatcher);

private:
    std::map<HcclCMDType, const CollAlgOpCreator> opCreators_;
    mutable std::mutex mu_;
};

#define REGISTER_OP_HELPER(ctr, type, name, collOpBase)       \
    static HcclResult g_func_##name##_##ctr             \
        = CollAlgOpRegistry::Instance()->Register(type, DefaultOpCreator<collOpBase>)

#define REGISTER_OP_HELPER_1(ctr, type, name, collOpBase) REGISTER_OP_HELPER(ctr, type, name, collOpBase)

#define REGISTER_OP(type, name, collOpBase) REGISTER_OP_HELPER_1(__COUNTER__, type, name, collOpBase)

}   // namespace hccl

#endif