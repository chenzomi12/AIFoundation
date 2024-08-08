/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_ALG_EXEC_REGISTRY_H
#define COLL_ALG_EXEC_REGISTRY_H

#include <unordered_map>
#include <functional>
#include <memory>

#include "coll_executor_base.h"

namespace hccl {

using CollExecCreator = std::function<CollExecutorBase *(const HcclDispatcher, std::unique_ptr<TopoMatcher> &)>;
template <typename P>
static CollExecutorBase *DefaultExecCreator(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher)
{
    static_assert(std::is_base_of<CollExecutorBase, P>::value,
        "Executor type must derived from Hccl::CollExecutorBase");
    return new (std::nothrow) P(dispatcher, topoMatcher);
}

class CollAlgExecRegistry {
public:
    static CollAlgExecRegistry *Instance();
    HcclResult Register(const std::string &tag, const CollExecCreator &collAlgExecCreator);
    std::unique_ptr<CollExecutorBase> GetAlgExec(const std::string &tag, const HcclDispatcher dispatcher,
                                                 std::unique_ptr<TopoMatcher> &topoMatcher);

private:
    std::unordered_map<std::string, const CollExecCreator> execCreators_;
    mutable std::mutex mu_;
};

#define REGISTER_EXEC_HELPER(ctr, tag, name, collExecBase)       \
    static HcclResult g_func_##name##_##ctr             \
        = CollAlgExecRegistry::Instance()->Register(tag, DefaultExecCreator<collExecBase>)

#define REGISTER_EXEC_HELPER_1(ctr, tag, name, collExecBase) REGISTER_EXEC_HELPER(ctr, tag, name, collExecBase)

#define REGISTER_EXEC(tag, name, collExecBase) REGISTER_EXEC_HELPER_1(__COUNTER__, tag, name, collExecBase)
}   // namespace hccl
#endif