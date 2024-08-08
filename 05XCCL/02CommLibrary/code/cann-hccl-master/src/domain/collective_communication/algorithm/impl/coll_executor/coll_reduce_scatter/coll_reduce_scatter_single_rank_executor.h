/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_REDUCESCATTER_SINGLE_RANK_EXECUTOR_H
#define COLL_REDUCESCATTER_SINGLE_RANK_EXECUTOR_H
#include "coll_reduce_scatter_executor.h"

namespace hccl {
class CollReduceScatterSingleRankExecutor : public CollReduceScatterExecutor {

public:
    explicit CollReduceScatterSingleRankExecutor(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher);
    ~CollReduceScatterSingleRankExecutor() = default;

private:
    HcclResult KernelRun(const OpParam &param, ExecMem &execMem) override;
};

} // namespace hccl

#endif