/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_ALLREDUCE_FOR_310P_DOUBLING_DIRECT_EXECUTOR_H
#define COLL_ALLREDUCE_FOR_310P_DOUBLING_DIRECT_EXECUTOR_H
#include "coll_all_reduce_executor.h"
namespace hccl {
class CollAllReduceFor310PDoublingDirectExecutor : public CollAllReduceExecutor {

public:
    CollAllReduceFor310PDoublingDirectExecutor(const HcclDispatcher dispatcher,
                                               std::unique_ptr<TopoMatcher> &topoMatcher);
    ~CollAllReduceFor310PDoublingDirectExecutor() = default;

private:
    /* *************** 资源计算 *************** */
    HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcTransportMemType(TransportMemType &inputType, TransportMemType &outputType);
    HcclResult CalcLevel0CommInfo(TransportMemType inputType,
        TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;

    /* *************** 算法编排 *************** */
    HcclResult KernelRun(const OpParam &param, ExecMem &execMem) override;
};
} // namespace hccl

#endif