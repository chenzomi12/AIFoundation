/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_RUN_ALLTOALLV_FULLMESH_EXECUTOR_H
#define COLL_RUN_ALLTOALLV_FULLMESH_EXECUTOR_H
#include "coll_all_to_all_executor.h"
namespace hccl {
class CollRunAlltoAllVFullMesh : public CollAlltoAllExecutor {

public:
    CollRunAlltoAllVFullMesh(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher);
    ~CollRunAlltoAllVFullMesh() = default;

private:
    HcclResult CalcStreamNum(u32& streamNum) override;
    HcclResult CalcLevel0CommInfo(TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcLevel2CommInfo(TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalAlltoAllFullMeshCommInfo(TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport);
    HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult KernelRun(const OpParam &param, ExecMem &execMem) override;
};

} // namespace hccl
#endif