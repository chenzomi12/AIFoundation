/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALL_REDUCE_OPERATOR_H
#define ALL_REDUCE_OPERATOR_H

#include "common_operator.h"

namespace hccl {
// 数据规模分类
enum class HcclDataCountType {
    HCCL_COUNT_SMALL = 0,
    HCCL_COUNT_MEDIUM,
    HCCL_COUNT_HUGE,
    HCCL_COUNT_RESERVED
};

class AllReduceOperator : public CommonOperator {
public:
    AllReduceOperator(std::unique_ptr<hcclImpl> &pImpl, std::unique_ptr<TopoMatcher> &topoMatcher);
    ~AllReduceOperator();
    HcclResult SelectAlg(const std::string& tag, const OpParam& param, std::string& algName, std::string& newTag);
    HcclResult GetAllReduceScratchSize(const u32 count, const HcclDataType dataType, u64 &scratchSize);

private:
    HcclResult SelectAlgfor310P3DUO(const OpParam& param, std::string& algName);

    HcclResult SelectAlgfor310P3(const OpParam& param, std::string& algName);

    HcclResult SelectAlgfor310PHelper(const OpParam& param, std::string& algName);

    HcclResult SelectAlgfor910A(const OpParam& param, std::string& algName);

    HcclResult SelectAlgfor910B(const OpParam& param, std::string& algName);

    HcclResult SelectAlgfor91073(const OpParam& param, std::string& algName);

    HcclResult MeshTopoSelector(std::string& algName, u64 unitSize);

    HcclResult DeterministicSelector(const OpParam& param, std::string& algName);

    HcclResult NonDeterministicSelector(const OpParam& param, std::string& algName, u64 unitSize);

    HcclDataCountType GetCountTypeForDeterAllReduce(const u64 count, const HcclDataType dataType);

    HcclResult GetScratchSizeForDeterAllReduce(const u32 count, const HcclDataType dataType,
        const u32 rankSize, u64 &outScratchSize);
};
}
#endif /** __ALL_REDUCE_OPERATOR_H__ */