/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REDUCE_OPERATOR_H
#define REDUCE_OPERATOR_H

#include "common_operator.h"
#include "coll_alg_op_registry.h"

namespace hccl {
class ReduceOperator : public CommonOperator {
public:
    ReduceOperator(std::unique_ptr<hcclImpl> &pImpl, std::unique_ptr<TopoMatcher> &topoMatcher);
    ~ReduceOperator();
    HcclResult Reduce(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
        HcclDataType dataType, HcclReduceOp op, u32 root, Stream stream);
    HcclResult ReduceOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
        HcclDataType dataType, HcclReduceOp op, u32 root, Stream stream,
        const std::unique_ptr<HcclOpBaseAtraceInfo> &opBaseAtraceInfo = nullptr);
private:

    HcclResult RunReduce(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem, u64 count,
                               HcclDataType dataType, HcclReduceOp op, u32 root, Stream &stream);

    HcclResult ReduceRingPlusHd(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem, u64 count,
                               HcclDataType dataType, HcclReduceOp op, u32 root, Stream &stream);

    HcclResult ReduceComm(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem, u64 count,
                                HcclDataType dataType, HcclReduceOp op, u32 root, Stream &stream);

    HcclResult ReduceMeshExecutor(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem, u64 count,
        HcclDataType dataType, HcclReduceOp op, u32 root, Stream &stream, HcomCollOpInfo *opInfo = nullptr);

    HcclResult ReduceDoubleRingExecutor(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem, u64 count,
        HcclDataType dataType, HcclReduceOp op, u32 root, Stream &stream, HcomCollOpInfo *opInfo = nullptr);

    HcclResult ReduceOutPlaceForOneRankSize(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
        HcclDataType dataType, HcclReduceOp op, u32 root, Stream stream,bool isRootRank,ReduceType reduceType,
        const std::unique_ptr<HcclOpBaseAtraceInfo> &opBaseAtraceInfo = nullptr);

    // 算法选择
    HcclResult SelectAlg(const std::string &tag, const OpParam &param, std::string &algName, std::string &newTag);
    HcclResult SelectAlgfor910A(const OpParam& param, std::string& algName);    // 算法选择 - 910A
    HcclResult SelectAlgfor910B(const OpParam& param, std::string& algName);    // 算法选择 - 910B
    HcclResult SelectAlgfor91073(const OpParam& param, std::string& algName);    // 算法选择 - 91073
};
}

#endif /** __REDUCE_EXECUTOR__ */