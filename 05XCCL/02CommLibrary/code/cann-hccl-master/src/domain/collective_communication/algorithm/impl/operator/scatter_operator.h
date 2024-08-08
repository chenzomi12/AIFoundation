/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SCATTER_OPERATOR_H
#define SCATTER_OPERATOR_H

#include "common_operator.h"
#include "coll_alg_op_registry.h"

namespace hccl {
class ScatterOperator : public CommonOperator {
public:
    ScatterOperator(std::unique_ptr<hcclImpl> &pImpl, std::unique_ptr<TopoMatcher> &topoMatcher);
    ~ScatterOperator();
    HcclResult Scatter(const std::string &tag, void *inputPtr, void *outputPtr, u64 recvCount, HcclDataType dataType,
        u32 root, Stream stream);
    HcclResult ScatterOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 recvCount,
        HcclDataType dataType, u32 root, Stream stream,
        const std::unique_ptr<HcclOpBaseAtraceInfo> &opBaseAtraceInfo = nullptr);
    HcclResult SelectAlg(const std::string& tag, const OpParam& param, std::string& algName,
        std::string& newTag);
private:
    HcclResult RunScatter(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
        u64 count, HcclDataType dataType, u32 root, Stream &stream);
    
    HcclResult ScatterComm(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
        u64 count, HcclDataType dataType, u32 root, Stream &stream);

    HcclResult SetScatterRingPrepareMem(DeviceMem& inputMem, u64 count,
        HcclDataType dataType, u32 &perDataSize, u32 &innerRankSize, u32 &commIndex, u32 root, u32 &subRoot,
        CommInfo* &currComm, Stream& stream);

    HcclResult ScatterRingExecutor(const std::string &tag, DeviceMem& inputMem, DeviceMem& outputMem,
        u64 count, HcclDataType dataType, u32 root, Stream& stream);

    HcclResult SetScatterMeshPrepareMem(DeviceMem& inputMem, u64 count,
        HcclDataType dataType, u32 &commIndex, u32 root, u32 &subRoot, u32 &innerRankSize,
        CommInfo* &currComm, Stream& stream);

    HcclResult ScatterMeshExecutor(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
        u64 count, HcclDataType dataType, u32 root, Stream &stream);

    HcclResult ScatterOutPlaceForOneRankSize(const std::string &tag, void *inputPtr, void *outputPtr, u64 recvCount,
        HcclDataType dataType, u32 root, Stream stream,
        const std::unique_ptr<HcclOpBaseAtraceInfo> &opBaseAtraceInfo = nullptr);
};
}

#endif /** __SCATTER_OPERATOR_H__ */