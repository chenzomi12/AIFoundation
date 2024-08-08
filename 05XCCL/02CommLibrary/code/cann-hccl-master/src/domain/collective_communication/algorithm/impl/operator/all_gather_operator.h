/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALL_GATHER_OPERATOR_H
#define ALL_GATHER_OPERATOR_H

#include "common_operator.h"

namespace hccl {
class AllGatherOperator : public CommonOperator {
public:
    AllGatherOperator(std::unique_ptr<hcclImpl> &pImpl, std::unique_ptr<TopoMatcher> &topoMatcher);
    ~AllGatherOperator();
    HcclResult AllGather(const std::string &tag, void *inputPtr, void *outputPtr, u64 inputCount,
        HcclDataType dataType, Stream stream, HcomCollOpInfo *opInfo = nullptr);
    HcclResult AllGatherOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 inputCount,
        HcclDataType dataType, Stream stream, const std::unique_ptr<HcclOpBaseAtraceInfo> &opBaseAtraceInfo = nullptr);
    HcclResult SelectAlg(const std::string& tag, const OpParam& param, std::string& algName, std::string& newTag) override;
private:
    // all gather private
    HcclResult GetAllGatherOutPlaceSplitLoop(void* commOutputPtr, bool isMeshTopo, const u32 unitSize,
        const u64 inputCount, u64 &maxCountPerLoop);
    HcclResult SetAllGatherOutPlaceHugedata(bool isMeshTopo, const u64 curSize, bool &hugeData);
    HcclResult AllGatherOutPlaceFor310P(const std::string &tag, void *inputPtr, void *outputPtr, u64 inputCount,
        HcclDataType dataType, Stream stream);
    HcclResult AllGatherOutPlaceFor310PForOneRankSize(const std::string &tag, void *inputPtr, void *outputPtr, u64 inputCount,
        HcclDataType dataType, Stream stream);
    HcclResult AllGatherCommFor310P(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
        u64 count, HcclDataType dataType, HcclReduceOp op, Stream &stream);
    HcclResult RunAllGather(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem, u64 count,
        HcclDataType dataType, HcclReduceOp op, Stream &stream, HcomCollOpInfo *opInfo = nullptr);

    HcclResult AllGatherComm(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem, u64 count,
        HcclDataType dataType, HcclReduceOp op, Stream &stream);

    HcclResult AllGatherMeshOpbaseExecutor(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem, u64 count,
        HcclDataType dataType, HcclReduceOp op, Stream &stream, HcomCollOpInfo *opInfo);

    HcclResult AllGatherMeshOpbasePipelineExecutor(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
        u64 count, HcclDataType dataType, HcclReduceOp op, Stream &stream, HcomCollOpInfo *opInfo);

    HcclResult AllGatherMeshExecutor(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem, u64 count,
        HcclDataType dataType, HcclReduceOp op, Stream &stream);

    HcclResult AllGatherDoubleRingExecutor(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem, u64 count,
        HcclDataType dataType, HcclReduceOp op, Stream &stream, const HcomCollOpInfo *opInfo = nullptr);

    HcclResult AllGatherDoubleRingConcurrentExecutor(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
        u64 count, HcclDataType dataType, HcclReduceOp op, Stream &stream, const HcomCollOpInfo *opInfo = nullptr);

    HcclResult AllGatherLevel2Executor(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
        u64 count, HcclDataType dataType, HcclReduceOp op, Stream &stream, const HcomCollOpInfo *opInfo = nullptr);

    HcclResult AllGatherRingExecutor(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem, u64 count,
        HcclDataType dataType, HcclReduceOp op, Stream &stream, const HcomCollOpInfo *opInfo = nullptr);

    HcclResult PrepareAllgatherSlice(u32 sliceNum, u64 inputMemSize, std::vector<Slice> &dataSegsSlice) const;

    HcclResult CalculateLevel1AllgatherSlice(u64 inputMemSize, u32 level0RankSize, u32 level1RankSize,
        std::vector<std::vector<Slice>> multRingsSliceZero, std::vector<std::vector<Slice>> &multRingsSlice) const;

    HcclResult CalculateLevel2AllgatherSlice(u64 inputMemSize, u32 level0RankSize, u32 level1RankSize,
        u32 level2RankSize, std::vector<Slice> dataSegsSlice, std::vector<Slice> &level0DataSlice) const;
    HcclResult SelectAlgfor310P3(const OpParam& param, std::string& algName);

    HcclResult SelectAlgfor910A(const OpParam& param, std::string& algName);

    HcclResult SelectAlgfor910B(const OpParam& param, std::string& algName);

    HcclResult SelectAlgfor91073(const OpParam& param, std::string& algName);
};

}

#endif /** __ALL_GATHER_OPERATOR_H__ */