/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BROADCAST_OPERATOR_H
#define BROADCAST_OPERATOR_H

#include "common_operator.h"

namespace hccl {
class BroadCastOperator : public CommonOperator {
public:
    BroadCastOperator(std::unique_ptr<hcclImpl> &pImpl, std::unique_ptr<TopoMatcher> &topoMatcher);
    ~BroadCastOperator();
    HcclResult Broadcast(const std::string &tag, void *ptr, u64 count, HcclDataType dataType, u32 root,
        Stream stream, HcomCollOpInfo *opInfo = nullptr);
    HcclResult BroadcastOutPlace(const std::string &tag, void *ptr, u64 count, HcclDataType dataType, u32 root,
        Stream stream, const std::unique_ptr<HcclOpBaseAtraceInfo> &opBaseAtraceInfo = nullptr);
    HcclResult SelectAlg(const std::string& tag, const OpParam& param, std::string& algName, std::string& newTag);

private:
    // broadcast
    HcclResult RunBroadCast(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem, u64 count,
        HcclDataType dataType, HcclReduceOp op, u32 root, Stream &stream, HcomCollOpInfo *opInfo = nullptr);
    HcclResult BroadCastMeshExecutor(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem, u64 count,
                                     HcclDataType dataType, HcclReduceOp op, u32 root, Stream &stream);

    HcclResult BroadCastComm(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem, u64 count,
                                HcclDataType dataType, HcclReduceOp op, u32 root, Stream &stream);

    HcclResult BroadCast4pRingExecutor(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem, u64 count,
        HcclDataType dataType, HcclReduceOp op, u32 root, Stream &stream);

    HcclResult BroadCastDoubleRingExecutor(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem, u64 count,
        HcclDataType dataType, HcclReduceOp op, u32 root, Stream &stream, const HcomCollOpInfo *opInfo = nullptr);

    HcclResult BroadCastRingExecutor(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem, u64 count,
        HcclDataType dataType, HcclReduceOp op, u32 root, Stream &stream, const HcomCollOpInfo *opInfo = nullptr);

    HcclResult BroadcastPlusBroadcast(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem, u64 count,
        HcclDataType dataType, HcclReduceOp op, u32 root, Stream &stream);

    HcclResult BroadcastStarExecutor(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem, u64 count,
        HcclDataType dataType, HcclReduceOp op, u32 root, Stream &stream);
    HcclResult BroadCastCommFor310P(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
        u64 count, HcclDataType dataType, u32 root, Stream &stream);

    HcclResult IsUserRankInSameServer(u32 userRankA, u32 userRankB, bool& inSameServer);

    HcclResult GetUserRankByDevIDInServerOfUserRankIn(u32 userRankIn, s32 devPyID, u32& userRankOut);

    HcclResult GetRankSliceSize(HcclDataType dataType, const u64 count, const u32 rankSize,
        std::vector<Slice> &sliceList);

    bool IsBroadcastSmallData(u64 size);

    HcclResult SelectAlgfor310P3(const OpParam& param, std::string& algName);

    HcclResult SelectAlgfor310P(const OpParam& param, std::string& algName);

    HcclResult SelectAlgfor910A(const OpParam& param, std::string& algName);

    HcclResult SelectAlgfor910B(const OpParam& param, std::string& algName);

    HcclResult SelectAlgfor91073(const OpParam& param, std::string& algName);
};
}

#endif /** _BROADCAST_OPERATOR_H__ */