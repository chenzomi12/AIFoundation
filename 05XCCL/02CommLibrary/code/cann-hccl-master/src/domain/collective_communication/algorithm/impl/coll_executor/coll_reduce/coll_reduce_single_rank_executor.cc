/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_reduce_single_rank_executor.h"

namespace hccl {

CollReduceSingleRankExecutor::CollReduceSingleRankExecutor(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollReduceExecutor(dispatcher, topoMatcher)
{
}

HcclResult CollReduceSingleRankExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    u64 totalSize = execMem.count * SIZE_TABLE[param.DataDes.dataType];
    ReduceType reduceType =
        ((param.reduceType != HCCL_REDUCE_PROD) && (param.DataDes.dataType != HCCL_DATA_TYPE_INT64)) ?
        ReduceType::INLINE_REDUCE :
        ReduceType::TBE_REDUCE;
    auto autoSelectedAlgTypeLevel1 = static_cast<u32>(algType_) >> HCCL_LEVEL_ALGO_WIDTH;
    bool isRootRank = param.root == topoAttr_.realUserRank ? true : false;
    bool hugeData = IsHugeData(totalSize); // override

    auto opMeta = HcclOpMetaInfo::GetOneForReduce(isRootRank, param.root, autoSelectedAlgTypeLevel1,
        param.DataDes.dataType, reduceType, hugeData);
    CHK_RET(InitTask(dispatcher_, const_cast<Stream&>(param.stream), opMeta.isEnableCache, opMeta.GetCacheKey()));

    DeviceMem srcMem(execMem.inputPtr, totalSize);
    DeviceMem dstMem(execMem.outputPtr, totalSize);
    HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream &>(param.stream));
    CHK_RET(LaunchTask(dispatcher_, const_cast<Stream &>(param.stream)));
    return HCCL_SUCCESS;
}

REGISTER_EXEC("ReduceSingleExecutor", ReduceSingleRank, CollReduceSingleRankExecutor);

} // namespace hccl