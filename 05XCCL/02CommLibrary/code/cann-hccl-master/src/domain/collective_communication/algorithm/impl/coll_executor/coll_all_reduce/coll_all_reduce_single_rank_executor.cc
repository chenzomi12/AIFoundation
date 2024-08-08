/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_reduce_single_rank_executor.h"

namespace hccl {

CollAllReduceSingleRankExecutor::CollAllReduceSingleRankExecutor(const HcclDispatcher dispatcher,
                                                                 std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAllReduceExecutor(dispatcher, topoMatcher)
{
}

HcclResult CollAllReduceSingleRankExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    u64 totalSize = execMem.count * SIZE_TABLE[param.DataDes.dataType];
    ReduceType reduceType =
        ((param.reduceType != HCCL_REDUCE_PROD) && (param.DataDes.dataType != HCCL_DATA_TYPE_INT64)) ?
        ReduceType::INLINE_REDUCE : ReduceType::TBE_REDUCE;
    auto originalAlgTypeLevel1 = static_cast<u32>(algType_) >> HCCL_LEVEL_ALGO_WIDTH;
    bool hugeData = totalSize > SDMA_SEND_MAX_SIZE;
    if (execMem.inputPtr == execMem.outputPtr) {
        auto opMeta = HcclOpMetaInfo::GetOneForAllReduce(originalAlgTypeLevel1, param.DataDes.dataType, reduceType,
            totalSize <= HCCL_SMALL_COUNT_128_KB, 1, hugeData, CopyPattern::ZCOPY); // 通过CopyPattern字段区分不同子图
        CHK_RET(InitTask(dispatcher_, const_cast<Stream&>(param.stream), opMeta.isEnableCache, opMeta.GetCacheKey()));
    } else {
        auto opMeta = HcclOpMetaInfo::GetOneForAllReduce(originalAlgTypeLevel1, param.DataDes.dataType, reduceType,
            totalSize <= HCCL_SMALL_COUNT_128_KB, 1, hugeData, CopyPattern::BCOPY); // 通过CopyPattern字段区分不同子图
        CHK_RET(InitTask(dispatcher_, const_cast<Stream&>(param.stream), opMeta.isEnableCache, opMeta.GetCacheKey()));
        // ranksize = 1; intput、output地址不同，input->output
        DeviceMem srcMem(execMem.inputPtr, totalSize);
        DeviceMem dstMem(execMem.outputPtr, totalSize);
        HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream));
    }
    CHK_RET(LaunchTask(dispatcher_, const_cast<Stream&>(param.stream)));
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllReduceSingleExecutor", AllReduceSingleRank, CollAllReduceSingleRankExecutor);

} // namespace hccl