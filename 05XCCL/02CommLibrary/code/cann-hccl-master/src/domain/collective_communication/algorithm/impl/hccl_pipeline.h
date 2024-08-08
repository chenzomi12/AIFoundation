/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_PIPELINE_H
#define HCCL_PIPELINE_H

#include "hccl_common.h"

namespace hccl {
constexpr u32 MAX_RING_PIPLINE_SERVER_NUM = 128; // 防止qp耗尽, Ring算法下Server间流水并行最多支持128 Server
constexpr u32 MIN_PER_LINK_DATA_SIZE = 4 * 1024 * 1024; // Server间流水并行分到每条链路上的最小数据量
constexpr u32 MIN_RING_DATA_SIZE = 64 * 1024; // Ring算法下, Server间支持流水并行的最小数据量
constexpr u64 MAX_PIPLINE_SLICE_NUM = 4; // 流水并行算法最大切分次数
constexpr u64 MIN_PIPLINE_SLICE_NUM = 2; // 流水并行算法最小切分次数

// 计算pipline流水并行的切分数量
u64 CalculatePiplineSliceNum(HcclCMDType opType, u64 dataSize, AlgType algType, DevType deviceType,
    u32 deviceNumPerAggregation, u32 moduleNum);
}
#endif
