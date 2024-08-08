/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_pipeline.h"
#include "externalinput_pub.h"
#include "workflow_pub.h"

namespace hccl {

u64 CalculatePiplineSliceNum(HcclCMDType opType, u64 dataSize, AlgType algType, DevType deviceType,
    u32 deviceNumPerAggregation, u32 moduleNum)
{
    u64 piplineSliceNum = 0;
    bool isInterRing = false;
    switch (algType) {
        case AlgType::ALG_DOUBLE_RING_PLUS_RING:
        case AlgType::ALG_8P_RING_PLUS_RING:
        case AlgType::ALG_4P_MESH_PLUS_RING:
        case AlgType::ALG_2P_MESH_PLUS_RING:
        case AlgType::ALG_1P_MESH_PLUS_RING:
        case AlgType::ALG_4P_RING_PLUS_RING:
        case AlgType::ALG_NP_SINGLE_RING_PLUS_RING:
        case AlgType::ALG_NP_MESH_PLUS_RING:
            isInterRing = true;
            break;
        default:
            isInterRing = false;
            break;
    }

    do {
        if (!GetExternalInputHcclEnablePipline()) {
            break;
        }
        /* 不支持pipline流水的场景 */
        // 支持的硬件场景
        if (deviceType != DevType::DEV_TYPE_910B || deviceNumPerAggregation < HCCL_DEVICE_NUM_TWO ||
            moduleNum < HCCL_DEVICE_NUM_TWO) {
            break;
        }
        // 支持的算子和算法场景
        if (opType != HcclCMDType::HCCL_CMD_ALLREDUCE ||
           (isInterRing && moduleNum > MAX_RING_PIPLINE_SERVER_NUM)) {
            break;
        }
        u64 sliceNumTemp = std::min(dataSize / deviceNumPerAggregation / MIN_PER_LINK_DATA_SIZE, MAX_PIPLINE_SLICE_NUM);
        // 图模式切分数量 <= 1时, 不做切分
        if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB &&
            sliceNumTemp <= MIN_PIPLINE_SLICE_NUM) {
            break;
        }

        /* 支持pipline流水, 但数据量不足以进行切分的场景 */
        // Server间使用Ring算法, 且单Server数据量<64KB时, 不做切分
        if ((isInterRing && dataSize / moduleNum < MIN_RING_DATA_SIZE)) {
            sliceNumTemp = 1;
        }
        // 支持pipline但数据量不满足切分条件时, 返回1, 用于单算子场景预申请流资源
        piplineSliceNum = (sliceNumTemp == 0) ? 1 : sliceNumTemp;
    } while (0);
    return piplineSliceNum;
}
}