/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_BASE_PUB_H
#define OP_BASE_PUB_H

#include <vector>
#include <hccl/hccl_types.h>
#include <hccl/hccl.h>

#include "hccl/base.h"

struct OpBaseMemPara {
    u64 beginIndex;
    u64 count;
    u64 tmpMemSize;
};

struct GatherPara {
    std::vector<u64> addrInfo;
    std::vector<u64> addrInfoCountPerRank;
    u32 rankSize;
    s32 addrLength;
};
#endif  // OP_BASE_PUB_H