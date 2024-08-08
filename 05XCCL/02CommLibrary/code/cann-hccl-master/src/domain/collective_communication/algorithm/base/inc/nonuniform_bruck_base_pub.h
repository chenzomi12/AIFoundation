/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NONUNIFORM_BRUCK_BASE_PUB_H
#define NONUNIFORM_BRUCK_BASE_PUB_H

#include <cmath>
#include "executor_base_pub.h"

namespace hccl {

constexpr u64 NB_ALLREDUCE_SMALL_SIZE = 128 * 1024; // server间allreduce数据大小128k及以下不切片

class NBBase : public ExecutorBase {
public:
    explicit NBBase(const HcclDispatcher dispatcher);
    ~NBBase() override;

    u32 CalcCeilLog2(const u32 num);

private:
};
}  // hccl

#endif  /* NONUNIFORM_BRUCK_BASE_PUB_H */
