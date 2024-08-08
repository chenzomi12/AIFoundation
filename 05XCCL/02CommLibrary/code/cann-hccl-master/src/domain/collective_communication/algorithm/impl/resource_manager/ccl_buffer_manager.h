/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCL_BUFFER_MANAGER_H
#define CCL_BUFFER_MANAGER_H

#include "mem_device_pub.h"

namespace hccl {

enum class MemAttr {
    IN_CCL_BUFFER = 0,
    OUT_CCL_BUFFER = 1
};

constexpr s64 AIV_FLAG_SIZE = 1024 * 1024; // aiv算子需要的flag区域大小
constexpr s64 AIV_DATA_SIZE = 33 * 1024 * 1024; // aiv算子需要的data区域大小

class CCLBufferManager {
public:
    CCLBufferManager();
    ~CCLBufferManager();
    HcclResult CreateCommCCLbuffer();
    HcclResult CreateCommAIVbuffer();
    HcclResult ReleaseCommCCLbuffer();
    HcclResult ReleaseCommAIVbuffer();
    HcclResult InitCCLbuffer(u64 inCCLbufferSize, u64 outCCLbufferSize);
    DeviceMem& GetInCCLbuffer();
    DeviceMem& GetInAIVbuffer();
    HcclResult GetInCCLbuffer(void* &buffer, u64 &size);
    u64 GetInCCLbufferSize();
    DeviceMem& GetOutCCLbuffer();
    DeviceMem& GetOutAIVbuffer();
    HcclResult GetOutCCLbuffer(void* &buffer, u64 &size);
    u64 GetOutCCLbufferSize();
    DeviceMem GetCommRegMem(const DeviceMem& mem, MemAttr memAttr, bool aivMode);
    HcclResult InitAlltoAllvParaBuffer(u64 inBufferSize, u64 outBufferSize);
    DeviceMem& GetInAlltoAllvParaBuffer();
    DeviceMem& GetOutAlltoAllvParaBuffer();
    void ReleaseAlltoAllvParaBuffer();
private:
    HcclResult CreateCCLbuffer(u64 size, DeviceMem &buffer);
    void* GetCCLbufferAddr(const DeviceMem &buffer);

    DeviceMem inCCLbuffer_;
    DeviceMem outCCLbuffer_;
    u64 inCCLbufferSize_;
    u64 outCCLbufferSize_;
    DeviceMem inAlltoAllvParaBuffer_;
    DeviceMem outAlltoAllvParaBuffer_;
    DeviceMem inAIVbuffer_ = DeviceMem();
    DeviceMem outAIVbuffer_ = DeviceMem();
};
} // namespace hccl

#endif // CCL_BUFFER_MANAGER_H
