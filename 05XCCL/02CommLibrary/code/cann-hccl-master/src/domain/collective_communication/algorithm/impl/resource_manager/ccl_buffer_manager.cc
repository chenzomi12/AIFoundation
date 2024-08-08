/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccl_buffer_manager.h"
#include "log.h"
#include "externalinput_pub.h"
#include "workflow_pub.h"
#include "adapter_rts_common.h"

namespace hccl {
CCLBufferManager::CCLBufferManager()
    :inCCLbuffer_(DeviceMem()), outCCLbuffer_(DeviceMem()),
    inCCLbufferSize_(0), outCCLbufferSize_(0),
    inAlltoAllvParaBuffer_(DeviceMem()), outAlltoAllvParaBuffer_(DeviceMem())
{
}

CCLBufferManager::~CCLBufferManager()
{
    ReleaseCommCCLbuffer();
    ReleaseAlltoAllvParaBuffer();
    ReleaseCommAIVbuffer();
}

HcclResult CCLBufferManager::CreateCCLbuffer(u64 size, DeviceMem &buffer)
{
    CHK_PRT_RET(!size, HCCL_INFO("[CCLBufferManager][CreateCCLbuffer]buffer size is zero. not need to malloc memory"),
        HCCL_SUCCESS);

    CHK_PRT_RET((size > ULONG_MAX),
        HCCL_ERROR("[CCLBufferManager][CreateCCLbuffer]buffer size is greater than %llu", ULONG_MAX), HCCL_E_PARA);

    buffer = DeviceMem::alloc(size);
    HCCL_INFO("[CreateCCLbuffer] buffer ptr[%p], size[%llu]", buffer.ptr(), buffer.size());
    CHK_PRT_RET(size && !buffer, HCCL_ERROR("[CCLBufferManager][CreateCCLbuffer]Create ccl buffer size[%llu] fail,"
        "please check environmental variable HCCL_BUFFSIZE.", size), HCCL_E_PTR);
    HCCL_RUN_INFO("[HCCL_TRACE][CreateCCLbuffer]Create ccl buffer success. buffer ptr[%p], size[%llu]",
        buffer.ptr(), buffer.size());
    return HCCL_SUCCESS;
}

HcclResult CCLBufferManager::CreateCommCCLbuffer()
{
    if (inCCLbuffer_.ptr() == nullptr) {
        if (inCCLbufferSize_ == 0) {
            inCCLbufferSize_ = GetExternalInputCCLBuffSize();
        }
        CHK_RET(CreateCCLbuffer(inCCLbufferSize_, inCCLbuffer_));
    }

    if (outCCLbuffer_.ptr() == nullptr) {
        if (outCCLbufferSize_ == 0) {
            outCCLbufferSize_ = GetExternalInputCCLBuffSize();
        }
        CHK_RET(CreateCCLbuffer(outCCLbufferSize_, outCCLbuffer_));
    }
    return HCCL_SUCCESS;
}

HcclResult CleanAIVbuffer(void *bufferPtr)
{
    // 从bufferPtr开始将之后的1M空间置为全零
    int32_t count = AIV_FLAG_SIZE / sizeof(int32_t);
    int32_t zeroMemTmp[count];
    for (int32_t i = 0; i < count; i++) {
        zeroMemTmp[i] = 0;
    }
    CHK_RET(hrtMemSyncCopy(bufferPtr, AIV_FLAG_SIZE, zeroMemTmp, AIV_FLAG_SIZE,
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    return HCCL_SUCCESS;
}

HcclResult CCLBufferManager::CreateCommAIVbuffer()
{
    if (inAIVbuffer_.ptr() == nullptr) {
        CHK_RET(CreateCCLbuffer(AIV_DATA_SIZE, inAIVbuffer_));
        CHK_RET(CleanAIVbuffer(static_cast<u8 *>(inAIVbuffer_.ptr()) + (AIV_DATA_SIZE - AIV_FLAG_SIZE)));
    }

    if (outAIVbuffer_.ptr() == nullptr) {
        CHK_RET(CreateCCLbuffer(AIV_FLAG_SIZE, outAIVbuffer_));
        CHK_RET(CleanAIVbuffer(outAIVbuffer_.ptr()));
    }
    return HCCL_SUCCESS;
}

HcclResult CCLBufferManager::ReleaseCommCCLbuffer()
{
    if (inCCLbuffer_.ptr() == nullptr && outCCLbuffer_.ptr() == nullptr) {
        HCCL_RUN_INFO("[HCCL_TRACE][ReleaseCCLbuffer]CCLBuffer is null, no need to release.");
        return HCCL_SUCCESS;
    }
    if (inCCLbuffer_.ptr() != nullptr ){
        HCCL_RUN_INFO("[HCCL_TRACE][ReleaseCCLbuffer]Release inCCLbuffer. buffer ptr[%p], size[%llu]",
        inCCLbuffer_.ptr(), inCCLbuffer_.size());
        inCCLbuffer_.free();
    }
    if (outCCLbuffer_.ptr() != nullptr) {
        HCCL_RUN_INFO("[HCCL_TRACE][ReleaseCCLbuffer]Release outCCLbuffer. buffer ptr[%p], size[%llu]",
        outCCLbuffer_.ptr(), outCCLbuffer_.size());
        outCCLbuffer_.free();
    }
    if (inCCLbuffer_.ptr() == nullptr && outCCLbuffer_.ptr() == nullptr) {
        HCCL_RUN_INFO("[HCCL_TRACE][ReleaseCCLbuffer]Release ccl buffer success.");
    }
    return HCCL_SUCCESS;
}

HcclResult CCLBufferManager::ReleaseCommAIVbuffer()
{
    HCCL_RUN_INFO("[HCCL_TRACE][ReleaseAIVbuffer]Release inAIVbuffer. buffer ptr[%p], size[%llu]",
        inAIVbuffer_.ptr(), inAIVbuffer_.size());
    inAIVbuffer_.free();
    HCCL_RUN_INFO("[HCCL_TRACE][ReleaseAIVbuffer]Release outAIVbuffer. buffer ptr[%p], size[%llu]",
        outAIVbuffer_.ptr(), outAIVbuffer_.size());
    outAIVbuffer_.free();
    if (inAIVbuffer_.ptr() == nullptr && outAIVbuffer_.ptr() == nullptr) {
        HCCL_RUN_INFO("[HCCL_TRACE][ReleaseAIVbuffer]Release AIV buffer success.");
    }
    return HCCL_SUCCESS;
}

HcclResult CCLBufferManager::InitCCLbuffer(u64 inCCLbufferSize, u64 outCCLbufferSize)
{
    if (inCCLbuffer_.ptr() == nullptr) {
        CHK_RET(CreateCCLbuffer(inCCLbufferSize, inCCLbuffer_));
        inCCLbufferSize_ = inCCLbufferSize;
    }
    if (outCCLbuffer_.ptr() == nullptr) {
        CHK_RET(CreateCCLbuffer(outCCLbufferSize, outCCLbuffer_));
        outCCLbufferSize_ = outCCLbufferSize;
    }
    return HCCL_SUCCESS;
}

void* CCLBufferManager::GetCCLbufferAddr(const DeviceMem &buffer)
{
    if (!buffer) {
        return nullptr;
    } else {
        return static_cast<void *>(reinterpret_cast<u8 *>(buffer.ptr()));
    }
}

DeviceMem& CCLBufferManager::GetInCCLbuffer()
{
    return inCCLbuffer_;
}

DeviceMem& CCLBufferManager::GetInAIVbuffer()
{
    return inAIVbuffer_;
}

DeviceMem& CCLBufferManager::GetOutAIVbuffer()
{
    return outAIVbuffer_;
}

HcclResult CCLBufferManager::GetInCCLbuffer(void* &buffer, u64 &size)
{
    buffer = GetCCLbufferAddr(inCCLbuffer_);
    size = inCCLbufferSize_;
    return HCCL_SUCCESS;
}

u64 CCLBufferManager::GetInCCLbufferSize()
{
    return inCCLbufferSize_;
}

DeviceMem& CCLBufferManager::GetOutCCLbuffer()
{
    return outCCLbuffer_;
}

HcclResult CCLBufferManager::GetOutCCLbuffer(void* &buffer, u64 &size)
{
    buffer = GetCCLbufferAddr(outCCLbuffer_);
    size = outCCLbufferSize_;
    return HCCL_SUCCESS;
}

u64 CCLBufferManager::GetOutCCLbufferSize()
{
    return outCCLbufferSize_;
}

DeviceMem CCLBufferManager::GetCommRegMem(const DeviceMem &mem, MemAttr memAttr, bool aivMode)
{
    u64 commMemSize = 0;
    if ((GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) && (!aivMode)) {
        // 单算子模式时，仅在第一次集合通信时创建子通信域，注册通信内存。需要将整个CCLbuffer注册进通信域。
        if (memAttr == MemAttr::IN_CCL_BUFFER) {
            commMemSize = inCCLbufferSize_;
        } else if (memAttr == MemAttr::OUT_CCL_BUFFER) {
            commMemSize = outCCLbufferSize_;
        }
    } else {
        commMemSize = mem.size();
    }
    DeviceMem commMem = DeviceMem::create(mem.ptr(), commMemSize);
    return commMem;
}

HcclResult CCLBufferManager::InitAlltoAllvParaBuffer(u64 inBufferSize, u64 outBufferSize)
{
    CHK_RET(CreateCCLbuffer(inBufferSize, inAlltoAllvParaBuffer_));
    CHK_RET(CreateCCLbuffer(outBufferSize, outAlltoAllvParaBuffer_));
    return HCCL_SUCCESS;
}

DeviceMem& CCLBufferManager::GetInAlltoAllvParaBuffer()
{
    return inAlltoAllvParaBuffer_;
}

DeviceMem& CCLBufferManager::GetOutAlltoAllvParaBuffer()
{
    return outAlltoAllvParaBuffer_;
}

void CCLBufferManager::ReleaseAlltoAllvParaBuffer()
{
    inAlltoAllvParaBuffer_.free();
    outAlltoAllvParaBuffer_.free();
}
} // namespace hccl