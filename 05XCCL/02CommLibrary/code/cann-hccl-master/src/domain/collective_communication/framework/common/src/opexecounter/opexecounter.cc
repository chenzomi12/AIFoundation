/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opexecounter.h"
#include "adapter_rts_common.h"
#include "externalinput_pub.h"
#include "heartbeat_pub.h"
#include "dispatcher.h"
namespace hccl {
OpExeCounter& OpExeCounter::GetInstance(s32 deviceLogicID)
{
    static OpExeCounter opCounter[MAX_MODULE_DEVICE_NUM];
    if (static_cast<u32>(deviceLogicID) >= MAX_MODULE_DEVICE_NUM - 1) {
        HCCL_WARNING("[OpExeCounter][GetInstance] deviceLogicID[%d] is invalid", deviceLogicID);
        return opCounter[MAX_MODULE_DEVICE_NUM - 1];
    }
    return opCounter[deviceLogicID];
}

HcclResult OpExeCounter::InitCounter()
{
    DevType devType = DevType::DEV_TYPE_910;
    CHK_RET(hrtGetDeviceType(devType));
    if (!GetExternalInputOpCounter() || devType == DevType::DEV_TYPE_310P3 || devType == DevType::DEV_TYPE_910 ||
        devType == DevType::DEV_TYPE_310P1) {
        isNeedOpCounter_ = false;
        HCCL_RUN_INFO("do not need add counter");
        return HCCL_SUCCESS;
    }
    if (refCount_ == 0) {
        int32_t defCount = 0;
        CHK_RET(hrtMalloc(&headCountMem_, sizeof(int32_t)));
        CHK_PTR_NULL(headCountMem_);
        CHK_RET(hrtMemSyncCopy(headCountMem_, sizeof(int32_t), &defCount,
            sizeof(int32_t), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

        CHK_RET(hrtMalloc(&tailCountMem_, sizeof(int32_t)));
        CHK_PTR_NULL(tailCountMem_);
        CHK_RET(hrtMemSyncCopy(tailCountMem_, sizeof(int32_t), &defCount,
            sizeof(int32_t), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

        int32_t addOneVal = 1;
        CHK_RET(hrtMalloc(&addOneMem_, sizeof(int32_t)));
        CHK_PTR_NULL(addOneMem_);
        CHK_RET(hrtMemSyncCopy(addOneMem_, sizeof(int32_t), &addOneVal,
            sizeof(int32_t), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
        
        HCCL_RUN_INFO("alloc counter mem resource.");
    }
    refCount_++;
    isNeedOpCounter_ = true;
    return HCCL_SUCCESS;
}

OpExeCounter::~OpExeCounter()
{}

HcclResult OpExeCounter::DeInitCounter()
{
    if (!isNeedOpCounter_) {
        HCCL_DEBUG("do not need add counter");
        return HCCL_SUCCESS;
    }
    refCount_--;
    if (refCount_ == 0) {
        if (headCountMem_ != nullptr) {
            CHK_RET(hrtFree(headCountMem_));
            headCountMem_ = nullptr;
        }
        if (tailCountMem_ != nullptr) {
            CHK_RET(hrtFree(tailCountMem_));
            tailCountMem_ = nullptr;
        }
        if (addOneMem_ != nullptr) {
            CHK_RET(hrtFree(addOneMem_));
            addOneMem_ = nullptr;
        }
        HCCL_RUN_INFO("free counter mem resource");
    }
    return HCCL_SUCCESS;
}

HcclResult OpExeCounter::AddCounter(const HcclDispatcher &dispatcher, Stream &stream, int flag) // flag 0为下发前计数，1为下发后计数
{
    if (!isNeedOpCounter_) {
        HCCL_DEBUG("do not need add counter");
        return HCCL_SUCCESS;
    }
    if ((stream.ptr() == nullptr)) {
        HCCL_WARNING("stream is nullptr");
        return HCCL_SUCCESS;
    }
    CHK_RET(HcclReduceAsync(dispatcher, static_cast<void *>(addOneMem_), 1, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM,
        stream, (flag == HEAD) ? static_cast<void *>(headCountMem_) : static_cast<void *>(tailCountMem_),
        INVALID_VALUE_RANKID, LinkType::LINK_ONCHIP, INLINE_REDUCE_BIT));

    HCCL_DEBUG("add %s count.", (flag == HEAD) ? "head" : "tail");

    return HCCL_SUCCESS;
}

HcclResult OpExeCounter::GetCounter(std::pair<int32_t, int32_t> &counter)
{
    if (!isNeedOpCounter_) {
        HCCL_DEBUG("do not need add counter");
        return HCCL_SUCCESS;
    }
    CHK_RET(hrtMemSyncCopy(&counter.first, sizeof(int32_t), headCountMem_, sizeof(int32_t),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST));

    CHK_RET(hrtMemSyncCopy(&counter.second, sizeof(int32_t), tailCountMem_, sizeof(int32_t),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST));
    
    HCCL_DEBUG("head:%d, tail:%d", counter.first, counter.second);
    
    return HCCL_SUCCESS;
}

HcclResult FftsHeadCounter(const HcclDispatcher &dispatcher, Stream &stream)
{
    if (!GetExternalInputHcclEnableFfts() || GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        HCCL_DEBUG("do not need add ffts mode counter");
        return HCCL_SUCCESS;
    }
    s32 devLogicID = 0;
    CHK_RET(hrtGetDevice(&devLogicID));
    return OpExeCounter::GetInstance(devLogicID).AddCounter(dispatcher, stream, HEAD);
}

HcclResult FftsTailCounter(const HcclDispatcher &dispatcher, Stream &stream)
{
    if (!GetExternalInputHcclEnableFfts() || GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        HCCL_DEBUG("do not need add ffts mode counter");
        return HCCL_SUCCESS;
    }
    s32 devLogicID = 0;
    CHK_RET(hrtGetDevice(&devLogicID));
    return OpExeCounter::GetInstance(devLogicID).AddCounter(dispatcher, stream, TAIL);
}

HcclResult StarsCounter(const HcclDispatcher &dispatcher, Stream &stream, int flag)
{
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && GetExternalInputHcclEnableFfts()) {
        HCCL_DEBUG("do not need add stars mode counter");
        return HCCL_SUCCESS;
    }
    s32 devLogicID = 0;
    CHK_RET(hrtGetDevice(&devLogicID));
    return OpExeCounter::GetInstance(devLogicID).AddCounter(dispatcher, stream, flag);
}

HcclResult GetOpCounter(const s32 &deviceId, std::pair<int32_t, int32_t> &counter)
{
    return OpExeCounter::GetInstance(deviceId).GetCounter(counter);
}

__attribute__((constructor)) void CallBackInit()
{
    RegisterStuckDetectCallBack(GetOpCounter);
    RegisterInitTaskCallBack(FftsHeadCounter);
    RegisterLaunchTaskCallBack(FftsTailCounter);
}

} // namespace hccl