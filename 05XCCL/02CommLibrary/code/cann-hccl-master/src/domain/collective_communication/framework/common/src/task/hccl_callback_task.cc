/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "externalinput_pub.h"
#include "workflow_pub.h"
#include "callback_thread_manager.h"
#include "device_capacity.h"
#include "hccl_callback_task.h"
#include "sal_pub.h"

namespace hccl {
constexpr u32 HOST_NIC_THREAD_WAIT = 10000;    // 等待hostnic监控线程时间1s(10000 * 100us);

HcclCallbackTask::HcclCallbackTask(u32 devicePhyId, u32 deviceLogicId,
    HcclDispatcher dispatcher, NICDeployment nicDeployment)
    : devicePhyId_(devicePhyId), deviceLogicId_(deviceLogicId),
    dispatcher_(dispatcher), nicDeployment_(nicDeployment),
    callbackThread_(nullptr), callbackThreadId_(INVALID_U64),
    callbackThreadShutDown_(false)
{
}

HcclCallbackTask::~HcclCallbackTask()
{
    // 析构时自动停止线程
    CloseCallbackThread();
}

void HcclCallbackTask::CallbackThread()
{
    CHK_PRT(hrtSetDevice(deviceLogicId_));
    callbackThreadId_ = pthread_self();
    while (!callbackThreadShutDown_) {
        // 等待1000ms，等待callback函数的返回
        HcclResult ret = hrtProcessReport(1000);
        if (ret != HCCL_SUCCESS) {
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
        }
    }
    CHK_PRT(hrtResetDevice(deviceLogicId_));
}

HcclResult HcclCallbackTask::CallbackRegStream(rtStream_t stream)
{
    // callback线程绑定stream场景：
    // 1、910A host RDMA、host tcp
    // 2、310P 2pg device侧下沉 tcp、标准roce（当前device SOC下沉场景都需要callback task，若之后有改动，判断条件需要更新）
    if ((static_cast<s32>(devicePhyId_) == HOST_DEVICE_ID) || (stream == nullptr) ||
        ((nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_DEVICE && !GetExternalInputHcclIsTcpMode() &&
        !Is310PDevice()))) {
        return HCCL_SUCCESS;
    }

    if (HcclGetCallbackResult(dispatcher_) != HCCL_SUCCESS) {
        HCCL_ERROR("[HcclCallbackTask][CallbackRegStream]errNo[0x%016llx] callback func err",
            HCCL_ERROR_CODE(HcclGetCallbackResult(dispatcher_)));
        return HcclGetCallbackResult(dispatcher_);
    }

    if (callbackThread_ == nullptr) {
        callbackThread_.reset(new (std::nothrow) std::thread(&HcclCallbackTask::CallbackThread, this));
        CHK_SMART_PTR_NULL(callbackThread_);
    }

    u32 countTime = 0;
    while (callbackThreadId_ == INVALID_U64) {
        SaluSleep(ONE_MILLISECOND_OF_USLEEP);
        countTime++;
        CHK_PRT_RET(countTime >= HOST_NIC_THREAD_WAIT,
            HCCL_ERROR("[HcclCallbackTask][CallbackRegStream]errNo[0x%016llx] waiting Callback Thread time out",
                HCCL_ERROR_CODE(HCCL_E_INTERNAL)), HCCL_E_INTERNAL);
    }

    // 如果当前stream已经被注册，则直接返回成功
    if (ThreadStreamManager::Instance().StreamHasBeenReged(stream)) {
        HCCL_INFO("[HcclCallbackTask][CallbackRegStream]Cur stream Already registered, stream[%p] tid:[%llu] ",
            stream, callbackThreadId_);
        return HCCL_SUCCESS;
    } else {
        CHK_RET(hrtSubscribeReport(callbackThreadId_, stream));
        // ReleaseTidAndStream 暂无相应调用
        CHK_RET(ThreadStreamManager::Instance().RegTidAndStream(callbackThreadId_, stream));
        HCCL_INFO("[HcclCallbackTask][CallbackRegStream]rt Subscribe Report success[%llu]", callbackThreadId_);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCallbackTask::CloseCallbackThread()
{
    if (nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_DEVICE && !GetExternalInputHcclIsTcpMode() &&
        !Is310PDevice()) {
        return HCCL_SUCCESS;
    }

    callbackThreadShutDown_ = true;
    if (callbackThread_ != nullptr && callbackThread_->joinable()) {
        callbackThread_->join();
        callbackThread_ = nullptr;
    }

    callbackThreadId_ = INVALID_U64;
    return HCCL_SUCCESS;
}
} // namespace hccl
