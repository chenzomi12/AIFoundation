/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log.h"
#include "comm_base_pub.h"
#include "transport_pub.h"
#include "task_loader.h"

namespace hccl {
TaskLoader::TaskLoader(const s32 deviceLogicId, const HcclDispatcher dispatcher)
    : deviceLogicId_(deviceLogicId), dispatcher_(dispatcher)
{}
TaskLoader::~TaskLoader()
{
    HcclResult ret = Finalize();
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[TaskLoader][Destroy]TaskLoader Finalize failed[%d] ", ret);
    }
}

void TaskLoader::Prepare(Stream *stream, SubCommInfo outerCommInfo)
{
    // 参数保存
    stream_ = stream;
    HCCL_INFO("[TaskLoader] Prepare stream[%p]", stream_->ptr());
    commInfo_ = outerCommInfo;
    executeResult_ = HCCL_SUCCESS;
}

HcclResult TaskLoader::Init()
{
    HCCL_INFO("[TaskLoader] Init");
    ringThread_.reset(new (std::nothrow) std::thread(&TaskLoader::ThreadExecuteFn, this));
    CHK_SMART_PTR_NULL(ringThread_);
    return HCCL_SUCCESS;
}

HcclResult TaskLoader::GetExecuteResult()
{
    HCCL_INFO("[TaskLoader] ExecuteResult [%d]", executeResult_);
    return executeResult_;
}

HcclResult TaskLoader::Finalize()
{
    if (ringThread_) {
        threadExit = true;
        NotifyStart();
        if (ringThread_->joinable()) {
            ringThread_->join();
        }
        ringThread_ = nullptr;
    }
    HCCL_INFO("[TaskLoader] Finalize");
    return HCCL_SUCCESS;
}

void TaskLoader::NotifyStart()
{
    std::unique_lock<std::mutex> lock(startMtx_);
    startReady = true; // 设置标志位为 true.
    startCv_.notify_one();
    HCCL_INFO("[TaskLoader] NotifyStart");
}

void TaskLoader::WaitStart()
{
    std::unique_lock<std::mutex> lock(startMtx_);
    while (!startReady) {     // 假设标志位不为 true, 则等待...
        startCv_.wait(lock); // 当前线程被堵塞, 当标志位变为 true 之后,
    }
    startReady = false;
}

void TaskLoader::NotifyDone()
{
    std::unique_lock<std::mutex> lock(doneMtx_);
    doneReady = true;
    doneCv_.notify_one();
}

void TaskLoader::WaitDone()
{
    std::unique_lock<std::mutex> lock(doneMtx_);
    while (!doneReady) {
        doneCv_.wait(lock);
    }
    doneReady = false;
}

HcclResult TaskLoader::ExecuteTransPortTaskInfo(TaskLogicInfo &info)
{
    u32 index = info.taskLogicCmd.index;

    std::shared_ptr<Transport> destTransport = nullptr;
    if (commInfo_.virtualLinks.size() <= index) {
        HCCL_ERROR("[ExecuteTransPortTaskInfo]index[%u] is bigger than vlink size[%llu]", index,
            commInfo_.virtualLinks.size());
    } else if (commInfo_.links.size() <= index) {
        HCCL_ERROR("[ExecuteTransPortTaskInfo]index[%u] is bigger than link size[%llu]", index,
            commInfo_.links.size());
    } else {
        destTransport = commInfo_.links[index];
    }

    CHK_SMART_PTR_NULL(destTransport);

    switch (info.taskFuncType) {
        case TaskLogicFuncType::TRANSPORT_TXACK_TYPE:
            destTransport->TxAck(*stream_);
            break;
        case TaskLogicFuncType::TRANSPORT_RXACK_TYPE:
            destTransport->RxAck(*stream_);
            break;
        case TaskLogicFuncType::TRANSPORT_TXASYNC_TYPE:
            destTransport->TxAsync(info.txAsync.txMems, *stream_);
            break;
        case TaskLogicFuncType::TRANSPORT_RXASYNC_TYPE:
            destTransport->RxAsync(info.rxAsync.rxMems, *stream_);
            break;
        case TaskLogicFuncType::TRANSPORT_TXDATASIGNAL_TYPE:
            destTransport->TxDataSignal(*stream_);
            break;
        case TaskLogicFuncType::TRANSPORT_RXDATASIGNAL_TYPE:
            destTransport->RxDataSignal(*stream_);
            break;
        default:
            HCCL_ERROR("[TaskLoader][ExecuteTransPortTaskInfo]Invalid taskFuncType[%d]", info.taskFuncType);
            return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult TaskLoader::ExecuteDispatcherTaskInfo(TaskLogicInfo &info)
{
    switch (info.taskFuncType) {
        case TaskLogicFuncType::DISPATCHER_SIGNALWAIT_TYPE:
            HcclSignalWait(dispatcher_,
                info.taskLogicPara.dispatcherTaskLogicPara.signalWait.signal,
                *stream_,
                info.taskLogicPara.dispatcherTaskLogicPara.signalWait.userRank,
                info.taskLogicPara.dispatcherTaskLogicPara.signalWait.remoteRank,
                info.taskLogicPara.dispatcherTaskLogicPara.signalWait.stage,
                true);
            break;
        case TaskLogicFuncType::DISPATCHER_SIGNALRECORD_TYPE:
            HcclSignalRecord(dispatcher_,
                info.taskLogicPara.dispatcherTaskLogicPara.signalRecord.signal,
                *stream_,
                info.taskLogicPara.dispatcherTaskLogicPara.signalRecord.userRank,
                info.taskLogicPara.dispatcherTaskLogicPara.signalRecord.offset,
                info.taskLogicPara.dispatcherTaskLogicPara.signalRecord.stage,
                true, INVALID_U64);
            break;
        case TaskLogicFuncType::DISPATCHER_MEMCPYASYNC_TYPE:
            HcclMemcpyAsync(dispatcher_,
                info.taskLogicPara.dispatcherTaskLogicPara.memAsync.dst,
                info.taskLogicPara.dispatcherTaskLogicPara.memAsync.destMax,
                info.taskLogicPara.dispatcherTaskLogicPara.memAsync.src,
                info.taskLogicPara.dispatcherTaskLogicPara.memAsync.count,
                info.taskLogicPara.dispatcherTaskLogicPara.memAsync.kind,
                *stream_,
                INVALID_VALUE_RANKID, LinkType::LINK_ONCHIP);
            break;
        default:
            HCCL_ERROR("[TaskLoader][ExecuteDispatcherTaskInfo]Invalid taskFuncType[%d]", info.taskFuncType);
            return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult TaskLoader::ExecuteTaskLogicPara(TaskLogicInfo &info)
{
    if (info.taskLogicCmd.taskLogicType == TaskLogicType::TRANSPORT_TYPE) {
        CHK_RET(ExecuteTransPortTaskInfo(info));
    } else if (info.taskLogicCmd.taskLogicType == TaskLogicType::DISPATCHER_TYPE) {
        CHK_RET(ExecuteDispatcherTaskInfo(info));
    } else {
        HCCL_ERROR("[TaskLoader][ExecuteTaskLogicPara]Invalid taskLogicType[%d]", info.taskLogicCmd.taskLogicType);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult TaskLoader::ExecuteService()
{
    TaskLogicInfo taskLogicInfo;
    while (stream_->PopTaskLogicInfo(taskLogicInfo) == HCCL_SUCCESS) {
        CHK_RET(ExecuteTaskLogicPara(taskLogicInfo));
    }
    return HCCL_SUCCESS;
}

HcclResult TaskLoader::ThreadExecuteFn()
{
    threadId_ = SalGetTid();
    HCCL_INFO("[TaskLoader][ThreadExecuteFn]deviceLogicId_[%d], threadId_[%u]", deviceLogicId_, threadId_);
    CHK_RET(hrtSetDevice(deviceLogicId_));

    while (true) {
        WaitStart(); // 等待线程执行通知
        if (threadExit) {
            HCCL_INFO("[TaskLoader][ThreadExecuteFn]threadExit deviceLogicId_[%d]", deviceLogicId_);
            break;
        }
        HcclResult ret = ExecuteService();
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[TaskLoader][ThreadExecuteFn]TaskLoader run ExecuteService fail");
            executeResult_ = ret;
        }
        NotifyDone(); // 通知主进程本线程执行完成
    }
    CHK_RET(hrtResetDevice(deviceLogicId_));

    return HCCL_SUCCESS;
}

uint32_t TaskLoader::GetTid()
{
    if (threadId_ == 0) {
        threadId_ = SalGetTid();
    }
    HCCL_INFO("[TaskLoader][GetTid]deviceLogicId_[%d], threadId_[%u]", deviceLogicId_, threadId_);
    return threadId_;
}

HcclResult TaskLoader::ClearTagCommInfo()
{
    commInfo_ = SubCommInfo{};
    return HCCL_SUCCESS;
}

}  // namespace hccl
