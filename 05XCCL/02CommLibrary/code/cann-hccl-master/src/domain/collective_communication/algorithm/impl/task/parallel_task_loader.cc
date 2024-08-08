/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "parallel_task_loader.h"
#include "profiling_manager_pub.h"

namespace hccl {
ParallelTaskLoader::ParallelTaskLoader(const s32 deviceLogicId, const HcclDispatcher dispatcher)
    : deviceLogicId_(deviceLogicId), dispatcher_(dispatcher), taskLoaderNum_(0)
{}

ParallelTaskLoader::~ParallelTaskLoader()
{}

HcclResult ParallelTaskLoader::Prepare(std::vector<Stream *> streamsPtr, SubCommInfo outerCommInfo)
{
    // 参数保存
    streamsPtr_.resize(streamsPtr.size());
    for (u32 streamIndex = 0; streamIndex < streamsPtr.size(); streamIndex++) {
        streamsPtr_[streamIndex] = streamsPtr[streamIndex];
    }
    commInfo_ = outerCommInfo;
    HCCL_INFO("[ParallelTaskLoader]Prepare streams size[%d], taskLoaderNum_[%u]", streamsPtr_.size(), taskLoaderNum_);

    // 当前现有的taskLoader线程可以满足业务多流的使用
    if (taskLoaderNum_ >= streamsPtr_.size()) {
        HCCL_INFO("[ParallelTaskLoader] taskloaderNum [%u]", taskLoaderNum_);
        return HCCL_SUCCESS;
    }

    streamTaskLoader_.resize(streamsPtr_.size());
    // 当前现有的taskLoader无法满足业务多流的使用，需要扩展多流资源
    for (u32 streamIndex = taskLoaderNum_; streamIndex < streamsPtr_.size(); streamIndex++) {
        streamTaskLoader_[streamIndex].reset(new (std::nothrow) TaskLoader(deviceLogicId_, dispatcher_));
        CHK_SMART_PTR_NULL(streamTaskLoader_[streamIndex]);
        HcclResult ret = streamTaskLoader_[streamIndex]->Init();
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[ParallelTaskLoader][Init]streamIndex[%u] TaskLoader failed, return[%d]", streamIndex, ret),
            ret);
    }
    taskLoaderNum_ = streamsPtr_.size();
    HCCL_INFO("[ParallelTaskLoader] Prepare success taskLoaderNum[%u]", taskLoaderNum_);

    return HCCL_SUCCESS;
}

HcclResult ParallelTaskLoader::StartTaskLoad()
{
    tidInfo_.resize(streamsPtr_.size());

    // 配置线程启动参数
    for (u32 streamIndex = 0; streamIndex < streamsPtr_.size(); streamIndex++) {
        streamTaskLoader_[streamIndex]->Prepare(streamsPtr_[streamIndex], commInfo_);
        // 获取线程ID
        tidInfo_[streamIndex] = streamTaskLoader_[streamIndex]->GetTid();
    }

    CHK_RET(hccl::ProfilingManagerPub::CallMsprofReportMultiThreadInfo(tidInfo_));

    // 通知流线程执行
    for (u32 streamIndex = 0; streamIndex < streamsPtr_.size(); streamIndex++) {
        streamTaskLoader_[streamIndex]->NotifyStart();
    }
    return HCCL_SUCCESS;
}

HcclResult ParallelTaskLoader::WaitTaskLoadFinish()
{
    // 等待流线程执行
    for (u32 streamIndex = 0; streamIndex < streamsPtr_.size(); streamIndex++) {
        streamTaskLoader_[streamIndex]->WaitDone();
        CHK_RET(streamTaskLoader_[streamIndex]->GetExecuteResult());
    }
    return HCCL_SUCCESS;
}

HcclResult ParallelTaskLoader::ClearTagCommInfo()
{
    commInfo_ = SubCommInfo{};
    for (u32 streamIndex = 0; streamIndex < streamsPtr_.size(); streamIndex++) {
        CHK_RET(streamTaskLoader_[streamIndex]->ClearTagCommInfo());
    }
    return HCCL_SUCCESS;
}
}  // namespace hccl
