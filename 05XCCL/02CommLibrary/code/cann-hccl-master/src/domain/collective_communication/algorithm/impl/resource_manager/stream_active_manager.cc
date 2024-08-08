/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "stream_active_manager.h"
#include "adapter_rts_common.h"

using namespace hccl;
std::atomic<bool> StreamActiveManager::initFlag_ = {false};
StreamActiveManager::StreamActiveManager()
{
}

StreamActiveManager::~StreamActiveManager()
{
    std::unique_lock<std::mutex> lock(streamActiveManagerMutex_);
    initFlag_ = false;
    streamActiveManager_.clear();
    lock.unlock();
}

StreamActiveManager &StreamActiveManager::GetInstance(s32 deviceLogicID)
{
    static StreamActiveManager streamActiveManager[MAX_MODULE_DEVICE_NUM];
    if (static_cast<u32>(deviceLogicID) >= MAX_MODULE_DEVICE_NUM) {
        HCCL_WARNING("[HcomGetCtxHomInfo][GetInstance] deviceLogicID[%d] is invalid", deviceLogicID);
        return streamActiveManager[0];
    }
    return streamActiveManager[deviceLogicID];
}

HcclResult StreamActiveManager::Init()
{
    initFlag_ = true;
    return HCCL_SUCCESS;
}

HcclResult StreamActiveManager::StreamActive(HcclRtStream activeStream, HcclRtStream stream)
{
    if (initFlag_ && streamActiveManager_.count(activeStream) == 0) {
        CHK_RET(hrtStreamActive(activeStream, stream));
        std::unique_lock<std::mutex> lock(streamActiveManagerMutex_);
        streamActiveManager_.insert(activeStream);
        lock.unlock();
        s32 activeStreamId = 0;
        CHK_RET(hrtGetStreamId(activeStream, activeStreamId));
        s32 streamId = 0;
        CHK_RET(hrtGetStreamId(stream, streamId));
        HCCL_INFO("StreamActive: activeStream[%d] stream[%d]", activeStreamId, streamId);
    }
    return HCCL_SUCCESS;
}

// ge在model析构时，先销毁流、在unload task，此时hccl获取不到流id
HcclResult StreamActiveManager::StreamsUnactive(const std::vector<Stream> &streams)
{
    if (initFlag_) {
        for (auto curStream : streams) {
            std::unique_lock<std::mutex> lock(streamActiveManagerMutex_);
                if (streamActiveManager_.count(curStream.ptr()) == 1) {
                    streamActiveManager_.erase(curStream.ptr());
                }
            lock.unlock();
        }
    }
    return HCCL_SUCCESS;
}
