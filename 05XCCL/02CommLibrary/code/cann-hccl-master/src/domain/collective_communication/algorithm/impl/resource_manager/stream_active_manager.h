/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STREAM_ACTIVE_MANAGER_H
#define STREAM_ACTIVE_MANAGER_H
#include <unordered_set>
#include <atomic>
#include "hccl_common.h"
#include "stream_pub.h"

namespace hccl {
class StreamActiveManager {
public:
    static StreamActiveManager& GetInstance(s32 deviceLogicID);
    HcclResult Init();
    HcclResult StreamActive(HcclRtStream activeStream, HcclRtStream stream);
    HcclResult StreamsUnactive(const std::vector<Stream> &streams);
private:
    StreamActiveManager();
    ~StreamActiveManager();
    std::unordered_set<HcclRtStream> streamActiveManager_; // set中存放当前进程中以已由hccl激活的stream
    std::mutex streamActiveManagerMutex_;
    static std::atomic<bool> initFlag_;
};
}  // namespace hccl
#endif // STREAM_ACTIVE_MANAGER_H
