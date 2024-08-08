/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_BASE_STREAM_MANAGER_PUB_H
#define OP_BASE_STREAM_MANAGER_PUB_H

#include <cstdint>
#include <string>
#include <vector>
#include <memory>

#include "base.h"
#include "stream_pub.h"

namespace hccl {
constexpr u64 MAX_SUBSTREAM_NUM = 16; // 最多支持申请的substream数量

class OpBaseStreamManager {
public:
    OpBaseStreamManager();
    ~OpBaseStreamManager();

    // 将指定 stream 注册为 master stream
    HcclResult RegisterMaster(Stream stream);

    // // 基于rtStream 注册 master stream，判断 master stream 是否需要重新构造
    HcclResult RegisterMaster(rtStream_t rtStream);

    // 如果 master stream 不存在，分配指定类型的 master stream
    HcclResult AllocMaster(const StreamType streamType);

    // 分配指定数量指定类型的 slave stream
    std::vector<Stream> AllocSlaves(const StreamType streamType, u32 num);

    // 获取Master Stream
    Stream GetMaster();

    // 清空所有 slave stream
    HcclResult ClearSlaves();

    // delete copy and move constructors and assign operators
    OpBaseStreamManager(OpBaseStreamManager const&) = delete;                 // Copy construct
    OpBaseStreamManager(OpBaseStreamManager&&) = delete;                      // Move construct
    OpBaseStreamManager& operator=(OpBaseStreamManager const&) = delete;      // Copy assign
    OpBaseStreamManager& operator=(OpBaseStreamManager &&) = delete;          // Move assign

private:
    HcclResult SetSlaveMode(Stream &slave);

    Stream master_;
    std::vector<Stream> slaves_;
    std::mutex masterMutex_;
    std::mutex slavesMutex_;
};
}  // namespace hccl

#endif /* OP_BASE_STREAM_MANAGER_PUB_H */