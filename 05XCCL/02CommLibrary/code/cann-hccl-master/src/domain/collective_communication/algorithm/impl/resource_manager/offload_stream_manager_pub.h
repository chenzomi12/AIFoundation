/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OFFLOAD_STREAM_MANAGER_PUB_H
#define OFFLOAD_STREAM_MANAGER_PUB_H

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <map>

#include "base.h"
#include "stream_pub.h"

namespace hccl {
class OffloadStreamManager {
public:
    OffloadStreamManager();
    ~OffloadStreamManager();

    // 将指定 Stream 注册为 Master Stream
    HcclResult RegisterMaster(const std::string &tag, Stream &stream);

    // 基于 tag 初始化 Slave Stream 资源池
    HcclResult RegisterSlaves(const std::string &tag, std::vector<Stream> &stream);

    // 基于 tag 获取 Master Stream
    Stream GetMaster(const std::string &tag);

    // 基于 tag 批量获取 Slave Stream 资源
    std::vector<Stream> GetSlaves(const std::string &tag, u32 num);

    // 基于 tag 销毁相应 Slave Streams
    HcclResult ClearSlaves(const std::string &tag);

    // 清空所有Slave streams
    HcclResult ClearSlaves();

    // delete copy and move constructors and assign operators
    OffloadStreamManager(OffloadStreamManager const&) = delete;                 // Copy construct
    OffloadStreamManager(OffloadStreamManager&&) = delete;                      // Move construct
    OffloadStreamManager& operator=(OffloadStreamManager const&) = delete;      // Copy assign
    OffloadStreamManager& operator=(OffloadStreamManager &&) = delete;          // Move assign

private:
    std::map<std::string, Stream> masterMap_;                // tag: master stream
    std::map<std::string, std::vector<Stream> > slavesMap_;  // tag : slave streams
    std::mutex masterMapMutex_;
    std::mutex slavesMapMutex_;
};
}  // namespace hccl

#endif /* OFFLOAD_STREAM_MANAGER_PUB_H */