/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "adapter_rts_common.h"
#include "offload_stream_manager.h"

namespace hccl {
OffloadStreamManager::OffloadStreamManager() = default;
OffloadStreamManager::~OffloadStreamManager() = default;

HcclResult OffloadStreamManager::RegisterMaster(const std::string &tag, Stream &stream)
{
    std::unique_lock<std::mutex> lock(masterMapMutex_);
    masterMap_[tag] = stream;
    HCCL_DEBUG("[OffloadStreamManager][RegisterMaster]register master stream[%p] success, tag[%s].",
        stream.ptr(), tag.c_str());
    return HCCL_SUCCESS;
}

HcclResult OffloadStreamManager::RegisterSlaves(const std::string &tag, std::vector<Stream> &stream)
{
    HCCL_DEBUG("[OffloadStreamManager][RegisterSlaves]start register slaves stream, tag[%s], size[%d].",
        tag.c_str(), stream.size());

    std::unique_lock<std::mutex> lock(slavesMapMutex_);
    auto iter = slavesMap_.find(tag);
    if (iter != slavesMap_.end()) {
        HCCL_ERROR("[OffloadStreamManager][RegisterSlaves]in offload stream manager, register slaves fail,"
            "tag[%s] has existed", tag.c_str());
        return HCCL_E_PARA;
    }
    slavesMap_.insert(std::make_pair(tag, stream));
    HCCL_INFO("[OffloadStreamManager][RegisterSlaves]register slaves stream success, tag[%s], size[%d].",
        tag.c_str(), stream.size());
    return HCCL_SUCCESS;
}

Stream OffloadStreamManager::GetMaster(const std::string &tag)
{
    std::unique_lock<std::mutex> lock(masterMapMutex_);
    auto iter = masterMap_.find(tag);
    if (iter == masterMap_.end()) {
        HCCL_ERROR("[OffloadStreamManager][GetMaster]cant find tag[%s]", tag.c_str());
        return Stream();
    }
    return iter->second;
}

std::vector<Stream> OffloadStreamManager::GetSlaves(const std::string &tag, u32 num)
{
    HCCL_DEBUG("[OffloadStreamManager][GetSlaves]requesting for [%d] slaves, tag[%s].", num, tag.c_str());
    if (num == 0) {
        HCCL_WARNING("[OffloadStreamManager][GetSlaves]requesting for 0 slaves, return empty vector.");
        return std::vector<Stream>(0);
    }

    std::unique_lock<std::mutex> lock(slavesMapMutex_);
    auto iter = slavesMap_.find(tag);
    if (iter == slavesMap_.end()) {
        HCCL_ERROR("[OffloadStreamManager][GetSlaves]cant find tag[%s]", tag.c_str());
        return std::vector<Stream>();
    }

    if (iter->second.size() < num) {
        HCCL_ERROR("[OffloadStreamManager][GetSlaves]" \
            "trying to get [%d] slaves fail, only [%d] slaves available, tag[%s].",
            num, iter->second.size(), tag.c_str());
        return std::vector<Stream>();
    }

    std::vector<Stream> res(iter->second.begin(), iter->second.begin() + num);
    iter->second.erase(iter->second.begin(), iter->second.begin() + num);
    HCCL_INFO("[OffloadStreamManager][GetSlaves]get [%d] slaves success, returning.", res.size());
    return res;
}

HcclResult OffloadStreamManager::ClearSlaves(const std::string &tag)
{
    std::unique_lock<std::mutex> lock(slavesMapMutex_);
    auto iter = slavesMap_.find((tag));
    if (iter != slavesMap_.end()) {
        slavesMap_.erase(tag);
    }
    HCCL_DEBUG("[OffloadStreamManager][ClearSlaves]Destroy slaves stream success, tag[%s]", tag.c_str());
    return HCCL_SUCCESS;
}

HcclResult OffloadStreamManager::ClearSlaves()
{
    std::unique_lock<std::mutex> lock(slavesMapMutex_);
    slavesMap_.clear();
    HCCL_DEBUG("[OffloadStreamManager][ClearSlaves]clear all slave streams.");
    return HCCL_SUCCESS;
}

}  // namespace hccl
