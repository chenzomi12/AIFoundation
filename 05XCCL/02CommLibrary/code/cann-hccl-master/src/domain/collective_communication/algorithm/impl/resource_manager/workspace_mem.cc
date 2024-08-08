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
#include "workspace_mem.h"

namespace hccl {

WorkSpaceMem::WorkSpaceMem() = default;

WorkSpaceMem::~WorkSpaceMem()= default;

/* 适配GE，设备内存由上层分配，HCCL主要用于管理该设备内存 */
HcclResult WorkSpaceMem::SetMemResource(const std::string &tag, void *ptr, u64 maxSize)
{
    if (ptr == nullptr) {
        HCCL_ERROR("[Set][MemResource]contruct fail, ptr is [null], tag[%s], maxSize[%llu]", tag.c_str(), maxSize);
        return HCCL_E_PTR;
    } else {
        std::unique_lock<std::mutex> lock(memResMutex_);
        WorkSpaceMem_t memResource{};
        memResource.curPtr = ptr;
        memResource.maxSize = maxSize;
        memResMap_.erase(tag);
        memResMap_.insert(std::make_pair(tag, memResource));
        lock.unlock();
        HCCL_INFO("[Set][MemResource]set mem resource success, tag[%s] ptr[%llu] max size[%llu]",
            tag.c_str(), std::hash<void *>{}(ptr), maxSize);
    }
    return HCCL_SUCCESS;
}

void* WorkSpaceMem::AllocMem(const std::string &tag, u64 size)
{
    /* 此处可能会与并发，加锁 */
    std::unique_lock<std::mutex> lock(memResMutex_);

    auto interIter = memResMap_.find(tag);
    if (interIter == memResMap_.end()) {
        HCCL_ERROR("[Alloc][Mem] Can't find tag, tag[%s], size[%llu]", tag.c_str(), size);
        return nullptr;
    }
    if (interIter->second.curPtr == nullptr) {
        HCCL_ERROR("[Alloc][Mem] Current resource(curPtr) is null, tag[%s], size[%llu]", tag.c_str(), size);
        return nullptr;
    }

    // 判断size 是否超出最大值
    if (interIter->second.maxSize < (interIter->second.totalSize + size)) {
        HCCL_ERROR("[Alloc][Mem] Current resource size is not enough, tag[%s], size[%llu], "\
            "maxSize[%llu], totalSize[%llu]", tag.c_str(), size,
            interIter->second.maxSize, interIter->second.totalSize);
        return nullptr;
    }

    void* tempPtr = interIter->second.curPtr;
    interIter->second.curPtr = reinterpret_cast<void *>(static_cast<char *>(interIter->second.curPtr) + size);
    interIter->second.totalSize += size;

    HCCL_INFO("[Alloc][Mem]Alloc mem success, tag[%s] size[%llu], totalSize[%llu], maxSize[%llu]",
        tag.c_str(), size, interIter->second.totalSize, interIter->second.maxSize);
    lock.unlock();
    return tempPtr;
}

HcclResult WorkSpaceMem::DestroyMemResource(const std::string &tag)
{
    HCCL_INFO("[Destroy][MemResource]Destroy workspace mem, tag[%s]", tag.c_str());

    std::unique_lock<std::mutex> lock(memResMutex_);

    auto interIter = memResMap_.find(tag);
    if (interIter != memResMap_.end()) {
        memResMap_.erase(tag);
    }
    lock.unlock();

    return HCCL_SUCCESS;
}

void WorkSpaceMem::DestroyMemResource()
{
    HCCL_INFO("[Destroy][MemResource]Destroy workspace all mem");
    memResMap_.clear();
}

bool WorkSpaceMem::IsExist(const std::string &tag)
{
    std::unique_lock<std::mutex> lock(memResMutex_);
    auto interIter = memResMap_.find(tag);
    return interIter != memResMap_.end();
}
}  // namespace hccl
