/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "workspace_resource_impl.h"
#include "workspace_resource.h"

namespace hccl {
WorkspaceResource::WorkspaceResource(u32 devicePhyId, s32 deviceLogicId, CCLBufferManager *cclBufferManagerPtr)
{
    pimpl_.reset((new (std::nothrow)WorkspaceResourceImpl(devicePhyId, deviceLogicId, cclBufferManagerPtr)));
}

WorkspaceResource::~WorkspaceResource()
{
}

HcclResult WorkspaceResource::GetWorkspaceMemSize(const std::string &opType, u64 count,
    HcclDataType dataType, u32 rankSize, u64 &memSize, DevType deviceType) const
{
    CHK_SMART_PTR_NULL(pimpl_);
    return pimpl_->GetWorkspaceMemSize(opType, count, dataType, rankSize, memSize, deviceType);
}

HcclResult WorkspaceResource::RegisterMaster(const std::string &tag, Stream stream)
{
    CHK_SMART_PTR_NULL(pimpl_);
    return pimpl_->RegisterMaster(tag, stream);
}

// 基于tag 初始设置资源，包含 Stream 资源 和 内存 资源
HcclResult WorkspaceResource::SetWorkspaceResource(const std::string &tag, void *memPtr,
    u64 &maxSize, std::vector<rtStream_t> &stream)
{
    CHK_SMART_PTR_NULL(pimpl_);
    return pimpl_->SetWorkspaceResource(tag, memPtr, maxSize, stream);
}

// 基于 tag 销毁资源，包含 Stream 资源 和 内存 资源
void WorkspaceResource::DestroyWorkspaceResource(const std::string &tag)
{
    CHK_SMART_PTR_RET_NULL(pimpl_);
    pimpl_->DestroyWorkspaceResource(tag);
    return;
}

// 销毁 Workspace全局资源，包含 Stream 资源 和 内存 资源
void WorkspaceResource::DestroyWorkspaceResource()
{
    CHK_SMART_PTR_RET_NULL(pimpl_);
    pimpl_->DestroyWorkspaceResource();
    return;
}

// 基于tag 分配 Stream 资源
std::vector<Stream> WorkspaceResource::AllocSlaveStreams(const std::string &tag, u32 num)
{
    HCCL_DEBUG("[WorkspaceResource][AllocSlaveStreams]requesting for [%d] slaves, tag[%s].", num, tag.c_str());
    // 安全性的保护，无实际业务意义
    if (!pimpl_) {
        HCCL_ERROR("[WorkspaceResource][AllocSlaveStreams] pimpl_ is nullptr.");
        return std::vector<Stream>();
    }
    return pimpl_->AllocSlaveStreams(tag, num);
}

// 基于tag 销毁 Stream 资源
HcclResult WorkspaceResource::DestroyStream(const std::string &tag)
{
    CHK_SMART_PTR_NULL(pimpl_);
    return pimpl_->DestroyStream(tag);
}

DeviceMem WorkspaceResource::AllocDeviceMem(const std::string &tag, u64 size)
{
    // 安全性的保护，无实际业务意义
    if (!pimpl_) {
        HCCL_ERROR("[WorkspaceResource][AllocDeviceMem] pimpl_ is nullptr.");
        return DeviceMem::create(nullptr, 0);
    }
    return pimpl_->AllocDeviceMem(tag, size);
}

// 基于tag 销毁 DeviceMem 资源
HcclResult WorkspaceResource::DestroyDeviceMem(const std::string &tag)
{
    CHK_SMART_PTR_NULL(pimpl_);
    return pimpl_->DestroyDeviceMem(tag);
}

HcclResult WorkspaceResource::CreateOpBasedResources(const HcclCMDType &opType, const std::string &tag,
    const HcomCollOpInfo &opInfo)
{
    CHK_SMART_PTR_NULL(pimpl_);
    return pimpl_->CreateOpBasedResources(opType, tag, opInfo);
}

HcclResult WorkspaceResource::CreateRemoteOpBasedResources(u64 memSize, const std::string &tag)
{
    CHK_SMART_PTR_NULL(pimpl_);
    return pimpl_->CreateRemoteOpBasedResources(memSize, tag);
}

HcclResult WorkspaceResource::CreateOrUpdateRemoteOpBasedResources(u64 memSize, const std::string &tag)
{
    CHK_SMART_PTR_NULL(pimpl_);
    return pimpl_->CreateOrUpdateRemoteOpBasedResources(memSize, tag);
}

HcclResult WorkspaceResource::DestroyRemoteOpBasedMem(const std::string &tag)
{
    CHK_SMART_PTR_NULL(pimpl_);
    return pimpl_->DestroyRemoteOpBasedMem(tag);
}
}  // namespace hccl
