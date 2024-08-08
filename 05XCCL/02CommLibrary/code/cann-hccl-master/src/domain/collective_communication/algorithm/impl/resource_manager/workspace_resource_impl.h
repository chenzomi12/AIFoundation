/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef WORKSPACE_RESOURCE_IMPL_H
#define WORKSPACE_RESOURCE_IMPL_H

#include <mutex>
#include <map>
#include <vector>
#include <hccl/hccl_types.h>
#include "base.h"
#include "stream_pub.h"
#include "mem_device_pub.h"
#include "workspace_mem.h"
#include "ccl_buffer_manager.h"
#include "offload_stream_manager_pub.h"

namespace hccl {
class WorkspaceResourceImpl {
public:
    WorkspaceResourceImpl(u32 devicePhyId, s32 deviceLogicId, CCLBufferManager *cclBufferManagerPtr);
    ~WorkspaceResourceImpl();

    HcclResult GetWorkspaceMemSize(const std::string &opType, u64 count,
        HcclDataType dataType, u32 rankSize, u64 &memSize, DevType deviceType) const;

    // 基于 tag 注册 Master Stream
    HcclResult RegisterMaster(const std::string &tag, Stream stream);

    // 基于tag 初始设置资源，包含 Stream 资源 和 内存 资源
    HcclResult SetWorkspaceResource(const std::string &tag, void *memPtr,
        u64 &maxSize, std::vector<rtStream_t> &stream);
    
    // 基于 tag 销毁资源，包含 Stream 资源 和 内存 资源
    void DestroyWorkspaceResource(const std::string &tag);

    // 销毁 Workspace全局资源，包含 Stream 资源 和 内存 资源
    void DestroyWorkspaceResource();

    // 基于 tag 批量分配 Stream 资源
    std::vector<Stream> AllocSlaveStreams(const std::string &tag, u32 num);
    
    // 基于 tag 释放 Stream 资源
    HcclResult DestroyStream(const std::string &tag);
    
    // 基于 tag 分配 DeviceMem 资源
    DeviceMem AllocDeviceMem(const std::string &tag, u64 size);

    // 基于 tag 释放 DeviceMem 资源
    HcclResult DestroyDeviceMem(const std::string &tag);

    HcclResult CreateOpBasedResources(const HcclCMDType &opType, const std::string &tag,
        const HcomCollOpInfo &opInfo);

    HcclResult CreateAndInsertDevMem(const std::string &tag, u64 memSize, std::vector<rtStream_t> &streamPtr);
    HcclResult CreateAndInsertRemoteOpStream(const std::string &tag, std::vector<rtStream_t> &streamPtr);
    HcclResult CreateRemoteOpBasedResources(u64 memSize, const std::string &tag);
    HcclResult CreateOrUpdateRemoteOpBasedResources(u64 memSize, const std::string &tag);
    HcclResult DestroyRemoteOpBasedMem(const std::string &tag);

    WorkSpaceMem workSpaceMem_;
private:
    bool IsExistResourceWorkSpaceMem(const std::string &tag);
    HcclResult SetMemResource(const std::string &tag, void *memPtr, u64 &maxSize);
    HcclResult SetStreamResource(const std::string &tag, std::vector<rtStream_t> &stream);
    HcclResult GetDevMemSize(const std::string &tag);
    HcclResult InsertDevMem(const std::string &tag, DeviceMem &deviceMem);
    HcclResult InsertRemoteOpStream(const std::string &tag, std::vector<Stream> &stream);
    HcclResult GetOpBasedMemSize(const HcclCMDType &opType, u64 &size, const HcomCollOpInfo &opInfo);

    OffloadStreamManager offloadStreamManager_;
    std::map<std::string, std::vector<Stream>> remoteOpStreamMap_;  // 当前实际没什么用
    std::map<std::string, DeviceMem> opBaseDeviceMemMap_;
    u32 devicePhyId_;
    s32 deviceLogicId_;
    u64 bufferQuota_;  // 可分配的buffer大小
    std::mutex memResMutex_;
    CCLBufferManager *cclBufferManagerPtr_;
};
}  // namespace hccl
#endif // WORKSPACE_RESOURCE_IMPL_H
