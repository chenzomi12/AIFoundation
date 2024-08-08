/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "externalinput_pub.h"
#include "device_capacity.h"
#include "adapter_rts_common.h"
#include "sal_pub.h"
#include "workspace_resource_impl.h"

namespace hccl {
const std::string HCCL_KERNEL_OP_TYPE_REDUCESCATTER = "HcomReduceScatter";
const std::string HCCL_KERNEL_OP_TYPE_ALLTOALL = "HcomAllToAll";

constexpr s64 HCCL_SUB_STREAM_NUM_THREE = 3; // 最大subStream 数量为3
WorkspaceResourceImpl::WorkspaceResourceImpl(u32 devicePhyId, s32 deviceLogicId, CCLBufferManager *cclBufferManagerPtr)
    : devicePhyId_(devicePhyId), deviceLogicId_(deviceLogicId), cclBufferManagerPtr_(cclBufferManagerPtr)
{
}

WorkspaceResourceImpl::~WorkspaceResourceImpl()
{
    opBaseDeviceMemMap_.clear();
    remoteOpStreamMap_.clear();
}

HcclResult WorkspaceResourceImpl::GetWorkspaceMemSize(const std::string &opType, u64 count,
    HcclDataType dataType, u32 rankSize, u64 &memSize, DevType deviceType) const
{
    // 以一个页的大小4kb 去分配
    u64 alignSize = HCCL_ALIGN_SIZE;
    u64 tempMemSize = HCCL_WORKSPACE_MEM_32_KB;

    u32 dataTypeSize = 0;
    HcclResult ret = SalGetDataTypeSize(dataType, dataTypeSize);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[WorkspaceResourceImpl][GetWorkspaceMemSize]op[%s] dataType[%s] is invalid. ret[%d]",
            opType.c_str(), GetDataTypeEnumStr(dataType).c_str(), ret), ret);

    u64 opMemSize = 0;
    // 判断是否需要申请额外的reduce scatter scratch mem
    if (opType == HCCL_KERNEL_OP_TYPE_REDUCESCATTER) {
        if (deviceType == DevType::DEV_TYPE_310P3) {
            opMemSize += 0;
        } else {
            // ReduceScatter 算子所需memory大小为 单个数据size * count * rank_size
            opMemSize += count * dataTypeSize * rankSize;
        }
    }

    if (opType == HCCL_KERNEL_OP_TYPE_ALLTOALL) {
        opMemSize += count * dataTypeSize * rankSize;
    }

    tempMemSize += opMemSize;
    memSize = (tempMemSize + alignSize - 1) / alignSize * alignSize;
    HCCL_INFO("[WorkspaceResourceImpl][GetWorkspaceMemSize]workspace memory memSize: "
        "op[%s], data type[%s], count[%llu], rank memSize[%u], memory memSize[%llu].",
        opType.c_str(), GetDataTypeEnumStr(dataType).c_str(), count, rankSize, memSize);

    return HCCL_SUCCESS;
}

HcclResult WorkspaceResourceImpl::RegisterMaster(const std::string &tag, Stream stream)
{
    return offloadStreamManager_.RegisterMaster(tag, stream);
}

HcclResult WorkspaceResourceImpl::SetMemResource(const std::string &tag, void *memPtr, u64 &maxSize)
{
    return workSpaceMem_.SetMemResource(tag, memPtr, maxSize);
}

HcclResult WorkspaceResourceImpl::SetStreamResource(const std::string &tag, std::vector<rtStream_t> &stream)
{
    HCCL_DEBUG("[WorkspaceResourceImpl][SetStreamResource]setting stream resources, input stream size[%u], tag[%s]",
        stream.size(), tag.c_str());
    // 后继考虑是否将 OffloadStreamManager 和 WorkSpaceMem 的实现直接包在这个类中
    std::vector<Stream> offloadSlaves;
    for (u32 i = 0; i < stream.size(); i++) {
        // 当前GE WorkSpaceResource中Create的流都是从流，后继若有主流情况，再作扩展
        offloadSlaves.emplace_back(Stream(stream[i], false));
        if (!offloadSlaves[i].ptr()) {
            HCCL_ERROR("[WorkspaceResourceImpl][SetStreamResource]create offload stream[%u] fail.", i);
            return HCCL_E_INTERNAL;
        }
    }
    return offloadStreamManager_.RegisterSlaves(tag, offloadSlaves);
}

// 基于tag 初始设置资源，包含 Stream 资源 和 DeviceMem 资源
HcclResult WorkspaceResourceImpl::SetWorkspaceResource(const std::string &tag, void *memPtr,
    u64 &maxSize, std::vector<rtStream_t> &stream)
{
    // 设定 workspace memory 资源
    if (memPtr == nullptr) {
        HCCL_WARNING("[WorkspaceResourceImpl][SetWorkspaceResource] workspace mem ptr is null, tag[%s] maxSize[%llu]",
            tag.c_str(), maxSize);
    } else {
        CHK_RET(SetMemResource(tag, memPtr, maxSize));
    }

    /* 设定 workspace stream 资源 */
    if (stream.size() != 0) {
        CHK_RET(SetStreamResource(tag, stream));
    }
    return HCCL_SUCCESS;
}

// 基于 tag 销毁资源，包含 Stream 资源 和 DeviceMem 资源
void WorkspaceResourceImpl::DestroyWorkspaceResource(const std::string &tag)
{
    // 销毁 work space memory 资源
    HcclResult ret = workSpaceMem_.DestroyMemResource(tag);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[WorkspaceResourceImpl][DestroyWorkspaceResource]Destroy workspace mem failed. ret[%d]", ret);
    }
    
    // 销毁 work space stream资源
    // 疑问：为什么有这个判断？
    if (static_cast<s32>(devicePhyId_) != HOST_DEVICE_ID) {
        ret = offloadStreamManager_.ClearSlaves(tag);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[WorkspaceResourceImpl][DestroyWorkspaceResource]Destroy workspace stream failed. "
                "ret[%d]", ret);
        }
    }
}

// 销毁 Workspace全局资源，包含 Stream 资源 和 内存 资源
void WorkspaceResourceImpl::DestroyWorkspaceResource()
{
    // 销毁 work space memory 资源
    workSpaceMem_.DestroyMemResource();

    // 销毁 work space stream资源
    // 疑问：为什么有这个判断？
    if (static_cast<s32>(devicePhyId_) != HOST_DEVICE_ID) {
        offloadStreamManager_.ClearSlaves();
    }
    return;
}

// 基于tag 分配 Stream 资源
std::vector<Stream> WorkspaceResourceImpl::AllocSlaveStreams(const std::string &tag, u32 num)
{
    return offloadStreamManager_.GetSlaves(tag, num);
}

// 基于tag 销毁 Stream 资源
HcclResult WorkspaceResourceImpl::DestroyStream(const std::string &tag)
{
    return offloadStreamManager_.ClearSlaves(tag);
}

// 基于tag 分配 DeviceMem 资源
DeviceMem WorkspaceResourceImpl::AllocDeviceMem(const std::string &tag, u64 size)
{
    return DeviceMem::create(workSpaceMem_.AllocMem(tag, size), size);
}

// 基于tag 销毁 DeviceMem 资源
HcclResult WorkspaceResourceImpl::DestroyDeviceMem(const std::string &tag)
{
    return workSpaceMem_.DestroyMemResource(tag);
}


bool WorkspaceResourceImpl::IsExistResourceWorkSpaceMem(const std::string &tag)
{
    return workSpaceMem_.IsExist(tag);
}

// 这个接口为什么叫这个名字，以及为什么作这个判断，没梳理清楚
HcclResult WorkspaceResourceImpl::GetDevMemSize(const std::string &tag)
{
    auto interIter = opBaseDeviceMemMap_.find(tag);
    if (interIter == opBaseDeviceMemMap_.end()) {
        HCCL_INFO("[WorkspaceResourceImpl][GetDevMemSize]tag[%s] is exit, don't need get memsize", tag.c_str());
        return HCCL_SUCCESS;
    } else {
        return (interIter->second.size() == HCCL_WORKSPACE_MEM_32_KB) ? HCCL_SUCCESS : HCCL_E_INTERNAL;
    }
}

HcclResult WorkspaceResourceImpl::InsertDevMem(const std::string &tag, DeviceMem &deviceMem)
{
    opBaseDeviceMemMap_.erase(tag);
    opBaseDeviceMemMap_.insert(std::pair<std::string, DeviceMem>(tag, std::move(deviceMem)));
    return HCCL_SUCCESS;
}

HcclResult WorkspaceResourceImpl::DestroyRemoteOpBasedMem(const std::string &tag)
{
    std::unique_lock<std::mutex> lock(memResMutex_);
    opBaseDeviceMemMap_.erase(tag);
    remoteOpStreamMap_.erase(tag);
    lock.unlock();

    DestroyWorkspaceResource(tag);

    return HCCL_SUCCESS;
}

// 获取算子所需workspace memory大小[byte]
HcclResult WorkspaceResourceImpl::GetOpBasedMemSize(const HcclCMDType &opType, u64 &size,
    const HcomCollOpInfo &opInfo)
{
    u64 opMemSize = 0;

    if (opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
        // ReduceScatter 算子所需memory大小为 CCLBuffSize
        DevType devType;
        CHK_RET(hrtGetDeviceType(devType));
        if (IsSupportSDMAReduce(opInfo.inputAddr, opInfo.outputAddr, opInfo.dataType, opInfo.reduceOp) &&
            IsSupportRDMAReduce(opInfo.dataType, opInfo.reduceOp) && devType == DevType::DEV_TYPE_910B) {
                opMemSize = 0;
            } else {
                if (cclBufferManagerPtr_ == nullptr) {
                    opMemSize = GetExternalInputCCLBuffSize();
                } else {
                    opMemSize = cclBufferManagerPtr_->GetInCCLbufferSize();
                }
            }
    } else {
        opMemSize = 0;
    }
    size = HCCL_WORKSPACE_MEM_32_KB + opMemSize;
    HCCL_INFO("[WorkspaceResourceImpl][GetOpBasedMemSize]workspace memory size: op[%d], memory size[%llu].",
        opType, size);
    return HCCL_SUCCESS;
}

HcclResult WorkspaceResourceImpl::CreateOpBasedResources(const HcclCMDType &opType, const std::string &tag,
    const HcomCollOpInfo &opInfo)
{
    if (IsExistResourceWorkSpaceMem(tag) && GetDevMemSize(tag) != HCCL_SUCCESS) {
        HCCL_INFO("[WorkspaceResourceImpl][CreateOpBasedResources]tag[%s] is exit, don't creat workspace Memory",
            tag.c_str());
        return HCCL_SUCCESS;
    }
    // HCCL内部申请设备内存
    u64 memSize = 0;
    CHK_RET(GetOpBasedMemSize(opType, memSize, opInfo));

    // 创建 device memory
    DeviceMem deviceMem = DeviceMem::alloc(memSize);
    CHK_PRT_RET(!deviceMem, HCCL_ERROR("[WorkspaceResourceImpl][CreateOpBasedResources]In create workspace mem,"
        " malloc failed."), HCCL_E_MEMORY);
    std::vector<rtStream_t> stream;
    u64 maxSize = deviceMem.size();
    CHK_RET(SetWorkspaceResource(tag, deviceMem.ptr(), maxSize, stream));
    HCCL_INFO("[WorkspaceResourceImpl][CreateOpBasedResources]create workspace memory success. "
        "tag[%s] workspace addr[%p] workspace size[%llu].", tag.c_str(), deviceMem.ptr(), deviceMem.size());
    CHK_RET(InsertDevMem(tag, deviceMem));
    return HCCL_SUCCESS;
}

HcclResult WorkspaceResourceImpl::InsertRemoteOpStream(const std::string &tag, std::vector<Stream> &stream)
{
    auto interIter = remoteOpStreamMap_.find(tag);
    CHK_PRT_RET(interIter != remoteOpStreamMap_.end(),
        HCCL_ERROR("[WorkspaceResourceImpl][InsertRemoteOpStream]tag[%s] is exit, "
            "don't insert remote operation stream", tag.c_str()), HCCL_E_INTERNAL);
    remoteOpStreamMap_[tag] = std::move(stream);
    return HCCL_SUCCESS;
}

HcclResult WorkspaceResourceImpl::CreateAndInsertDevMem(const std::string &tag, u64 memSize,
    std::vector<rtStream_t> &streamPtr)
{
    // 创建device memory
    DeviceMem deviceMem = DeviceMem::alloc(memSize);
    CHK_PRT_RET(!deviceMem, HCCL_ERROR("[WorkspaceResourceImpl][CreateAndInsertDevMem]"
        "In create workspace mem malloc failed."), HCCL_E_MEMORY);

    CHK_RET(SetWorkspaceResource(tag, deviceMem.ptr(), memSize, streamPtr));
        HCCL_INFO("[WorkspaceResourceImpl][CreateAndInsertDevMem]create workspace memory success. "
            "tag[%s] workspace addr[%p] workspace size[%llu]", tag.c_str(), deviceMem.ptr(), deviceMem.size());

    // 资源管理
    CHK_RET(InsertDevMem(tag, deviceMem));
    return HCCL_SUCCESS;
}

HcclResult WorkspaceResourceImpl::CreateAndInsertRemoteOpStream(const std::string &tag,
    std::vector<rtStream_t> &streamPtr)
{
    // 创建三条流
    Stream stream1(StreamType::STREAM_TYPE_ONLINE);
    CHK_PRT_RET(stream1.ptr() == nullptr, HCCL_ERROR("[WorkspaceResourceImpl][CreateAndInsertRemoteOpStream]"
        "In create workspace stream 1,malloc failed."), HCCL_E_MEMORY);
    Stream stream2(StreamType::STREAM_TYPE_ONLINE);
    CHK_PRT_RET(stream2.ptr() == nullptr, HCCL_ERROR("[WorkspaceResourceImpl][CreateAndInsertRemoteOpStream]"
        "In create workspace stream 2,malloc failed."), HCCL_E_MEMORY);
    Stream stream3(StreamType::STREAM_TYPE_ONLINE);
    CHK_PRT_RET(stream3.ptr() == nullptr, HCCL_ERROR("[WorkspaceResourceImpl][CreateAndInsertRemoteOpStream]"
        "In create workspace stream 3,malloc failed."), HCCL_E_MEMORY);

    // 将流指针插入streamPtr
    streamPtr.push_back(stream1.ptr());
    streamPtr.push_back(stream2.ptr());
    streamPtr.push_back(stream3.ptr());

    // 管理流对象
    std::vector<Stream> streamObjs;
    streamObjs.reserve(HCCL_SUB_STREAM_NUM_THREE);
    streamObjs.push_back(std::move(stream1));
    streamObjs.push_back(std::move(stream2));
    streamObjs.push_back(std::move(stream3));

    CHK_RET(InsertRemoteOpStream(tag, streamObjs)); // 当前无实际作用
    return HCCL_SUCCESS;
}

HcclResult WorkspaceResourceImpl::CreateRemoteOpBasedResources(u64 memSize, const std::string &tag)
{
    if (IsExistResourceWorkSpaceMem(tag)) {
        HCCL_INFO("[WorkspaceResourceImpl][CreateRemoteOpBasedResources]tag[%s] is exit, "
            "don't create workspace Memory", tag.c_str());
        return HCCL_SUCCESS;
    }

    std::vector<rtStream_t> streamPtr;
    CHK_RET(CreateAndInsertRemoteOpStream(tag, streamPtr));
    CHK_RET(CreateAndInsertDevMem(tag, memSize, streamPtr));

    return HCCL_SUCCESS;
}

HcclResult WorkspaceResourceImpl::CreateOrUpdateRemoteOpBasedResources(u64 memSize, const std::string &tag)
{
    if (IsExistResourceWorkSpaceMem(tag)) {
        auto interMemIter = workSpaceMem_.memResMap_.find(tag);
        if (interMemIter->second.maxSize >= memSize) {
            HCCL_INFO("[WorkspaceResourceImpl][CreateOrUpdateRemoteOpBasedResources]tag[%s] is exit, "
                "and memSize meets the requirements, don't create workspace Memory", tag.c_str());
            return HCCL_SUCCESS;
        }
    }

    // workspace stream 资源已存在, 将流指针插入stream ptr
    std::vector<rtStream_t> streamPtr;
    auto interStreamIter = remoteOpStreamMap_.find(tag);
    if (interStreamIter != remoteOpStreamMap_.end()) {
        // 复用workspace stream 资源
        for (auto stream : interStreamIter->second) {
            streamPtr.push_back(stream.ptr());
        }
    } else {
        CHK_RET(CreateAndInsertRemoteOpStream(tag, streamPtr));
    }

    CHK_RET(CreateAndInsertDevMem(tag, memSize, streamPtr));
    return HCCL_SUCCESS;
}

}  // namespace hccl
