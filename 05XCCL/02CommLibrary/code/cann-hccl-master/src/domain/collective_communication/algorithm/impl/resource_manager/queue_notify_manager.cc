/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "queue_notify_manager.h"
#include <algorithm>
#include "device_capacity.h"

namespace hccl {
constexpr u32 NOTIFY_MAX_NUM = 2048;
const std::string HCCL_ALLTOALL = "ALLTOALL";
QueueNotifyManager::QueueNotifyManager()
{
}

QueueNotifyManager::~QueueNotifyManager()
{
    HcclResult ret = Destroy();
    if (ret != HCCL_SUCCESS) {
        HCCL_WARNING("destroy QueueNotifyManager resources failed, ret[%d]", ret);
    }
}

HcclResult QueueNotifyManager::Init()
{
    notifies_.reserve(NOTIFY_MAX_NUM);
    notifiesForA2A_.reserve(NOTIFY_MAX_NUM);
    deviceNotifies_.reserve(NOTIFY_MAX_NUM);
    return HCCL_SUCCESS;
}

HcclResult QueueNotifyManager::Alloc(const std::string &tag, u32 notifyNum,
    std::vector<std::shared_ptr<LocalNotify>> &localNotifys, const NotifyLoadType type)
{
    if (type == NotifyLoadType::HOST_NOTIFY) {
        std::string upTag = tag;
        std::transform(upTag.begin(), upTag.end(), upTag.begin(), ::toupper);
        bool hasAlltoAll = upTag.find(HCCL_ALLTOALL) != std::string::npos;
        HCCL_INFO("RegisterOp hasAlltoAll[%d]", hasAlltoAll);
        NotifyPoolNoIPC &notifies = hasAlltoAll ? notifiesForA2A_ : notifies_;
        CHK_RET(AllocNotifies(type, notifies, notifyNum));
        localNotifys.assign(notifies.begin(), notifies.begin() + notifyNum);
    } else if (type == NotifyLoadType::DEVICE_NOTIFY) { // 申请device上使用的notify资源
        CHK_RET(AllocNotifies(type, deviceNotifies_, notifyNum));
        localNotifys.assign(deviceNotifies_.begin(),
            deviceNotifies_.begin() + notifyNum);
    }
    return HCCL_SUCCESS;
}

HcclResult QueueNotifyManager::AllocNotifies(const NotifyLoadType type, NotifyPoolNoIPC &notifies, u32 notifyNum)
{
    if (notifies.size() < notifyNum) {
        for (u32 i = notifies.size(); i < notifyNum; i++) {
            notifies.emplace_back(nullptr);
            CHK_RET(CreateNotify(notifies[i], type));
            CHK_SMART_PTR_NULL(notifies[i]);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult QueueNotifyManager::CreateNotify(std::shared_ptr<LocalNotify> &localNotify, const NotifyLoadType type)
{
    EXECEPTION_CATCH((localNotify = std::make_shared<LocalNotify>()), return HCCL_E_PTR);
    CHK_SMART_PTR_NULL(localNotify);

    HcclResult ret = HCCL_SUCCESS;
    bool errorFlag = false;
    do {
        ret = localNotify->Init(type);
        CHK_PRT_BREAK(ret != HCCL_SUCCESS, HCCL_ERROR("[QueueNotifyManager][CreateNotify]localNotify init failed, "
            "ret[%d]", ret), errorFlag = true);

        HCCL_DEBUG("Is310PDevice[%d]", Is310PDevice());
        if (Is310PDevice()) {
            ret = localNotify->SetIpc();
            CHK_PRT_BREAK(ret != HCCL_SUCCESS, HCCL_ERROR("[QueueNotifyManager][CreateNotify]localNotify "
                "set ipc failed, ret[%d]", ret), errorFlag = true);
        }
    } while (0);

    if (errorFlag) {
        HCCL_ERROR("[QueueNotifyManager][CreateNotify]localNotify create failed ,ret[%d]", ret);
        localNotify = nullptr;
        return ret;
    }

    return HCCL_SUCCESS;
}

HcclResult QueueNotifyManager::Destroy()
{
    HCCL_INFO("QueueNotifyManager Destroy.");

    CHK_RET(DestroyNotifies(notifies_));
    CHK_RET(DestroyNotifies(deviceNotifies_));
    CHK_RET(DestroyNotifies(notifiesForA2A_));

    HCCL_INFO("QueueNotifyManager Destroy success.");
    return HCCL_SUCCESS;
}

HcclResult QueueNotifyManager::DestroyNotifies(NotifyPoolNoIPC &notifies)
{
    for (auto &notify : notifies) {
        CHK_SMART_PTR_NULL(notify);
        CHK_RET(notify->Destroy());
    }
    notifies.clear();
    return HCCL_SUCCESS;
}

}  // namespace hccl
