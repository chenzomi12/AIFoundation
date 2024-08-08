/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUEUE_NOTIFY_MANAGER_H
#define QUEUE_NOTIFY_MANAGER_H

#include <mutex>
#include <map>
#include "base.h"
#include "local_notify.h"
#include "dispatcher.h"

namespace hccl {


class QueueNotifyManager {
using NotifyPoolNoIPC = std::vector<std::shared_ptr<LocalNotify>>;
public:
    QueueNotifyManager();

    ~QueueNotifyManager();

    HcclResult Init();

    HcclResult Alloc(const std::string &tag, u32 notifyNum, std::vector<std::shared_ptr<LocalNotify>> &localNotifys,
        const NotifyLoadType type = NotifyLoadType::HOST_NOTIFY);

    HcclResult Destroy();

    QueueNotifyManager(QueueNotifyManager const&) = delete;                 // Copy construct
    QueueNotifyManager(QueueNotifyManager&&) = delete;                      // Move construct
    QueueNotifyManager& operator=(QueueNotifyManager const&) = delete;      // Copy assign
    QueueNotifyManager& operator=(QueueNotifyManager &&) = delete;          // Move assign

private:
    HcclResult AllocNotifies(const NotifyLoadType type, NotifyPoolNoIPC &notifies, u32 notifyNum);
    HcclResult CreateNotify(std::shared_ptr<LocalNotify> &localNotify, const NotifyLoadType type);
    HcclResult DestroyNotifies(NotifyPoolNoIPC &notifies);

    NotifyPoolNoIPC notifies_;
    NotifyPoolNoIPC deviceNotifies_;
    NotifyPoolNoIPC notifiesForA2A_;
};
}  // namespace hccl
#endif /* QUEUE_NOTIFY_MANAGER_H */