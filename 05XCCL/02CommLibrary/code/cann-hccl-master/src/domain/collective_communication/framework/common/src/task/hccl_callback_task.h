/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef HCCL_CALLBACK_TASK_H
#define HCCL_CALLBACK_TASK_H

#include "base.h"
#include "dispatcher.h"

namespace hccl {
class HcclCallbackTask {
public:
    HcclCallbackTask(u32 devicePhyId, u32 deviceLogicId,
        HcclDispatcher dispatcher, NICDeployment nicDeployment);
    ~HcclCallbackTask();
    HcclResult CallbackRegStream(rtStream_t stream);
private:
    void CallbackThread();
    HcclResult CloseCallbackThread();
    u32 devicePhyId_;
    s32 deviceLogicId_;
    HcclDispatcher dispatcher_;
    NICDeployment nicDeployment_;
    std::unique_ptr<std::thread> callbackThread_;
    u64 callbackThreadId_;
    bool callbackThreadShutDown_;
};
} // namespace hccl
#endif // HCCL_CALLBACK_TASK_H