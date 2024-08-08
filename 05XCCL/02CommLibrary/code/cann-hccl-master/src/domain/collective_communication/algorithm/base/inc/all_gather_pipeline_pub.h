/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALL_GATHER_PIPELINE_PUB_H
#define ALL_GATHER_PIPELINE_PUB_H

#include <vector>
#include <memory>
#include <list>
#include <hccl/hccl_types.h>
#include "hccl/base.h"
#include "comm_base_pub.h"
#include "externalinput_pub.h"
#include "mem_device_pub.h"
#include "stream_pub.h"
#include "executor_base_pub.h"
#include "coll_alg_param.h"

namespace hccl {

class AllGatherPipeline : public ExecutorBase {
public:
    explicit AllGatherPipeline(const HcclDispatcher dispatcher);
    ~AllGatherPipeline() override;

    HcclResult Prepare(HcomCollOpInfo *opInfo, u32 userRank, u64 &count, DeviceMem &cclBufferPartOne,
        DeviceMem &cclBufferPartTwo, std::unique_ptr<CommBase> &commOuter,
        std::unique_ptr<CommBase> &commInner, Stream &mainStream, std::vector<Stream> &subStream,
        std::vector<std::shared_ptr<LocalNotify>> &notifyMain, std::vector<std::shared_ptr<LocalNotify>> &notifySub);

    // 适配新CollExecutor接口
    HcclResult Prepare(HcomCollOpInfo *opInfo, u32 userRank, u64 &count, DeviceMem &cclBufferPartOne,
        DeviceMem &cclBufferPartTwo, SubCommInfo &outerCommInfo, SubCommInfo &innerCommInfo,
        Stream &mainStream, std::vector<Stream> &subStream,
        std::vector<std::shared_ptr<LocalNotify>> &notifyMain, std::vector<std::shared_ptr<LocalNotify>> &notifySub);

    HcclResult RunAsync();

protected:

private:
    HcclResult MainRecordSub();
    HcclResult SubWaitMain();
    HcclResult MainWaitSub();
    HcclResult SubRecordMain();

    HcomCollOpInfo *opInfo_;
    u64 memSliceCount_;
    u32 userRank_;

    void* usrInMemAddr_;
    void* usrOutMemAddr_;
    std::vector<DeviceMem> dmaMem_;

    std::vector<Stream> subStream_;

    std::vector<std::shared_ptr<LocalNotify>> streamNotifyMain_;
    std::vector<std::shared_ptr<LocalNotify>> streamNotifySub_;

    u32 intraRankSize_;
    u32 interRankSize_;
    u32 intraRankId_;
    u32 interRankId_;

    std::vector<LINK> intraLinks_;
    std::vector<LINK> interLinks_;
};
}  // namespace hccl

#endif /* ALL_GATHER_PIPELINE_PUB_H */