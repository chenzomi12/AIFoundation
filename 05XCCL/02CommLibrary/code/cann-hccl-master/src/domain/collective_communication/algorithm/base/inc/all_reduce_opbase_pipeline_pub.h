/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALL_REDUCE_OPBASE_PIPELINE_PUB_H
#define ALL_REDUCE_OPBASE_PIPELINE_PUB_H

#include <vector>
#include <memory>
#include <hccl/hccl_types.h>
#include "hccl/base.h"
#include "comm_base_pub.h"
#include "externalinput_pub.h"
#include "mem_device_pub.h"
#include "stream_pub.h"
#include "dispatcher.h"
#include "executor_base_pub.h"
#include "reducer_pub.h"
#include "sender_pub.h"
#include "coll_alg_param.h"

namespace hccl {

class AllReduceOpbasePipeline : public ExecutorBase {
public:
    explicit AllReduceOpbasePipeline (const HcclDispatcher dispatcher, const u64 reduceAttrBitMap);
    ~AllReduceOpbasePipeline() override;

    HcclResult Prepare(const HcomCollOpInfo *opInfo,
                       DeviceMem &cclBufferA,
                       DeviceMem &cclBufferB,
                       const u64 count,
                       const SubCommInfo &innerCommInfo,
                       const SubCommInfo &outerCommInfo,
                       Stream &mainStream,
                       std::vector<Stream> &subStream,
                       std::vector<std::shared_ptr<LocalNotify>> &notifyMain,
                       std::vector<std::shared_ptr<LocalNotify>> &notifySub);

    HcclResult RunAsync();

protected:

private:
    HcclResult RunReduceScatterIntraServer(u32 step);
    HcclResult RunReduceScatterInterServer(u32 step, const LINK &prevInterLink, const LINK &nextInterLink);
    HcclResult RunAllGatherIntraServer(u32 step);
    HcclResult RunAllGatherInterServer(u32 step, const LINK &prevInterLink, const LINK &nextInterLink);
    HcclResult CopyToScratchBuffer(u32 step);
    HcclResult MainWaitSub();
    HcclResult SubRecordMain();
    HcclResult MainRecordSub();
    HcclResult SubWaitMain();

    HcomCollOpInfo *opInfo_;

    void* usrInMem_;
    void* usrOutMem_;
    u64 count_;
    u32 unitSize_;
    u64 curSize_;
    u64 sliceCount_;
    u64 memSliceSize_;
    u64 lastSliceCount_;
    u64 lastSliceSize_;

    HcclReduceOp reductionOp_;
    HcclDataType dataType_;

    DeviceMem cclBufferA_;
    DeviceMem cclBufferB_;
    std::vector<DeviceMem> dmaMem_;

    std::vector<Stream> subStreams_;

    std::vector<std::shared_ptr<LocalNotify>> streamNotifyMain_;
    std::vector<std::shared_ptr<LocalNotify>> streamNotifySub_;

    std::unique_ptr<Sender> senderInfo_;
    std::unique_ptr<Reducer> reducerInfo_;
    const u64 reduceAttr_; /* 0x1:表示data_type + reduce_type支持inlinereduce  */
    u32 intraRankSize_;
    u32 interRankSize_;
    u32 intraRankId_;
    u32 interRankId_;
    u32 rankId_;

    std::vector<LINK> intraLinks_;
    std::vector<LINK> interLinks_;
};
}  // namespace hccl

#endif /* ALL_REDUCE_OPBASE_PIPELINE_PUB_H */
