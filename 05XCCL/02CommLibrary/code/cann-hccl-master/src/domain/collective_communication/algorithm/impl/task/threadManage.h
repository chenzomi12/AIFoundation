/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef THREAD_MANAGE_H
#define THREAD_MANAGE_H

#include <condition_variable>
#include "threadManage.h"
#include "dispatcher.h"
#include "comm_base_pub.h"
#include "externalinput_pub.h"
#include "coll_alg_param.h"

namespace hccl {
enum class ExecutorType {
    REDUCE_SCATTER_RING,
    ALLGATHER_RING,
    REDUCE_SCATTER_RING_DIRECT,
    ALLGATHER_RING_DIRECT,
    TYPE_RESERVED
};

class ThreadManage {
public:
    explicit ThreadManage(s32 deviceLogicId, u32 userRank, const HcclDispatcher dispatcher);

    ~ThreadManage();

    HcclResult Init();
    HcclResult Prepare(DeviceMem &inputMem, DeviceMem &outputMem, DeviceMem &scratchMem, const u64 count,
                       const HcclDataType dataType, const Stream &stream, const HcclReduceOp reductionOp,
                       const u32 root, const std::vector<Slice> &slices, const u64 baseOffset,
                       std::vector<u32> nicRankList, const std::string &tag,
                       s32 profStage, void* commRing, std::shared_ptr<LocalNotify> &signalAux,
                       std::shared_ptr<LocalNotify> &signalMain, u32 ringIndex,
                       ExecutorType type, u64 reduceAttr = 0, const HcomCollOpInfo *opInfo = nullptr,
                       std::vector<Stream> subStreamsInOneRing = {},
                       std::vector<std::shared_ptr<LocalNotify>> mainSignalsInOneRing = {},
                       std::vector<std::shared_ptr<LocalNotify>> subSignalsInOneRing = {},
                       std::vector<u32> ringsOrder = {},
                       std::vector<Slice> userMemInputSlices = {});
    HcclResult Prepare(DeviceMem &inputMem, DeviceMem &outputMem, DeviceMem &scratchMem, const u64 count,
                       const HcclDataType dataType, const Stream &stream, const HcclReduceOp reductionOp,
                       const u32 root, const std::vector<Slice> &slices, const u64 baseOffset,
                       std::vector<u32> nicRankList, const std::string &tag,
                       s32 profStage, const SubCommInfo &ringSubCommInfo, std::shared_ptr<LocalNotify> &signalAux,
                       std::shared_ptr<LocalNotify> &signalMain, u32 ringIndex,
                       ExecutorType type, u64 reduceAttr = 0, const HcomCollOpInfo *opInfo = nullptr,
                       std::vector<Stream> subStreamsInOneRing = {},
                       std::vector<std::shared_ptr<LocalNotify>> mainSignalsInOneRing = {},
                       std::vector<std::shared_ptr<LocalNotify>> subSignalsInOneRing = {},
                       std::vector<u32> ringsOrder = {},
                       std::vector<Slice> userMemInputSlices = {});

    HcclResult Finalize();
    void NotifyStart();
    void WaitStart();
    void NotifyDone();
    void WaitDone();
    uint32_t GetTid();
protected:

private:
    HcclResult ThreadExecuteFn();
    HcclResult ExecuteService();
    std::shared_ptr<std::thread>ringThread_;
    uint32_t threadId_ = 0;
    s32 deviceLogicId_;
    u32 userRank_;
    const HcclDispatcher dispatcher_;

    std::mutex startMtx_;
    std::mutex doneMtx_;
    std::condition_variable startCv_;
    std::condition_variable doneCv_;
    bool startReady = false;
    bool doneReady = false;
    bool threadExit = false;

    DeviceMem inputMem_;
    DeviceMem outputMem_;
    DeviceMem scratchMem_;
    Stream stream_;

    u64 count_ = 0;
    HcclDataType dataType_ = HCCL_DATA_TYPE_RESERVED;
    HcclReduceOp reductionOp_ = HCCL_REDUCE_RESERVED;
    u32 root_ = 0;
    std::vector<Slice> slices_;
    u64 baseOffset_ = 0;
    std::vector<u32> nicRankList_;
    std::string tag_;
    s32 profStage_ = 0;
    bool newExecutorFlag_{false};
    SubCommInfo ringSubCommInfo_;
    void* commRing_ = nullptr;
    std::shared_ptr<LocalNotify> signalAux_ = nullptr;
    std::shared_ptr<LocalNotify> signalMain_ = nullptr;
    u32 ringIndex_  = 0;
    u64 reduceAttr_ = 0;
    const HcomCollOpInfo *opInfo_;
    std::vector<Stream> subStreamsInOneRing_;
    std::vector<std::shared_ptr<LocalNotify>> mainSignalsInOneRing_;
    std::vector<std::shared_ptr<LocalNotify>> subSignalsInOneRing_;
    std::vector<u32> ringsOrder_;
    std::vector<Slice> userMemInputSlices_;
    ExecutorType executorType_ = ExecutorType::TYPE_RESERVED;
    HcclRtContext context_;
};
}  // namespace hccl

#endif /* * THREAD_MANAGE_H */
