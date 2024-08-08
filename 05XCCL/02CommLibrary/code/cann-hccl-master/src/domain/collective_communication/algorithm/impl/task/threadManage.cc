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
#include "executor_base_pub.h"
#include "hccl_impl_pub.h"
#include "reduce_scatter_ring_pub.h"
#include "reduce_scatter_ring_concurrent_direct_pub.h"
#include "all_gather_ring_pub.h"
#include "all_gather_ring_concurrent_direct_pub.h"
#include "threadManage.h"
#include "coll_executor_base.h"

namespace hccl {
ThreadManage::ThreadManage(s32 deviceLogicId, u32 userRank, const HcclDispatcher dispatcher)
    : deviceLogicId_(deviceLogicId), userRank_(userRank), dispatcher_(dispatcher), context_(nullptr)
{}
ThreadManage::~ThreadManage()
{
    HcclResult ret = Finalize();
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[ThreadManage][Destroy]threadManage Finalize failed[%d] ", ret);
    }
}

HcclResult ThreadManage::Init()
{
    HCCL_INFO("ThreadManage::Init");
    CHK_RET(hrtCtxGetCurrent(&context_));

    ringThread_.reset(new (std::nothrow) std::thread(&ThreadManage::ThreadExecuteFn, this));
    CHK_SMART_PTR_NULL(ringThread_);
    return HCCL_SUCCESS;
}

HcclResult ThreadManage::Finalize()
{
    if (ringThread_) {
        threadExit = true;
        NotifyStart();
        if (ringThread_->joinable()) {
            ringThread_->join();
        }
        ringThread_ = nullptr;
    }
    return HCCL_SUCCESS;
}

void ThreadManage::NotifyStart()
{
    std::unique_lock<std::mutex> lock(startMtx_);
    startReady = true; // 设置标志位为 true.
    startCv_.notify_one();
}

void ThreadManage::WaitStart()
{
    std::unique_lock<std::mutex> lock(startMtx_);
    while (!startReady) {    // 假设标志位不为 true, 则等待...
        startCv_.wait(lock); // 当前线程被堵塞, 当标志位变为 true 之后,
    }
    startReady = false;
}

void ThreadManage::NotifyDone()
{
    std::unique_lock<std::mutex> lock(doneMtx_);
    doneReady = true;
    doneCv_.notify_one();
}

void ThreadManage::WaitDone()
{
    std::unique_lock<std::mutex> lock(doneMtx_);
    while (!doneReady) {
        doneCv_.wait(lock);
    }
    doneReady = false;
}

HcclResult ThreadManage::ExecuteService()
{
    HcclResult ret = HCCL_SUCCESS;
    hccl::CommBase* commRing = nullptr;
    if (newExecutorFlag_ == false) {
        commRing = static_cast<hccl::CommBase *>(commRing_);
        CHK_SMART_PTR_NULL(commRing);
    }

    std::unique_ptr<ExecutorBase> executor;
    if (executorType_ == ExecutorType::REDUCE_SCATTER_RING) {
        executor.reset(new (std::nothrow) ReduceScatterRing(dispatcher_, reduceAttr_));
    } else if (executorType_ == ExecutorType::ALLGATHER_RING) {
        executor.reset(new (std::nothrow) AllGatherRing(dispatcher_));
    } else if (executorType_ == ExecutorType::REDUCE_SCATTER_RING_DIRECT) {
        executor.reset(new (std::nothrow) ReduceScatterRingConcurrentDirect(
            dispatcher_, reduceAttr_, opInfo_, userRank_, subStreamsInOneRing_, mainSignalsInOneRing_,
            subSignalsInOneRing_, ringsOrder_, userMemInputSlices_));
    } else if (executorType_ == ExecutorType::ALLGATHER_RING_DIRECT) {
        executor.reset(new (std::nothrow) AllGatherRingConcurrentDirect(
            dispatcher_, opInfo_, userRank_, subStreamsInOneRing_, mainSignalsInOneRing_,
            subSignalsInOneRing_, ringsOrder_, userMemInputSlices_));
    }
    CHK_SMART_PTR_NULL(executor);

    /* 从环等待启动 */
    ret = LocalNotify::Wait(stream_, dispatcher_, signalAux_, profStage_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Execute][Service]stream[%u] wait failed", ringIndex_), ret);

    ret = executor->Prepare(inputMem_, outputMem_, scratchMem_, count_, dataType_, stream_, reductionOp_,
        OUTER_BRIDGE_RANK_ID, slices_, baseOffset_, nicRankList_);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Execute][Service]stream[%u],prepare failed,return[%d]", ringIndex_, ret), ret);

    if (newExecutorFlag_) {
        ret = executor->RegisterProfiler(((ringIndex_ + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
            (ringSubCommInfo_.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + ringSubCommInfo_.localRank,
            profStage_, HCCL_EXEC_STEP_NOT_SET, stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Execute][Service]stream[%u],register Profiler failed,return[%d]", ringIndex_, ret), ret);

        ret = CollExecutorBase::RunTemplate(executor, ringSubCommInfo_);
    } else {
        ret = executor->RegisterProfiler(((ringIndex_ + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
            (commRing->RankSize() << PROF_RANKSIZE_OFFSET_OF_PLANEID) + commRing->Rank(),
            profStage_, HCCL_EXEC_STEP_NOT_SET, stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Execute][Service]stream[%u],register Profiler failed,return[%d]", ringIndex_, ret), ret);

        ret = commRing->RunExecutor(executor);
    }

    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Execute][Service]stream[%u],run failed,return[%d]", ringIndex_, ret), ret);
    /* 从环record通知主环结束 */
    ret = LocalNotify::Post(stream_, dispatcher_, signalMain_, profStage_);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Execute][Service]stream[%u] record failed", ringIndex_), ret);
    return HCCL_SUCCESS;
}


HcclResult ThreadManage::ThreadExecuteFn()
{
    threadId_ = SalGetTid();
    HCCL_INFO("[ThreadManage][ThreadExecuteFn]deviceLogicId_[%d], threadId_[%u]", deviceLogicId_, threadId_);

    CHK_RET(hrtSetDevice(deviceLogicId_));
    if (context_ != nullptr) {
        CHK_RET(hrtCtxSetCurrent(context_));
    }
    while (true) {
        WaitStart(); // 等待线程执行通知
        if (threadExit) {
            HCCL_INFO("threadExit deviceLogicId_[%d] ringIndex_[%u]", deviceLogicId_, ringIndex_);
            break;
        }
        HcclResult ret = ExecuteService();
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[ThreadManage][ThreadExecuteFn]ThreadManage run ExecuteService fail");
        }
        NotifyDone(); // 通知主进程本线程执行完成
    }
    CHK_RET(hrtResetDevice(deviceLogicId_));

    return HCCL_SUCCESS;
}

HcclResult ThreadManage::Prepare(DeviceMem &inputMem, DeviceMem &outputMem, DeviceMem &scratchMem, const u64 count,
    const HcclDataType dataType, const Stream &stream, const HcclReduceOp reductionOp, const u32 root,
    const std::vector<Slice> &slices, const u64 baseOffset, std::vector<u32> nicRankList,
    const std::string &tag, s32 profStage, void* commRing,
    std::shared_ptr<LocalNotify> &signalAux, std::shared_ptr<LocalNotify> &signalMain, u32 ringIndex,
    ExecutorType type, u64 reduceAttr, const HcomCollOpInfo *opInfo,
    std::vector<Stream> subStreamsInOneRing, std::vector<std::shared_ptr<LocalNotify>> mainSignalsInOneRing,
    std::vector<std::shared_ptr<LocalNotify>> subSignalsInOneRing, std::vector<u32> ringsOrder,
    std::vector<Slice> userMemInputSlices)
{
    newExecutorFlag_ = false;
    /* * 参数保存 */
    inputMem_ = inputMem;
    outputMem_ = outputMem;
    scratchMem_ = scratchMem;
    count_ = count;
    dataType_ = dataType;
    stream_ = stream;
    reductionOp_ = reductionOp;
    root_ = root;
    baseOffset_ = baseOffset;
    profStage_ = profStage;
    commRing_ = commRing;
    signalAux_ = signalAux;
    signalMain_ = signalMain;
    ringIndex_ = ringIndex;
    reduceAttr_ = reduceAttr;
    opInfo_ = opInfo;
    subStreamsInOneRing_ = subStreamsInOneRing;
    mainSignalsInOneRing_ = mainSignalsInOneRing;
    subSignalsInOneRing_ = subSignalsInOneRing;
    ringsOrder_ = ringsOrder;
    userMemInputSlices_ = userMemInputSlices;
    executorType_ = type;

    tag_.assign(tag.begin(), tag.end());
    slices_.assign(slices.begin(), slices.end());
    nicRankList_.assign(nicRankList.begin(), nicRankList.end());
    HCCL_PROFILER_ADD_STREAM(stream_.ptr(), tag_, 0, AlgType::ALG_RESERVED);
    return HCCL_SUCCESS;
}

HcclResult ThreadManage::Prepare(DeviceMem &inputMem, DeviceMem &outputMem, DeviceMem &scratchMem, const u64 count,
    const HcclDataType dataType, const Stream &stream, const HcclReduceOp reductionOp, const u32 root,
    const std::vector<Slice> &slices, const u64 baseOffset, std::vector<u32> nicRankList,
    const std::string &tag, s32 profStage, const SubCommInfo &ringSubCommInfo,
    std::shared_ptr<LocalNotify> &signalAux, std::shared_ptr<LocalNotify> &signalMain, u32 ringIndex,
    ExecutorType type, u64 reduceAttr, const HcomCollOpInfo *opInfo,
    std::vector<Stream> subStreamsInOneRing, std::vector<std::shared_ptr<LocalNotify>> mainSignalsInOneRing,
    std::vector<std::shared_ptr<LocalNotify>> subSignalsInOneRing, std::vector<u32> ringsOrder,
    std::vector<Slice> userMemInputSlices)
{
    newExecutorFlag_ = true;
    /* * 参数保存 */
    inputMem_ = inputMem;
    outputMem_ = outputMem;
    scratchMem_ = scratchMem;
    count_ = count;
    dataType_ = dataType;
    stream_ = stream;
    reductionOp_ = reductionOp;
    root_ = root;
    baseOffset_ = baseOffset;
    profStage_ = profStage;
    ringSubCommInfo_ = ringSubCommInfo;
    signalAux_ = signalAux;
    signalMain_ = signalMain;
    ringIndex_ = ringIndex;
    reduceAttr_ = reduceAttr;
    opInfo_ = opInfo;
    subStreamsInOneRing_ = subStreamsInOneRing;
    mainSignalsInOneRing_ = mainSignalsInOneRing;
    subSignalsInOneRing_ = subSignalsInOneRing;
    ringsOrder_ = ringsOrder;
    userMemInputSlices_ = userMemInputSlices;
    executorType_ = type;

    tag_.assign(tag.begin(), tag.end());
    slices_.assign(slices.begin(), slices.end());
    nicRankList_.assign(nicRankList.begin(), nicRankList.end());
    HCCL_PROFILER_ADD_STREAM(stream_.ptr(), tag_, 0, AlgType::ALG_RESERVED);
    return HCCL_SUCCESS;
}
uint32_t ThreadManage::GetTid()
{
    if (threadId_ == 0) {
        threadId_ = SalGetTid();
    }
    HCCL_INFO("[ThreadManage][GetTid]deviceLogicId_[%d], threadId_[%u]", deviceLogicId_, threadId_);
    return threadId_;
}

} // namespace hccl
