/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_NATIVE_EXECUTOR_BASE_H
#define COLL_NATIVE_EXECUTOR_BASE_H

#include "coll_executor_base.h"
#include "device_capacity.h"
#include "dispatcher.h"
#include "stream_active_manager.h"
#include "comm_factory_pub.h"
#include "rank_consistent.h"

namespace hccl {

struct ExecMem {
    u64 count = 0;
    DeviceMem inputMem;         /* 单算子模式时是InCCLMem, 图模式时是InUserMem */
    DeviceMem outputMem;        /* 单算子模式时是OutCCLMem, 图模式时是OutUserMem */
    DeviceMem scratchMem;
    void *inputPtr = nullptr;   /* InUserMem的地址，图模式时与inputMem的地址相同 */
    void *outputPtr = nullptr;  /* OutUserMem的地址，图模式时与outputMem的地址相同 */
};

class CollNativeExecutorBase : public CollExecutorBase {
public:
    CollNativeExecutorBase(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher);
    ~CollNativeExecutorBase() = default;

    HcclResult CalcResRequest(const OpParam& param, AlgResourceRequest &resourceRequest) override;
    bool CheckNeedRecreateComm(u64 lastScratchMemSize) override;

protected:
    /* *************** 资源计算 *************** */
    virtual void ParseParam(const OpParam& param);
    virtual HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport);
    virtual HcclResult CalcLevel0CommInfo(TransportMemType inputType,
        TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport);
    virtual HcclResult CalcLevel1CommInfo(TransportMemType inputType,
        TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport); // 默认情况下可根据algType_支持NHR、NHRV1、NB、HD、Ring算法。
    virtual HcclResult CalcLevel2CommInfo(TransportMemType inputType,
        TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport);
    virtual HcclResult CalcStreamNum(u32& streamNum);
    virtual HcclResult CalcScratchMemSize(u64& scratchMemSize);
    HcclResult CalcNotifyNum(u32 streamNum, u32 &notifyNum);
    virtual HcclResult GetIfNeedAivBuffer(bool &needAivBuffer);

    // 考虑新建一个资源计算类ResourceCalculator，将资源推导、资源解析的都放进去。
    // 推导通信域信息的公用函数，不同Executor的在计算inner、outer、level2时使用。
    HcclResult CalcCommPlaneInfo(const std::string &tag, const CommParaInfo &commParaInfo,
        std::vector<SingleSubCommTransport> &commTransport, TransportMemType inPutMemType,
        TransportMemType outPutMemType);
    HcclResult BuildResourceRequest(u64 scratchMemSize, u32 streamNum, u32 notifyNum, bool needAivBuffer,
        std::vector<LevelNSubCommTransport>& opTransport, AlgResourceRequest& resourceRequest);
    HcclResult PrintTransportRequest(AlgResourceRequest& resourceRequest);
    /* *************** 算法编排 *************** */
    // 虚函数，执行具体的数据搬运、reduce操作。  各Executor重载。
    // 按Inner、Outer、Level2可继续进行拆分。
    virtual HcclResult KernelRun(const OpParam &param, ExecMem &execMem);

    // 算法编排资源获取
    HcclResult GetStreamInfo(const AlgResourceResponse &algRes);
    // 图模式下激活从流
    HcclResult ActiveSlaveStreams(const Stream &stream);
    // 将从流添加至Profiling
    HcclResult AddSubStreamToProfiling();
    // 检查通信域大小
    HcclResult CheckCommSize(const CommPlane levelIndex, const u32 subLevelIndex);

    // 获取不同类型通信域中的 transport 信息
    // 为了避免循环调用时反复校验Range引发性能问题，此处不做Range校验，建议调用该接口前先调用CheckCommSize避免OutOfRange问题
    SubCommInfo GetSubCommInfo(const CommPlane levelIndex, const u32 subLevelIndex);

    // 算法类型工具类
    AlgTypeLevel0 GetLevel0AlgType(const AlgType algType) const;
    AlgTypeLevel1 GetLevel1AlgType(const AlgType algType) const;
    AlgTypeLevel2 GetLevel2AlgType(const AlgType algType) const;

    bool UseInterServerRingAlgo(AlgType algType);
    bool UseInterServerHDAlgo(AlgType algType);
    bool UseInterServerNHRAlgo(AlgType algType);
    bool UseInterServerNHRV1Algo(AlgType algType);
    bool UseInterServerNBAlgo(AlgType algType);
    bool UseLevel2RingAlgo(AlgType algType);
    bool UseInterServerPipelineAlgo(AlgType algType);
    HcclResult GetRankByUserRank(CommPlane levelIndex, u32 subLevelIndex, u32 userRank, u32 &rank);
    HcclResult GetUserRankByRank(CommPlane levelIndex, u32 subLevelIndex, u32 rank, u32 &userRank);

    /* ---------------以下为 protected 成员变量定义领域-------------------------- */
    std::string tag_;
    u32 root_ = INVALID_VALUE_RANKID;
    const AlgResourceResponse *algResResp_ = nullptr;
    innerStreamInfo_t streamInfo_;

    // Infos got from topoMatcher_
    const HcclTopoInfo topoAttr_;
    const HcclAlgoInfo algoAttr_;
    TopoType topoType_;
    bool is310P3Common_ = false;
};
}
#endif
