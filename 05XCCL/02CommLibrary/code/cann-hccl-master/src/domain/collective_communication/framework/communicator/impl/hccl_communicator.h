/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_COMMUNICATOR_H
#define HCCL_COMMUNICATOR_H

#include <atomic>
#include <memory>
#include <hccl/hccl_types.h>

#include "hccl/base.h"
#include "hccl_impl_pub.h"
#include "comm_factory_pub.h"
#include "heartbeat_pub.h"
#include "opexecounter_pub.h"
#include "op_base_stream_manager_pub.h"
#include "offload_stream_manager_pub.h"
#include "profiler_manager.h"

#include "topoinfo_parse.h"
#include "hccl_alg.h"
#include "ccl_buffer_manager.h"
#include "hccl_opbase_atrace_info_pub.h"
#include "hccl_callback_task.h"
#include "aicpu_operator_pub.h"
#include "mr_manager.h"
#include "transport_heterog_def.h"
#include "resource_manager/queue_notify_manager.h"
#include "hccl_network_pub.h"
#include "comm.h"
#include "device_capacity.h"
#include "transport_manager.h"
#include "coll_alg_operator.h"
#include "opretry_manage_pub.h"

namespace hccl {
using ServRankInfo_t = std::map<std::string, std::vector<RankInfo_t> >;

constexpr s32 COMM_MAX_DEVICE_ID = 31;
constexpr u32 COMM_MAX_WORK_SPACE_SIZE = 16 * 1024 * 1024; // 默认16MB
const std::string COMM_LOOPBACK_IP = "127.0.0.1";

class HcclCommunicator {
public:
    explicit HcclCommunicator();

    virtual ~HcclCommunicator();

    // 对外接口
    virtual HcclResult Init(HcclCommParams &params, const RankTable_t &rankTable);
    virtual HcclResult Init(HcclCommParams &params, const std::vector<RankInfo> &rankList,
        WorldGroupInfo &globalData);

    virtual HcclResult GetAlgType(AlgType &algType, HcclCMDType opType);

    virtual HcclResult GetDeviceNumPerAggregation(u32 &deviceNumPerAggregation);

    virtual HcclResult GetBandWidthPerNPU(u32 level, float &bandWidth);

    virtual HcclResult AllGather(const std::string &tag, void *inputPtr, void *outputPtr, u64 inputCount,
        HcclDataType dataType, HcclRtStream stream, HcomCollOpInfo *opInfo = nullptr);

    virtual HcclResult AllGatherOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 inputCount,
        HcclDataType dataType, HcclRtStream stream);

    virtual HcclResult AllReduce(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
        HcclDataType dataType, HcclReduceOp op, HcclRtStream stream,
        SyncMode syncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE, const HcomCollOpInfo *opInfo = nullptr);

    virtual HcclResult AllReduceOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
        HcclDataType dataType, HcclReduceOp op, HcclRtStream stream,
        SyncMode syncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE);

    virtual HcclResult AlltoAllV(const void *sendBuf, const void *sendCounts, const void *sdispls,
        HcclDataType sendType, const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType,
        rtStream_t stream, const std::string &tag);

    virtual HcclResult AlltoAllVOutPlace(const void *sendBuf, const void *sendCounts, const void *sdispls,
        HcclDataType sendType, const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType,
        rtStream_t stream, const std::string &tag);

    virtual HcclResult AlltoAllVC(const void *sendBuf, const void *sendCountMatrix, HcclDataType sendType,
        const void *recvBuf, HcclDataType recvType, rtStream_t stream, const std::string &tag);

    virtual HcclResult AlltoAllVCOutPlace(const void *sendBuf, const void *sendCountMatrix, HcclDataType sendType,
        const void *recvBuf, HcclDataType recvType, rtStream_t stream, const std::string &tag);

    virtual HcclResult AlltoAll(const void *sendBuf, u64 sendCount, HcclDataType sendType,
        const void *recvBuf, u64 recvCount, HcclDataType recvType, rtStream_t stream, const std::string &tag);

    virtual HcclResult Broadcast(const std::string &tag, void *ptr, u64 count, HcclDataType dataType, u32 root,
        HcclRtStream stream);

    virtual HcclResult BroadcastOutPlace(const std::string &tag, void *ptr, u64 count, HcclDataType dataType, u32 root,
        HcclRtStream stream);

    virtual HcclResult Scatter(const std::string &tag, void *inputPtr, void *outputPtr, u64 recvCount,
        HcclDataType dataType, u32 root, HcclRtStream stream);

    virtual HcclResult ScatterOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 recvCount,
        HcclDataType dataType, u32 root, HcclRtStream stream);

    virtual HcclResult Reduce(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
        HcclDataType dataType, HcclReduceOp op, u32 root, HcclRtStream stream);

    virtual HcclResult ReduceOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
        HcclDataType dataType, HcclReduceOp op, u32 root, HcclRtStream stream);

    virtual HcclResult ReduceScatter(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
        HcclDataType dataType, HcclReduceOp op, HcclRtStream stream, HcomCollOpInfo *opInfo = nullptr);

    virtual HcclResult ReduceScatterOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
        HcclDataType dataType, HcclReduceOp op, HcclRtStream stream);

    virtual HcclResult ProcessSendRecvTasks(const std::string &tag, std::vector<HcclSendRecvItem *> &orderedList,
        u32 itemNum, u32 startIndex, rtStream_t stream);

    virtual HcclResult Send(const std::string &tag, void *inputPtr, u64 count, HcclDataType dataType,
        u32 destRank, rtStream_t stream);

    virtual HcclResult SendOutPlace(const std::string &tag, void *inputPtr, u64 count, HcclDataType dataType,
        u32 destRank, rtStream_t stream);

    virtual HcclResult Receive(const std::string &tag, void *outputPtr, u64 count, HcclDataType dataType,
        u32 srcRank, rtStream_t stream);

    virtual HcclResult ReceiveOutPlace(const std::string &tag, void *outputPtr, u64 count, HcclDataType dataType,
        u32 srcRank, rtStream_t stream);

    virtual HcclResult Gather(const std::string &tag, void *inputPtr, void *outputPtr, u32 rootRank, u64 inputCount,
        HcclDataType dataType, rtStream_t stream);

    virtual HcclResult SupportDeterministicOptim(bool &isDeterministicOptim);

    virtual HcclResult GetCqeError(HcclResult &result);

    //  对内接口
    virtual HcclResult CheckDataType(const HcclDataType dataType, bool needReduce);

    virtual HcclResult CheckReduceDataType(const HcclDataType dataType, const HcclReduceOp op);

    virtual HcclResult ReleaseCommInfos();

    virtual std::vector<u64> GenerateSendCountMatrix(u64 count, u32 rankSize);

    virtual HcclResult GetAlltoAllStagedWorkSpaceMemSize(u64 *sendCounts, u64 *sdispls, HcclDataType sendType,
        u64 *recvCounts, u64 *rdispls, HcclDataType recvType, u64 &memSize);

    virtual HcclResult GetAlltoAllStagedWorkSpaceMemSize(std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo,
        u64 &memSize);

    virtual HcclResult GetAllReduceScratchSize(const u32 count, const HcclDataType dataType, u64 &scratchSize) const;

    virtual bool IsStandardCard();

    virtual bool Is310PDuoCard();

    HcclResult CheckDeviceType(const DevType deviceType) const;

    HcclResult CheckReductionOp(const HcclReduceOp op) const;

    HcclResult CheckUserRank(const u32 userRank) const;

    HcclResult CheckCount(const u64 count) const;

    HcclResult GetGroupCommonData(WorldGroupInfo &groupCommonData) const;

    HcclResult GetHccsLinkNum(u32 &numHccsLink);

    HcclResult GetGroupRanksInfo(const std::vector<u32> &groupRanks, std::vector<RankInfo> &ranksInfo);

    static bool CompareWithDevicePhyId(const RankInfo_t &left, const RankInfo_t &right);

    static bool CompareWithServerId(const ServerInfo_t &left, const ServerInfo_t &right);

    static bool CompareWithUserRank(const RankInfo &left, const RankInfo &right);

    static bool CompareWithNicName(const NetworkInfo_t &left, const NetworkInfo_t &right);
    u32 GetUserRank();
    u32 GetGroupRank();
    u32 GetRankSize();
    HcclResult GetNicInfo(const NICDeployment &nicDeploy, const u32 curRankIndex,
        const std::vector<RankInfo_t> &servRankList, RankInfo &rankInfo) const;
    /* * 以下两函数用于防止重复初始化 */
    HcclResult AtomicInitSet();
    void AtomicInitClear();
    bool GetNicInitialized();
    void DestroyAlgResource(AlgResourceResponse &res);
    HcclResult DestroyNetworkResources();
    HcclResult DisablePreResource();
    HcclResult GetWorkspaceSubStreamNum(u64 &streamNum, u64 dataSize, HcclCMDType opType);
    HcclResult GetWorkspaceMemSize(const std::string &opType, u64 count, HcclDataType dataType,
                                   u32 &rankSize, u64 &size, DevType &deviceType) const;
    HcclResult SetWorkspaceResource(const std::string &tag, void *memPtr, u64 &maxSize,
        std::vector<rtStream_t> &stream);
    HcclResult CreateOpBasedResources(const HcclCMDType &opType, const std::string &tag,
        const HcomCollOpInfo &opInfo);
    HcclResult CreateRemoteOpBasedResources(u64 memSize, const std::string &tag);
    HcclResult DestroyRemoteOpBasedMem(const std::string &tag);
    void DestroyWorkspaceResource(const std::string &tag);
    HcclResult GetInCCLbuffer(void* &buffer, u64 &size);
    HcclResult GetOutCCLbuffer(void* &buffer, u64 &size);
    void ReleaseCommCCLbuffer();
    HcclResult CreateCommCCLbuffer();
    HcclResult InitCCLbuffer(u64 inCCLbufferSize, u64 outCCLbufferSize);
    HcclResult SetQosCfg(const u32 qosCfg);
    HcclResult ResetQosCfg();
    HcclResult GetQosCfg(u32& qosCfg);

    // 目前支持按tag对资源释放、解绑定
    virtual HcclResult ClearOpResource(const std::string &tag);

    HcclResult SetGlobalWorkSpace(std::vector<void *> &globalWorkSpaceAddr);
    // 获得rdma with reduce算子溢出的task信息后清除
    HcclResult GetandClearOverFlowTasks(std::vector<HcclDumpInfo> &hcclDumpInfo);

    HcclResult GetDeviceId(s32 &deviceId) const;

    HcclResult SetMeshAggregationRankSize(u32 size);

    virtual void Break();
    HcclResult SetDevicePid(s32 devicePid);
    HcclResult DestroyCDomainResource(s32 tag);

    std::unordered_map<std::string, std::map<u32, HcclIpAddress>> GetPhyIdNicInfo();
    std::vector<u32> GetRanksPort();
    std::vector<RankInfo> GetRanksList();
    HcclResult SetWorldGroupInfo(
        std::unordered_map<std::string, std::map<u32, HcclIpAddress>> phyIdNicInfoMap,
        std::vector<RankInfo> worldRankInfoList, std::vector<u32> ranksPort);
    virtual HcclResult SaveOpbaseKeyTraceInfo(std::string &logInfo);
    virtual bool GetCommResource(const std::string &tag, void **commContext);

    virtual HcclResult GetAicpuOpStreamNotify(HcclRtStream *opStream, void** aicpuNotify);

    virtual HcclResult CreateCommResource(const std::string &tag, rtStream_t aiCpuStream, bool isOpbaseMode,
        void **commContext);
    virtual HcclResult Mc2AiCpuKernelLaunch(const rtStream_t stm, u64 addr);
    virtual HcclResult AicpuUnfoldKernelLaunch(void *inputPtr, void *outputPtr, const rtStream_t stm, u64 addr,
        HcclKFCTilingData &tilingDate);
    virtual HcclResult Mc2AiCpuStreamAllocAndGet(u32 streamMode, rtStream_t &aiCpuStream);
    HcclResult GetTopoDesc(HcclTopoDescs *topoDescs, uint32_t topoSize);
    HcclResult ReStartVnic(const HcclCommParams &params, const RankTable_t &rankTable);
    static std::string GetUniqueId(void);

    u8 GetDeterministicConfig() const;  // 获取确定性计算配置
    HcclResult SetDeterministicConfig(const u8 deterministic);  // 设置确定性计算配置

private:
    HcclResult InitCommParams(HcclCommParams &params);
    HcclResult InitRankInfo(const RankTable_t &rankTable);
    HcclResult InitRankInfoSubGroup(const std::vector<RankInfo> &rankList, WorldGroupInfo &groupCommonData);
    HcclResult InitTopoInfo(const RankTable_t &rankTable);
    HcclResult InitTopoInfo(const std::vector<RankInfo> &rankList); // For Subgroup
    HcclResult InitNetResource(const RankTable_t &rankTable);
    HcclResult InitDebug();
    HcclResult InitDebugSubGroup();
    HcclResult InitATraceInfo();
    HcclResult InitNotifyManager();
    HcclResult InitDispatcher();
    HcclResult InitStreamManager();
    HcclResult InitSocketManager();
    HcclResult InitTransportManager();
    HcclResult InitMemoryManager();
    HcclResult InitMemoryManagerSubGroup();
    HcclResult InitHcclAlg();
    bool IsDiffDeviceModule(const std::vector<RankInfo_t> &rankList) const;
    HcclResult InitProfiling();
    HcclResult DeinitProfiling();
    HcclResult InitProfiler();
    HcclResult RegistTaskExceptionHandler() const;
    HcclResult UnRegistTaskExceptionHandler() const;
    HcclResult InitPreResource(const RankTable_t &rankTable);
    HcclResult InitTcpMode(const RankTable_t &rankTable) const;
    HcclResult InitRaResource();
    bool IsNeedNicInit();
    HcclResult InitNic();
    HcclResult DeinitNic();

    HcclResult RegisterToHeartBeat();

    HcclResult MrManagerInit();
    HcclResult MrManagerDeInit();
    HcclResult InitRecvMsgAndRequestBuffer();
    HcclResult InitMemBlocksAndRecvWrMem();

    HcclResult PrintOpbaseKeyTraceInfo(void);
    HcclResult GetServerNum(const std::vector<RankInfo_t> &ranks);
    HcclResult GetServerId(const RankTable_t &rankTable);
    HcclResult GetInnerServerAverageDevice(const RankTable_t &rankTable);
    HcclResult GetInnerServerAverageDevice(const std::vector<RankInfo> &rankList);
    HcclResult TransformRankInfoByServerId(const std::vector<RankInfo_t> &rankList,
        ServRankInfo_t &servRankInfo) const;
    HcclResult TransformRankList(const std::vector<RankInfo> &rankListIn,
        std::vector<RankInfo_t> &rankListOut);
    HcclResult CheckDevPhyId(const s32 &devicePhyId) const;
    HcclResult CheckRankTable(const RankTable_t &rankTable, const ServRankInfo_t &servRankInfo);
    HcclResult CheckDevCount(const u32 devNum);
    HcclResult SortRankInfoList();
    HcclResult CheckNicDeploy(NICDeployment nicDeploy, DevType deviceType) const;
    bool Check2N(u32 num) const;
    void CollectAlgoAttr(HcclAlgoAttr &algoAttr);
    void CollectTopoAttr(HcclTopoAttr &topoAttr);
    HcclResult InitPara();
    HcclResult GetRankInfoList(const RankTable_t &rankTable);
    u32 CalMeshAggRankSize(int halfDevNum) const;
    HcclResult CalAndSetMeshAggRankSize();

    bool IsUsedRdmaOuterAndIpInvalid();
    HcclResult GetComm(const std::string &tag, CommBase **comm);
    HcclResult Mc2CreateAndLaunchContext(rtStream_t aiCpuStream, bool isOpbaseMode, void **commContext);
    HcclResult SetCommResource(u64 commBufferSize, void *commInPtr, void *commOutPtr,
        CommBase *comm, innerStreamInfo_t &streamInfo, Stream &stream);
    HcclResult GetAicpuOpStreamAndNotify(HcclRtStream *opStream, void** aicpuNotify);
    HcclResult SetAicpuNotifyInvaild();
    HcclResult AicpuKfcTilingDataLaunch(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
        HcclDataType dataType, HcclReduceOp op, HcclRtStream stream, HcclCMDType opType);

    HcclResult AllReduceAicpuUnfold(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
        HcclDataType dataType, HcclReduceOp op, HcclRtStream stream);
    HcclResult CreateMutiStreamResFor310P(const std::string &tag, innerStreamInfo_t &streamInfo);
    HcclResult CheckSuperDeviceId(const RankTable_t &rankTable);

    u32 deviceNumPerServer_;
    HcclDispatcher dispatcher_; // dispatcher放到最后析构
    HcclDispatcher vDispatcher_; // virtualDispatcher放到最后析构
    std::unique_ptr<NotifyPool> notifyPool_;
    std::unique_ptr<HcclCallbackTask> callbackTask_;
    std::atomic_flag initializedFlag_;
    u32 userRank_;  // 本group中的userrank
    u32 realUserRank_;  // world group中的userrank
    u32 userRankSize_;
    std::vector<RankInfo> rankInfoList_;  // world group内rank的信息, 按照rank id递增依次排列
    std::vector<HbRankInfo> hbRankInfoList_;  // world group内rank的信息, 按照rank id递增依次排列, 心跳模块使用
    bool drvInit_;                          // ra是否初始化
    ServRankInfo_t servRankInfo_;
    std::string serverId_;
    std::unordered_map<u32, std::unordered_map<int, std::vector<int>>> pairLinkInfo_; // server内所有device间的链路类型
    bool inlineReduceSwitchOn_;
    NICDeployment nicDeployment_;
    u32 devicePhyId_;
    s32 deviceLogicId_;
    std::vector<HcclIpAddress> devIpAddr_;
    HcclIpAddress hostIp_;
    u32 hostPort_;
    u32 localRank_;
    SocketHandle hostSocketHandle_;
    SocketHandle loopbackHeterogSocketHandle_;
    bool isUsedRdmaOuter_; // 节点内是否使用rdma, 包括a+x和标卡
    bool nicInitialized_;
    bool hcomGroupNicInit_;
    // profiling 相关资源
    HcomProfilingMode profilingMode_;
    std::string profilingOption_;
    ProfilingDeviceCommResInfo hcclMc2Info_;
    bool raResourceInit_;
    bool interServer_;
    std::unique_ptr<WorkspaceResource> workSpaceRes_;
    std::vector<u32> enableP2PDevices_;
    bool isSingleMeshAggregation_;
    CCLBufferManager cclBufferManager_;
    bool isExecuteProfilingInit_;
    DevType deviceType_;
    std::string collectiveId_;
    HcclComm commHandle_;
    std::vector<u32> ranksPort_;
    std::vector<u32> groupRanksPort_;
    std::unique_ptr<MrManager> mrManager_;
    std::unordered_map<std::string, std::map<u32, HcclIpAddress>> rankDevicePhyIdNicInfoMap_;
    std::unordered_map<u32, HcclRtContext> rtCtxMap_; // {devPhyId, rtCtx}
    WorkMode commWorkMode_;
    u32 meshAggregationRankSize_;
    std::map<HcomOperationType, std::string> opTypeTagMap_;
    bool isHaveCpuRank_;
    bool isUseRankPort_{ false };
    bool isSetHDCModeInfo_{ false };
    std::map<std::string, HostMem> tagWorkSpaceMem_;
    std::string identifier_;
    s32 devicePid_;
    std::unique_ptr<LocklessRingMemoryAllocate<HcclMessageInfo>> pMsgInfosMem_;
    std::unique_ptr<LocklessRingMemoryAllocate<HcclRequestInfo>> pReqInfosMem_;
    std::unique_ptr<HeterogMemBlocksManager> memBlocksManager_;
    std::unique_ptr<LocklessRingMemoryAllocate<RecvWrInfo>> pRecvWrInfosMem_;
    TransportResInfo transportResInfo_;
    bool multiModuleDiffDeviceNumMode_;
    DeviceMem commContext_;
    std::shared_ptr<ProfilerManager> profilerManager_;
    bool isStandardCard_ = false;
    bool is310PDuoCard_ = false;
    std::string superPodId_;
    u32 superDeviceId_ = INVALID_UINT;
    bool useSdidForDeviceId_ = false;
    bool isUsedInterHccsMode_ = false;
    std::vector<RankInfo> worldRankInfoList_;
    std::unique_ptr<HcclOpBaseAtraceInfo> opBaseAtraceInfo_;
private:
    bool IsAtomicInit();
    bool IsSupportEnableRoce();
    bool IsEnableRoce();
    HcclResult InitCombinOpara();
    HcclResult InitWorkSpace();
    void ReleaseWorkSpacebuffer();
    HcclResult CreateWorkSpace(u64 size, DeviceMem &buffer) const;
    HcclResult GetWorkSpace(u64 *workSpaceSize, u64 *workSpace) const;
    void ReleaseCommContextbuffer();
    HcclResult CreateDeviceCommContext(u64 size, DeviceMem &buffer) const;

    HcclResult CreateAndGetAiCpuNotify(std::shared_ptr<LocalNotify> &localNotify, HcclSignalInfo &notifyInfo);
    HcclResult GetAiCpuNotifyData(const std::shared_ptr<LocalNotify> &localNotify, HcclSignalInfo &notifyInfo);
    HcclResult ReplaceCommInfoByTag(const std::string &tag, std::unique_ptr<CommInfo> &commInfo);
    HcclResult CreateCommAndStreamRes(const std::string &tag, Stream &stream);
    HcclResult GetModuleIdx(const RankInfo_t &rankInfo, u32 &moduleIdx);
    HcclResult GetModuleInfo(const std::vector<RankInfo_t> &rankList);
    HcclResult SetInterModeInSuperPod();
    HcclResult CheckSingleServerComm(const std::vector<RankInfo_t> &rankList) const;
    HcclResult SetInfoToDevice(const OpParam &opParam, const std::unique_ptr<PreProcessMetaInfo> &preMetaInfo,
        const HcclWorkflowMode &mode, Stream &stream);
    HcclResult GetInfoFromDevice(const OpParam &opParam, const std::unique_ptr<PreProcessMetaInfo> &preMetaInfo,
        const HcclWorkflowMode &mode, Stream &stream, HostMem& hostCollectBuffer);
    HcclResult RegressCalPreOp(const std::unique_ptr<CollAlgOperator>& algOperator,
        const OpParam &opParam, std::unique_ptr<PreProcessMetaInfo> &preMetaInfo);

    HcclResult ExecOp(HcclCMDType opType, const OpParam &opParam);
    // batchsendrecv专用，增量建链
    HcclResult ExecOpExt(HcclCMDType opType, const OpParam &opParam);
    HcclResult CalcTinySendRecvMem(const OpParam &opParam, AlgResourceResponse &algResResponse,
        DeviceMem &tinySendRecvMem);
    HcclResult AllocAlgResource(const std::string &tag, HcclCMDType opType, const OpParam &opParam,
        AlgResourceRequest &resRequest, AlgResourceResponse &algResResponse);
    HcclResult IncreAllocLink(const std::string &newTag, const OpParam &opParam,
        AlgResourceRequest &resRequest, AlgResourceResponse &algResResponse);
    DeviceMem GetWorkspaceScracthMem(const std::string &tag, u64 allocMemSize);
    std::vector<Stream> GetWorkspaceSubStreams(const std::string &tag, u32 num);
    // HcclImplBase中Comm资源是否存在
    inline bool IsExistCommRes(const std::string &tag)
    {
        std::unique_lock<std::mutex> commLock(commLock_);
        return (tagCommInfo_.find(tag) != tagCommInfo_.end());
    }

    // HcclImplBase中MutiStream资源是否存在
    inline bool IsExistMutiStreamRes(const std::string &tag)
    {
        std::unique_lock<std::mutex> mutiStreamLock(tagStreamInfoLock_);
        return (tagStreamInfo_.find(tag) != tagStreamInfo_.end());
    }

    void GetAndSetSyncMode(SyncMode& preSyncMode, SyncMode newSyncMode);
    void RestorePreSyncMode(SyncMode preSyncMode, SyncMode newSyncMode);
    bool Is310P3Common()
    {
        return !isHaveCpuRank_ && !Is310PDevice() && deviceType_ == DevType::DEV_TYPE_310P3;
    }

    HcclResult AicpuUnfold(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
        HcclDataType dataType, HcclReduceOp op, HcclRtStream stream, HcclCMDType cmdType);
    u32 GetLocalNicPort();
    std::string GetSupportDataType(bool needReduce);
    HcclResult InitHDCommunicate();
    HcclResult InitOpRetry();

    HcclIpAddress loopBackIp_;
    bool profilingInitiated_;
    u64 callbackThreadId_;
    u32 role_;
    bool mrManagerInit_;
    std::map<u64, std::vector<rtStream_t>> callbackStreamMap_;
    bool isHostUseDevNic_;
    std::mutex socketListenMutex_;

    HcclAlg *implAlg_ = nullptr;

    u32 deviceNumPerAggregation_;
    std::vector<u32> nicList_;
    std::unordered_map<u32, u32> pairLinkCounter_; // server内所有device间的链路类型计数
    bool isAllRankSamePlane_;
    std::unique_ptr<TopoInfoParse> topoInfoParse_; // 对rank table device选取的校验模块
    u32 serverNum_;
    u32 moduleNum_;
    u32 superPodNum_ = 0;
    bool isAlgoLevel1Default_ = false;
    HcclCombinOpParam combinOpara_;
    Stream opStream_;
    HcclRtNotify aicpuOpNotify_[2] = { nullptr };
    std::shared_ptr<LocalNotify> localAiCpuNotify_ = { nullptr };
    std::shared_ptr<LocalNotify> localAiCpuOpNotify_[2] = { nullptr };
    u32 workSpaceSize_;
    DeviceMem workSpace_;
    DeviceMem mc2DeviceMem_;
    std::vector<HcclRtEvent> aiCpuNoIpcEvnet_;
    bool isDiffDeviceModule_;
    tagCommInfo_t tagCommInfo_;    // 以tag为粒度分配comm实例和资源
    std::mutex commLock_;
    tagStreamInfo_t tagStreamInfo_;
    std::mutex tagStreamInfoLock_;

    std::vector<Stream> auxRingCommStreamsDev_;
    bool isServerInter_{ false };
    bool isSupportRdmaLite_{ false };          // 是否支持RDMA Lite

    HcclIpAddress localVnicIp_;
    u32 localVnicListenPort_;
    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap_;

    std::unique_ptr<OpBaseStreamManager> opStreamManager_ = { nullptr };
    std::unique_ptr<QueueNotifyManager> queueNotifyManager_ = { nullptr };
    std::unique_ptr<QueueNotifyManager> queueNotifyManagerRefac_ = { nullptr };
    std::unique_ptr<HcclSocketManager> socketManager_;
    std::unique_ptr<TransportManager> transportManager_ = { nullptr };
    std::unordered_map<std::string, AlgResourceResponse> resMap_; // tag : AlgResourceResponse
    bool retryEnable_ = false;
    Stream alltoalltream_;

    std::unique_ptr<OpRetryManagerPub> opRetryManager_ = { nullptr };

    std::shared_ptr<HDCommunicate> kfcControlTransferH2D_;
    std::shared_ptr<HDCommunicate> kfcStatusTransferD2H_;
    HcclCommConnections commConnections_;
    std::shared_ptr<HcclOpStreamRes> opRetryStreamPtr_;
};

}  // end namespace hccl
#endif  // HCCL_IMPL_BASE_H
