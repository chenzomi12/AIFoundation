/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_COMM_PUB_H
#define HCCL_COMM_PUB_H

#include <vector>
#include <memory>
#include <map>
#include <mutex>
#include "base.h"
#include "hccl_common.h"
#include "mem_device_pub.h"
#include "topoinfo_struct.h"
#include "comm.h"
#include "topoinfo_struct.h"
#include "transport_heterog_def.h"

namespace hccl {
/* * 默认的rank_table, ranklist为空数组;  后续HCCL可以用于判断是否走新分支 */
extern RankTable_t g_hcclDefaultRankTable;

class HcclCommunicator;

class hcclComm {
public:
    explicit hcclComm(u64 inCCLbufferSize = 0, u64 outCCLbufferSize = 0, std::string identifier = "");
    ~hcclComm();

    /**********************************************************************
     函 数 名  : hcclComm::init
     功能描述  : 集合通信域初始化
     输入参数  : HcclCommParams& params
             const RankTable_t &rankTable
     输出参数  : 无
     返 回 值  : HcclResult
    **********************************************************************/
    HcclResult init(HcclCommParams &params, const RankTable_t &rankTable = g_hcclDefaultRankTable);
    HcclResult init(HcclCommParams &params, const std::vector<RankInfo> &rankList, WorldGroupInfo &groupCommonData);

    /**********************************************************************
     功能描述  : 创建以group为名字的集合通信
     输入参数  : const std::string& group
             const u32& groupRank
             const std::vector<u32>& groupRanks
     输出参数  : std::shared_ptr<hcclComm>& groupComm
     返 回 值  : HcclResult
    **********************************************************************/
    HcclResult CreateGroup(const std::string &group, const u32 &groupRank, const u32 &userRank,
        const std::vector<u32> &groupRanks, std::shared_ptr<hcclComm> &groupComm);

    /**********************************************************************
     功能描述  : 销毁以group为名字的集合通信
     输入参数  : const std::string& group
     返 回 值  : HcclResult
    **********************************************************************/
    HcclResult DestroyGroup(const std::string &group) const;

    /**********************************************************************
     功能描述  : 查询当前的算法类型
     输出参数  : AlgType &algType
     返 回 值  : HcclResult
    **********************************************************************/
    HcclResult GetAlgType(AlgType &algType, HcclCMDType opType);

    /**********************************************************************
     功能描述  : all gather功能实现
     输入参数  : const char *tag
                 const void* input_ptr
                 void *outputPtr
                 s32 inputCount
                 HcclDataType datatype
                 rtStream_t stream
     输出参数  : void* output_ptr
     返 回 值  : HcclResult
    **********************************************************************/
    HcclResult AllGather(const std::string &tag, void *inputPtr, void *outputPtr, u64 inputCount, HcclDataType dataType,
        rtStream_t stream, HcomCollOpInfo *opInfo = nullptr);
    HcclResult AllGatherOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 inputCount,
        HcclDataType dataType, rtStream_t stream);

    /* *********************************************************************
     功能描述  : all reduce功能实现
     输入参数  : const char *tag
                 const void* input_ptr
                 void *outputPtr
                 s32 count
                 HcclDataType data_type
                 HcclReduceOp op
                 rtStream_t stream
     输出参数  : void* output_ptr
     返 回 值  : HcclResult
    ********************************************************************* */
    HcclResult AllReduce(const std::string &tag, void *inputPtr, void *outputPtr, u64 count, HcclDataType dataType,
        HcclReduceOp op, rtStream_t stream, SyncMode syncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE);
    HcclResult AllReduceOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
        HcclDataType dataType, HcclReduceOp op, rtStream_t stream,
        SyncMode syncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE);
    /* *********************************************************************
     功能描述  : broadcast功能实现
     输入参数  :const char *tag
                 void* ptr
                 s32 count
                 HcclDataType dataType
                 s32 root
                 rtStream_t stream
     输出参数  : void* ptr
     返 回 值  : HcclResult
    ********************************************************************* */
    HcclResult Broadcast(const std::string &tag, void *ptr, u64 count, HcclDataType dataType,
        u32 root, rtStream_t stream);
    HcclResult BroadcastOutPlace(const std::string &tag, void *ptr, u64 count, HcclDataType dataType, u32 root,
        rtStream_t stream);
    /* *********************************************************************
     功能描述  : scatter功能实现
     输入参数  : const char *tag
                const void* input_ptr
                void *outputPtr
                u64 recvCount
                HcclDataType dataType
                u32 root
                rtStream_t stream
     输出参数  : void* ptr
     返 回 值  : HcclResult
    ********************************************************************* */
    HcclResult Scatter(const std::string &tag, void *inputPtr, void *outputPtr, u64 recvCount, HcclDataType dataType,
        u32 root, rtStream_t stream);
    HcclResult ScatterOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 recvCount,
        HcclDataType dataType, u32 root, rtStream_t stream);
    /**********************************************************************
     功能描述  : reduce功能实现
     输入参数  : const char *tag
                 const void* input_ptr
                 void *outputPtr
                 s32 count
                 HcclDataType data_type
                 HcclReduceOp op
                 s32 root,
                 rtStream_t stream
     输出参数  : void* output_ptr
     返 回 值  : HcclResult
    **********************************************************************/
    HcclResult Reduce(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
        HcclDataType dataType, HcclReduceOp op, u32 root, rtStream_t stream);
    HcclResult ReduceOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
        HcclDataType dataType, HcclReduceOp op, u32 root, rtStream_t stream);

    /**********************************************************************
     功能描述  : reduce-scatter功能实现
     输入参数  : const char *tag
                 const void* input_ptr
                 void *outputPtr
                 s32 count
                 HcclDataType data_type
                 HcclReduceOp op
                 rtStream_t stream
     输出参数  : void* output_ptr
     返 回 值  : HcclResult
    **********************************************************************/
    HcclResult ReduceScatter(const std::string &tag, void *inputPtr, void *outputPtr, u64 recvCount,
        HcclDataType dataType, HcclReduceOp op, rtStream_t stream);
    HcclResult ReduceScatterOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 recvCount,
        HcclDataType dataType, HcclReduceOp op, rtStream_t stream);

    HcclResult ProcessSendRecvTasks(const std::string &tag, std::vector<struct HcclSendRecvItemDef *> &orderedList,
        u32 itemNum, u32 startIndex, rtStream_t stream);

    HcclResult send(const std::string &tag, void *inputPtr, u64 count, HcclDataType dataType, u32 destRank,
        rtStream_t stream);
    HcclResult SendOutPlace(const std::string &tag, void *inputPtr, u64 count, HcclDataType dataType, u32 destRank,
        rtStream_t stream);

    HcclResult receive(const std::string &tag, void *outputPtr, u64 count, HcclDataType dataType, u32 srcRank,
        rtStream_t stream);
    HcclResult ReceiveOutPlace(const std::string &tag, void *outputPtr, u64 count, HcclDataType dataType, u32 srcRank,
        rtStream_t stream);

    HcclResult AlltoAllV(const void *sendBuf, const void *sendCounts, const void *sdispls, HcclDataType sendType,
        const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType, rtStream_t stream,
        const std::string &tag);
    HcclResult AlltoAllVOutPlace(const void *sendBuf, const void *sendCounts, const void *sdispls,
        HcclDataType sendType, const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType,
        rtStream_t stream, const std::string &tag);

    HcclResult AlltoAllVC(const void *sendBuf, const void *sendCountMatrix, HcclDataType sendType, const void *recvBuf,
        HcclDataType recvType, rtStream_t stream, const std::string &tag);
    HcclResult AlltoAllVCOutPlace(const void *sendBuf, const void *sendCountMatrix, HcclDataType sendType,
        const void *recvBuf, HcclDataType recvType, rtStream_t stream, const std::string &tag);

    HcclResult AlltoAll(const void *sendBuf, u64 sendCount, HcclDataType sendType, const void *recvBuf,
        u64 recvCount, HcclDataType recvType, rtStream_t stream, const std::string &tag);

    HcclResult Gather(const std::string &tag, void *inputPtr, void *outputPtr, u32 rootRank, u64 inputCount,
        HcclDataType dataType, rtStream_t stream);

    /**********************************************************************
     功能描述  : 生成唯一的集合通信域标识
     输入参数  : 无
     输出参数  : HcclRootInfo* rootInfo
     返 回 值  : HcclResult
    **********************************************************************/
    static HcclResult GetUniqueId(HcclRootInfo *uniqueId);

    HcclResult GetInCCLbuffer(void* &buffer, u64 &size);
    HcclResult GetOutCCLbuffer(void* &buffer, u64 &size);
    HcclResult GetUserRank(u32 &userRank);
    HcclResult GetGroupRank(u32 &userRank);
    HcclResult GetRankSize(u32 &rankSize);
    void ReleaseCommCCLbuffer() const;
    void RealeaseBarrierMemory();
    HcclResult CreateCommCCLbuffer() const;
    HcclResult CreateIndirectCCLbuf();
    void ReleaseIndirectCCLbuf();
    HcclResult GetIndirectInCCLbuf(void* &ptr, u64 &size);
    HcclResult GetIndirectOutCCLbuf(void* &ptr, u64 &size);
    HcclResult GetWorkspaceSubStreamNum(u64 &streamNum, u64 dataSize = 0,
        HcclCMDType optype = HcclCMDType::HCCL_CMD_INVALID) const;
    HcclResult GetWorkspaceMemSize(const std::string &opType, u64 count, HcclDataType dataType,
                                   u32 &rankSize, u64 &size);
    HcclResult GetAllReduceScratchSize(const u32 count, const HcclDataType dataType, u64 &scratchSize) const;
    HcclResult SetWorkspaceResource(const std::string &tag, void *memPtr, u64 maxSize,
                                    std::vector<rtStream_t> &stream);
    HcclResult CreateOpBasedResources(const HcclCMDType &opType, const std::string &tag,
        const HcomCollOpInfo &opInfo);

    std::string GetIdentifier();
    HcclResult CreateBarrierMemory();
    HcclResult ReleaseSubComms() const;
    HcclResult GetAlltoAllStagedWorkSpaceMemSize(u64 *sendCounts, u64 *sdispls,
        HcclDataType sendType, u64 *recvCounts, u64 *rdispls, HcclDataType recvType, u64 &memSize) const;
    HcclResult GetAlltoAllStagedWorkSpaceMemSize(std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo,
        u64 &memSize) const;
    // 目前支持按tag对资源释放、解绑定
    HcclResult ClearOpResource(const std::string &tag);
    HcclResult Isend(void *buffer, s32 count, HcclDataType dataType, u32 peerRank, s32 tag, HcclRequest &request,
        HcclUserRequire &userRequire) const;
    HcclResult Improbe(u32 peerRank, s32 tag, s32 &flag, HcclMessage &msgHandle, HcclStatus &status) const;
    HcclResult Imrecv(void *buffer, s32 count, HcclDataType dataType, HcclMessage msg, HcclRequest &request) const;
    HcclResult HcclTest(HcclRequest hcclRequest, s32 &flag, HcclStatus &compState) const;
    // 获取溢出Flag内存传给RTS
    HcclResult SetGlobalWorkSpace(std::vector<void *> &globalWorkSpaceAddr);
    // 获取rdma with reduce算子溢出的task信息，然后清除
    HcclResult GetandClearOverFlowTasks(std::vector<HcclDumpInfo> &hcclDumpInfo);
    HcclResult SupportDeterministicOptim(bool &isDeterministicOptim);
    HcclResult GetHccsLinkNum(u32 &numHccsLink);
    HcclResult GetDeviceId(s32 &deviceId);
    HcclResult GetDevType(DevType &devType);
    HcclResult IsStandardCard(bool &isStandardCard);
    HcclResult Is310PDuoCard(bool &is310PDuoCard);
    HcclResult AbortSelf(s32 tag);

    HcclResult SetQosCfg(const u32 qosCfg);
    HcclResult ResetQosCfg();
    HcclResult GetQosCfg(u32& qosCfg);
    HcclResult RegTransportLinks(s32 linkNum, void *transportPara);
    HcclResult GetDeviceNumPerAggregation(u32 &deviceNumPerAggregation);
    HcclResult GetBandWidthPerNPU(u32 level, float &bandWidth);
    bool IsNeedResetDevice();
    HcclResult ResetDeviceEnable();
    HcclResult CommCheckErrorCqe(HcclResult &result);
    HcclResult SaveOpbaseKeyTraceInfo(std::string &logInfo);
    HcclResult CreateCommResource(const std::string &tag, rtStream_t aiCpuStream, bool isOpbaseMode,
        void **commContext);
    bool GetCommResource(const std::string &tag, void **commContext);
    HcclResult GetAicpuOpStreamNotify(HcclRtStream *opStream, void** aicpuNotify);
    HcclResult Mc2AiCpuStreamAllocAndGet(u32 streamMode, rtStream_t &aiCpuStream);
    HcclResult GetAiCpuNotifyData(HcclRtNotify notifyHandle, HcclSignalInfo &notifyInfo);
    HcclResult AddAiCpuNotify(HcclRtNotify *notifyHandle);
    HcclResult GetTopoDesc(HcclTopoDescs *topoDescs, uint32_t topoSize);
    HcclResult ReStartVnic(const HcclCommParams &params, const RankTable_t &rankTable);
    HcclResult SetDeterministicConfig(const u8 deterministic);  // 设置确定性计算配置
    u64 GetConfigInCCLbufferSize();     // 获取通信域配置的输入buffer大小
    u64 GetConfigOutCCLbufferSize();    // 获取通信域配置的输出buffer大小

    void* barrierSendBuf;
    void* barrierRecvBuf;
    std::mutex operatorlock_;
protected:
    /* * 禁止用户对API类的实体做拷贝构造或拷贝赋值的操作，内部有指针成员变量 */
    hcclComm(const hcclComm &) = delete;
    hcclComm &operator=(const hcclComm &) = delete;
private:
    HcclResult InitImpl(DevType deviceType);
    void UpdateIsHaveCpuRank(const RankTable_t &rankTable);
    void UpdateIsHaveCpuRank(const std::vector<RankInfo> &rankList);
    DeviceMem indirectInCCLbuffer_; /* 保存inCCLbuffer指针的地址 */
    DeviceMem indirectOutCCLbuffer_; /* 保存outCCLbuffer_指针的地址 */
    u64 inCCLbufferSize_;
    u64 outCCLbufferSize_;
    DevType deviceType_;
    DeviceMem barrierInMemory_;
    DeviceMem barrierOutMemory_;
    bool isFirstBarrier_;
    const std::string identifier_;
    bool isHeterogComm_;

    bool isResetDevice_;
    bool isSpecialType_;
    bool isHaveCpuRank_{false};
    std::unique_ptr<HcclCommunicator> communicator_;
};
}  // namespace hccl

using HcclCommPtr = std::shared_ptr<hccl::hcclComm>;
#endif /* HCCL_COMM_PUB_H */
