/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMM_BASE_PUB_H
#define COMM_BASE_PUB_H

#include <memory>
#include <mutex>
#include <map>
#include <sys/types.h>
#include <hccl/hccl_types.h>

#include "hccl_common.h"
#include "hccl_socket_manager.h"
#include "transport_pub.h"
#include "executor_base_pub.h"
#include "alltoallv_pairwise_pub.h"
#include "alltoallv_staged_base_pub.h"
#include "common.h"


namespace hccl {
constexpr u32 HCCL_RANK_SIZE_EQ_ONE = 1;
constexpr u32 HCCL_RANK_SIZE_EQ_TWO = 2;
constexpr u32 HCCL_RANK_ZERO = 0;
constexpr u32 HCCL_RANK_OFFSET = 1;

constexpr u32 FACTOR_NUM_TWO = 2;
constexpr s32 DEVICE_PER_MODULE = 8;         // 单module支持最大device数量

class CommBase {
public:
    explicit CommBase(const std::string &collectiveId,
                      const u32 userRank, const u32 userRankSize, const u32 rank,
                      const u32 rankSize, const std::vector<RankInfo> paraVector,
                      const TopoType topoFlag,
                      const HcclDispatcher dispatcher,
                      const std::unique_ptr<NotifyPool> &notifyPool,
                      std::map<HcclIpAddress, HcclNetDevCtx> &netDevCtxMap,
                      const IntraExchanger &exchanger,
                      const DeviceMem &inputMem, const DeviceMem &outputMem, const bool isUsedRdmaOuter,
                      const void* transportResourceInfoAddr = nullptr, size_t transportResourceInfoSize = 0,
                      const std::string &tag = "",
                      const NICDeployment nicDeployInner = NICDeployment::NIC_DEPLOYMENT_DEVICE,
                      bool isAlltoAllCommMesh = false, const bool useOneDoorbell = false,
                      const bool isAicpuModeEn = false, const u32 rankRoot = INVALID_UINT,
                      const bool isHaveCpuRank = false, const bool useSdidForDeviceId = false);
    virtual ~CommBase();

    inline const std::string &CollectiveId() const
    {
        return collectiveId_;
    }

    inline const std::vector<LINK> &TransportInfo() const
    {
        return transportInfo_;
    }

    inline const u32 UserRank() const
    {
        return userRank_;
    }

    inline const u32 UserRankSize() const
    {
        return userRankSize_;
    }

    inline const u32 Rank() const
    {
        return rank_;
    }

    inline const u32 RankSize() const
    {
        return rankSize_;
    }

    inline const std::string &Tag() const
    {
        return tag_;
    }

    inline void SetHeterogP2PType()
    {
        isNeedHeterogP2P_ = true;
    }

    inline void SetHostUseDevNic()
    {
        isHostUseDevNic_ = true;
    }

    virtual HcclResult Init(); // 初始化必要信息
    virtual HcclResult DeInit();

    std::shared_ptr<Transport> &GetTransportByRank(const u32 dstRank); // 获取当前rank与dst rank的link信息
    HcclResult GetRankByUserRank(const u32 userRank, u32 &rank) const; // 获取当前userrank在重新排序后的rank
    HcclResult GetUserRankByRank(const u32 rank, u32 &userRank) const; // 获取当前rank的userrank
    HcclResult GetRaSocket(const u32 role, HcclSocketInfo conn[], const u32 num);
    HcclResult CreateIntraThread(const u32 role, u32 dstRank,
        const std::vector<std::shared_ptr<HcclSocket> > &sockets); // 节点内建链起线程
    HcclResult CreateInterThread(const u32 role, u32 dstRank,
        const std::vector<std::shared_ptr<HcclSocket> > &sockets); // 节点间建链起线程
    HcclResult RunExecutor(const std::unique_ptr<ExecutorBase> &executor);
    HcclResult RunExecutorStaged(const std::unique_ptr<ExecutorBase> &executor, const RunStage &stage);
    HcclResult RunAlltoAll(const std::unique_ptr<AlltoAllVPairWise> &executor);
    HcclResult RunAlltoAllVStaged(const std::unique_ptr<AlltoAllVStagedBase> &executor);
    HcclResult RunAlltoAllVStagedMesh(const std::unique_ptr<AlltoAllVStagedBase> &executor);
    std::shared_ptr<Transport> &GetTrasportInfoByVTransportInfoIndex(u32 index);
    HcclResult BuildAsync(u32& status);
    HcclResult BuildQuerry(u32& status);
    HcclResult SetHDCModeInfo(
        std::unordered_map<std::string, std::map<u32, HcclIpAddress>> &rankDevicePhyIdNicInfoMap,
        std::vector<u32> &ranksPort, bool isSetHDCModeInfo, bool isUseRankPort);

    void Break()
    {
        for (auto &ptr : transportInfo_) {
            if (ptr == nullptr) {
                continue;
            }
            ptr->Break();
        }
    }

    std::shared_ptr<Transport> linkDummy_; // 用于无效的link返回
protected:
    // 创建当前rank与其他rank之间的link:ring\mesh\H-D分别在对应派生类实现
    virtual HcclResult CreateLinks();
    // 计算当前rank与其他rank之间的link个数:server/client两种角色,ring\mesh\H-D分别在对应派生类实现
    virtual HcclResult CalcLink();
    // 获取每个 link 需要的 socket 数量
    virtual u32 GetSocketsPerLink();
    virtual bool NeedDataReceivedAck();
    virtual void SetMachineLinkMode(MachinePara &machinePara);
    virtual HcclResult CreateIntraLinks(); // 当前rank在服务器内与对端的建链
    virtual HcclResult CreateInterLinks(); // 当前rank在服务期间与对端的建链
    virtual HcclResult SetMachinePara(MachineType machineType, const std::string &serverId, u32 dstRank,
        const std::vector<std::shared_ptr<HcclSocket> > &sockets, MachinePara &machinePara);
    virtual void SetTransportParam(TransportPara &para, MachinePara &machinePara);
    virtual HcclResult MakeClientInfo(const u32 dstRank, RankInfo &dstRankInfo, bool isInterRdma, bool isInterHccs);
    virtual HcclResult MakeServerInfo(const u32 dstRank, RankInfo &dstRankInfo, bool isInterRdma, bool isInterHccs);

    virtual HcclResult CreateExchangerNetwork(); // server间HCCS通信模式下，创建节点间的建链关系
    HcclResult GetRankLinkInfo(bool &isInterServer, bool &isInterHccs, std::map<u32, HcclSocketRole> &rankRole);

    HcclResult CalcLinksNum(const MachineType machineType,
        const u32 dstRank); // 填充dst_inter_server_num_/dst_inter_client_num_
    HcclResult CreateDestLink(const ErrContextPub &error_context, const MachineType machineType,
        const std::string &serverId, const u32 dstRank, const std::string &threadStr,
        const std::vector<std::shared_ptr<HcclSocket> > &sockets); // 创建transhport
    HcclResult TransportInit(const u32 dstRank, MachinePara &machinePara);
    HcclResult SetRankMap(); // 获取rank->userrank以及userrank->rank的映射关系
    HcclResult GetBuildStatus(u32& status);
    HcclResult TransportBuildAsync(const MachineType machineType, const std::string &serverId, u32 dstRank,
        const std::vector<std::shared_ptr<HcclSocket> > &sockets, u32& status);
    HcclResult TransportBuildQuerry(u32 dstRank, u32& status);

    const std::string collectiveId_; /* * 本节点所在的通信域ID */

    const u32 userRank_;      /* * 本节点的用户原始rank号 */
    const u32 userRankSize_; /* * 本节点所在的用户通信域rank size */

    const u32 rank_;      /* * 本节点在子通信域的rank号 */
    const u32 rankSize_; /* * 本节点所在的子通信域的rank size */
    std::vector<RankInfo> paraVector_;        // 子通信域内各个rank的基本信息
    // 与link_info_中每个link类型对应的vector:默认值均为-1，建链成功后填入对应type
    std::vector<TransportType> transportType_;
    std::unordered_map<std::string, std::map<u32, HcclIpAddress>> rankDevicePhyIdNicInfoMap_;
    std::vector<u32> ranksPort_;

    s32 deviceLogicId_;
    u32 devicePhyId_;
    std::map<u32, std::unique_ptr<HcclSocketManager>> pyhIdResourseSockets_{};
protected:
    const TopoType topoFlag_;     // 当前通信域内服务器间拓扑组合类型

    const std::string tag_; /* * 多stream是的tag信息 */
    std::vector<std::shared_ptr<Transport> > transportInfo_; // 当前rank与其他rank对应的link信息
    std::vector<std::shared_ptr<Transport> > vTransportInfo_; // 当前rank与其他rank对应的virtual link信息

    std::vector<u32> rankMap_;       // rank->userrank的映射关系:vector下标是user_rank，存的数据是rank
    std::vector<u32> userRankMap_; // userrank->rank的映射关系:vector下标是rank，存的数据是user_rank

    const HcclDispatcher dispatcher_;    // signal调度句柄(event/notify机制)
    const std::unique_ptr<NotifyPool> &notifyPool_;
    std::map<HcclIpAddress, HcclNetDevCtx> &netDevCtxMap_;
    const IntraExchanger &exchanger_;
    std::map <u32, std::vector<std::shared_ptr<HcclSocket> > > intraSocketsMap_;
    const DeviceMem inputMem_;
    const DeviceMem outputMem_;
    const bool isUsedRdmaOuter_;

    const void* transportResourceInfoAddr_;
    size_t transportResourceInfoSize_;

    // 当前rank作为client端，需要连接的节点间server端的集合:键为rank 号，value为 HcclRankLinkInfo
    // 当前rank作为client端, 需要连接的节点间server端的集合
    std::map<u32, HcclRankLinkInfo> dstInterServerMap_;
    // 当前rank作为server端, 需要连接的节点间client端的集合
    std::map<u32, HcclRankLinkInfo> dstInterClientMap_;
    
    // 当前rank作为client端，需要连接的节点内server端的集合
    std::vector<u32> dstIntraServerVec_;
    // 当前rank作为server端，需要连接的节点内client端的集合
    std::vector<u32> dstIntraClientVec_;

    std::vector<std::unique_ptr<std::thread> > linkThreads_; // 建链所需线程
    u32 threadsRapplyNum_;                                   // 线程使用计数器

    dev_t shmDev_; // dev_t 为u32类型
    bool isAlltoAllCommMesh_;
    NICDeployment nicDeployInner_;
    std::vector<u32> transportStatus_;
    bool isNeedHeterogP2P_;
    bool useOneDoorbell_;

    // 获取当前rank和目的rank之间的link type
    HcclResult SetTransportType(const u32 dstRank);
    HcclResult CheckLinks() const; // 校验当前rank与dst rank建链的链路有效性
    HcclResult CreateVirturalTransport();

    // 统一接口
    HcclResult GetRankIPInfo(bool isInterServer, bool isInterHccs, bool &isSupportReuse,
        std::map<u32, HcclSocketRole> &rankRole, HcclIpAddress &localIP,
        std::map<u32, HcclRankLinkInfo> &dstServerMap,
        std::map<u32, HcclRankLinkInfo> &dstClientMap,
        std::shared_ptr<HcclSocketManager> &socketManager);

    // server内使用
    HcclResult GetIntraRankIPInfo(std::map<u32, HcclSocketRole> &rankRole,
        HcclIpAddress &localIP,
        std::map<u32, HcclRankLinkInfo> &dstServerMap,
        std::map<u32, HcclRankLinkInfo> &dstClientMap);

    // 310p device侧、isHaveCpuRank_ 业务场景使用
    HcclResult GetIntraRankIPInfo(std::vector<u32> &dstIntraVec,
        HcclIpAddress &localIP,
        std::map<u32, HcclRankLinkInfo> &dstServerMap,
        std::map<u32, HcclRankLinkInfo> &dstClientMap);

    // 超节点server间HCCS场景使用
    HcclResult GetSuperNodeIntraRankIPInfo(std::map<u32, HcclSocketRole> &rankRole,
        HcclIpAddress &localIP,
        std::map<u32, HcclRankLinkInfo> &dstServerMap,
        std::map<u32, HcclRankLinkInfo> &dstClientMap);

    bool IsSupportInterHccs(const u32 dstRank); // 是否支持走节点间HCCS通信
    u32 GetInterRemotePort(s32 devicePhyId, u32 dstUserRank);
    void PrintCreateInterLinksInfo(); // 打印当前rank在服务器间建链通信topo对
    const bool isAicpuModeEn_;
    std::unique_ptr<HcclSocketManager> interSocketManager_;
    const u32 subUserRankRoot_;
    bool isHostUseDevNic_{ false };
    bool isSetHDCModeInfo_{ false };
    bool isUseRankPort_{ false };
    bool isHaveCpuRank_{ false };
    bool useSdidForDeviceId_{ false };
    std::shared_ptr<HcclSocketManager> socketManager_{ nullptr };
};
}  // namespace hccl

#endif /* COMM_BASE_PUB_H */
