/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMM_FACTORY_PUB_H
#define COMM_FACTORY_PUB_H

#include <vector>
#include <map>
#include <memory>

#include <hccl/base.h>
#include <hccl/hccl_types.h>
#include "hccl_common.h"
#include "comm_base_pub.h"
#include "hccl_socket_manager.h"
#include "dispatcher.h"
#include "coll_alg_param.h"

namespace hccl {
constexpr u32 COMM_P2P_QUERRY_WAIT_TIME = 100;
enum class CommType {
    COMM_TAG_RING_INNER = 0,
    COMM_TAG_RING_COMBINED,
    COMM_TAG_HALVING_DOUBLING,
    COMM_TAG_STAR,
    COMM_TAG_NONUNIFORM_HIERARCHICAL_RING,
    COMM_TAG_WHOLE_NHR,
    COMM_TAG_NONUNIFORM_HIERARCHICAL_RING_V1,
    COMM_TAG_WHOLE_NHR_V1,
    COMM_TAG_NONUNIFORM_BRUCK,
    COMM_TAG_WHOLE_NB,
    COMM_TAG_MESH_COMBINED,
    COMM_TAG_MESH,
    COMM_TAG_P2P,
    COMM_TAG_PARTIAL_MESH_COMBINED,
    COMM_TAG_MAX,
};

// 通信域级别
typedef enum {
    COMM_LEVEL0 = 0,    // 一级通信域(server内)
    COMM_LEVEL0_RDMA,
    COMM_LEVEL1,        // 二级通信域(server间)
    COMM_LEVEL1_RDMA,
    COMM_LEVEL2,        // 三级通信域(超节点间)
    COMM_MESH_L0,       // mesh内
    COMM_MESH_L1,       // mesh间
    COMM_COMBINE,       // 打平通信域，大ring环
    COMM_COMBINE_ORDER, // 打平通信域，按rank排序
    COMM_LEVEL_RESERVED,
} CommPlane;

// 通信域建链信息
struct CommParaInfo {
    CommPlane commPlane = COMM_LEVEL_RESERVED;
    CommType commType = CommType::COMM_TAG_MAX;
    u32 root = INVALID_VALUE_RANKID;
    u32 peerUserRank = INVALID_VALUE_RANKID;
    bool isAicpuModeEn = false;
    bool meshSinglePlane = false;
    std::set<u32> batchSendRecvtargetRanks;
    bool forceRdma = false;

    CommParaInfo() {}
    CommParaInfo (CommPlane commPlane, CommType commType, u32 root = INVALID_VALUE_RANKID,
        u32 peerUserRank = INVALID_VALUE_RANKID, bool isAicpuModeEn = false, bool meshSinglePlane = false,
        std::set<u32> batchSendRecvtargetRanks = std::set<u32>(), bool forceRdma = false)
        : commPlane(commPlane), commType(commType), root(root), peerUserRank(peerUserRank),
        isAicpuModeEn(isAicpuModeEn), meshSinglePlane(meshSinglePlane),
        batchSendRecvtargetRanks(batchSendRecvtargetRanks), forceRdma(forceRdma)
    {
    }
};

bool Ascending(const RankInfo &first, const RankInfo &second);  // 排序规则自小到大
bool CompareWithUserRankAscend(const RankInfo &left, const RankInfo &right); // 按UserRank升序
// 生成多个ring环的设备物理ID排序
std::vector<std::vector<u32>> GetRingsOrderByTopoType(u32 ranksSize, TopoType topoType, std::vector<u32> &nicList);

class ExchangerNetwork;
class CommFactory {
public:
    explicit CommFactory(const std::string &identifier, const u32 userRank, const u32 userRankSize,
                         const HcclDispatcher dispatcher, const std::unique_ptr<NotifyPool> &notifyPool,
                         std::map<HcclIpAddress, HcclNetDevCtx> &netDevCtxMap,
                         const bool isUsedRdmaOuter = false,
                         const TopoType topoFlag = TopoType::TOPO_TYPE_COMMON,
                         const DevType deviceType = DevType::DEV_TYPE_910,
                         const std::vector<RankInfo> rankVector = std::vector<RankInfo>(0),
                         const NICDeployment nicDeploymentInner = NICDeployment::NIC_DEPLOYMENT_DEVICE,
                         bool isHeterogComm = false,
                         bool isDiffAggregation = false,
                         const void* transportResourceInfoAddr = nullptr, size_t transportResourceInfoSize = 0,
                         u32 meshAggregationRankSize = 0, bool isHaveCpuRank = false,
                         bool isUsedInterHccsMode = false, bool useSdidForDeviceId = false);

    virtual ~CommFactory();

    HcclResult Init();  // 初始化必要信息
    HcclResult InitComm();  // 310初始化必要信息

    std::vector<std::unique_ptr<CommBase> > CreateCommP2PAsync(const std::string &tag,
        const DeviceMem& inputMem, const DeviceMem& outputMem, const u32 dstUserRank, u32& status);
    HcclResult CreateCommP2PQuerry(std::vector<std::unique_ptr<CommBase> >& comm, u32& status);

    // 创建单层通信域
    HcclResult CreateCommPlane(const std::string &tag,
                               const DeviceMem &inputMem,
                               const DeviceMem &outputMem,
                               const CommParaInfo &commParaInfo,
                               std::vector<std::unique_ptr<CommBase> > &commVec);

    // 提供bcast多环使用,根据指定root节点,获取与当前userrank所在同一平面的子root节点
    u32 GetSubRootUserRank(const u32 userRank, const u32 rootUserRank);
    u32 GetSubRootUserRankWithSuperPod(const u32 userRank, const u32 rootUserRank);
    // 提供scatter使用,根据指定root节点和当前节点的userRank,获取与当前userRank所在同一平面的子root节点
    u32 GetSubRootForScatter(const u32 root);
    // 提供网口裁剪使用，在无节点间通信域场景下，获取本rank在节点间子通信域(多平面)内当前平面的rank号
    u32 GetInnerCommRank(const u32 ringIdx);
    HcclResult SetHDCModeInfo(
        std::unordered_map<std::string, std::map<u32, HcclIpAddress>> &rankDevicePhyIdNicInfoMap,
        std::vector<u32> &ranksPort, bool isSetHDCModeInfo, bool isUseRankPort);

    HcclResult GetCommPlaneRanks(std::vector<std::vector<std::vector<u32>>> &CommPlaneRanks);
    HcclResult GetIsBridgeVector(std::vector<bool> &isBridgeVector);
    HcclResult GetIsUsedRdmaMap(std::unordered_map<u32, bool> &isUsedRdmaMap);
    HcclResult GetRankVecInfo(std::vector<std::vector<std::vector<u32>>> &serverAndsuperPodToRank);

protected:
    /* 禁止用户对工厂类的实体做拷贝构造或拷贝赋值的操作，内部有指针成员变量 */
    CommFactory(const CommFactory &) = delete;
    CommFactory &operator=(const CommFactory &) = delete;
private:
    HcclResult CheckCommPara(const std::string &tag, const DeviceMem &inputMem, const DeviceMem &outputMem,
        const CommParaInfo &commParaInfo);

    HcclResult GetIsUsedRdma(const CommParaInfo &commParaInfo, bool &isUsedRdma);

    HcclResult CreateCommRing(const std::string &tag, const DeviceMem &inputMem, const DeviceMem &outputMem,
        const CommParaInfo &commParaInfo, const std::vector<std::vector<RankInfo> > &commPlaneVec,
        bool isUsedRdma, std::vector<std::unique_ptr<CommBase> > &commVec);

    HcclResult CreateCommHD(const std::string &tag, const DeviceMem &inputMem, const DeviceMem &outputMem,
        const CommParaInfo &commParaInfo, const std::vector<std::vector<RankInfo> > &commPlaneVec,
        bool isUsedRdma, std::vector<std::unique_ptr<CommBase> > &commVec);

    HcclResult CreateCommStar(const std::string &tag, const DeviceMem &inputMem, const DeviceMem &outputMem,
        const CommParaInfo &commParaInfo, const std::vector<std::vector<RankInfo> > &commPlaneVec,
        bool isUsedRdma, std::vector<std::unique_ptr<CommBase> > &commVec);

    HcclResult CreateCommNHR(const std::string &tag, const DeviceMem &inputMem, const DeviceMem &outputMem,
        const CommParaInfo &commParaInfo, const std::vector<std::vector<RankInfo> > &commPlaneVec,
        bool isUsedRdma, std::vector<std::unique_ptr<CommBase> > &commVec);

    HcclResult CreateCommNHRV1(const std::string &tag, const DeviceMem &inputMem, const DeviceMem &outputMem,
        const CommParaInfo &commParaInfo, const std::vector<std::vector<RankInfo> > &commPlaneVec,
        bool isUsedRdma, std::vector<std::unique_ptr<CommBase> > &commVec);

    HcclResult CreateCommNB(const std::string &tag, const DeviceMem &inputMem, const DeviceMem &outputMem,
        const CommParaInfo &commParaInfo, const std::vector<std::vector<RankInfo> > &commPlaneVec,
        bool isUsedRdma, std::vector<std::unique_ptr<CommBase> > &commVec);

    HcclResult CreateCommMesh(const std::string &tag, const DeviceMem &inputMem, const DeviceMem &outputMem,
        const CommParaInfo &commParaInfo, const std::vector<std::vector<RankInfo> > &commPlaneVec,
        bool isUsedRdma, std::vector<std::unique_ptr<CommBase> > &commVec);

    HcclResult CreateCommP2P(const std::string &tag, const DeviceMem &inputMem, const DeviceMem &outputMem,
        const CommParaInfo &commParaInfo, const std::vector<std::vector<RankInfo> > &commPlaneVec,
        bool isUsedRdma, std::vector<std::unique_ptr<CommBase> > &commVec);

    HcclResult CreateCommP2PSync(const std::string &tag, const DeviceMem &inputMem, const DeviceMem &outputMem,
        const CommParaInfo &commParaInfo, const std::vector<std::vector<RankInfo> > &commPlaneVec,
        bool isUsedRdma, std::vector<std::unique_ptr<CommBase> > &commVec);

    HcclResult SetIsUsedRdma(const CommParaInfo &commParaInfo, std::vector<SingleSubCommTransport> &commTransport,
        bool isUsedRdma);

    /*
    * *********************************************************************************
    * comm_factory后续不承担建链功能，只进行通信关系推导
    * *********************************************************************************
    */
    HcclResult GetRankMap(const CommParaInfo &commParaInfo, std::vector<SingleSubCommTransport> &commTransport);

    HcclResult SetRankMap();

    HcclResult GetSub2UserRankMap(CommPlane commPlane, u32 ringIndex, std::map<u32, u32> &subCommRank2UserRank);

    HcclResult GetUserRank2SubMap(CommPlane commPlane, u32 ringIndex, std::map<u32, u32> &subCommRank2UserRank);

    HcclResult SetRankInfo(); // 该函数用于解决后续所要用到的所有数据结构:rank_data_、rank_map_
    HcclResult SetTopologyInfo(); // 设置拓扑信息
    HcclResult SetTopoDefaultInfo(); // 填充combined_rank_vector_:default单层拓扑
    HcclResult SetTopoDefaultInfoFor8P(); // 填充combined_rank_vector_:8P满配
    HcclResult SetTopoInfoForLevel0(); // 填充一层拓扑: server内
    HcclResult SetTopoInfoForLevel1(); // 填充二层拓扑: superPod内server间
    HcclResult SetTopoInfoForLevel2(); // 填充三层拓扑: superPod间
    HcclResult SetTopoInfoForMeshL0(); // 填充mesh内拓扑
    HcclResult SetTopoInfoForMeshL1(); // 填充mesh间拓扑

    HcclResult SetSingleOuter();
    HcclResult SetSingleOuterFor8P();
    HcclResult SetMultiOuter(u32 ringNum);
    HcclResult CheckInitInfo(); // 该函数用于检测factory的构造函数入参合法性
    HcclResult CheckPlaneInfo(); // 该函数用于检测平面信息合法性
    HcclResult CheckServerInfo(); // 该函数用于检测server信息合法性

    HcclResult CheckSuperPodInfo(); // 该函数用于检测superPod信息合法性

    HcclResult SetBridgeLinkInfo(RankInfo &bridgePara, u32 bridgeUserRank);
    // 获取本rank在子通信域(多平面)内当前平面的rank号
    const u32 GetSubCollectiveRank(const std::vector<RankInfo> &vecPara) const;
    HcclResult GetServerIdx(const RankInfo &rankInfo, u32 &serverIdx) const;
    HcclResult GetModuleIdx(const RankInfo &rankInfo, u32 &moduleIdx);
    bool IsDiffDeviceModuleInServer() const;
    bool JudgmentSetHeterogP2p(u32 rank) const;
    void CreateStarLinkPara(std::vector<RankInfo> &linkParas);
    bool IsUseSdidForVnicIp(); // 是否使用sdid作为vnicip

    const std::string identifier_; // 本节点所在的通信域ID
    const u32 userRank_;       //  本节点的用户原始rank号
    const u32 userRankSize_;  // 本节点所在的用户通信域rank size
    RankInfo rankData_;           // 当前rank的相关信息
    const TopoType topoFlag_;     // 当前通信域内服务器间拓扑组合类型
    const DevType deviceType_;  // 当前rank所归属设备的类型
    // comm特性位图:0x1=支持inline-reduce;0x2=支持RDMA,1=RDMA,
    // 0=TCP;0x4=支持RDMA异步,前提是支持RDMA;Others=保留;

    const HcclDispatcher dispatcher_;  // signal调度句柄(event/notify机制)
    const std::unique_ptr<NotifyPool> &notifyPool_;
    std::map<HcclIpAddress, HcclNetDevCtx> &netDevCtxMap_;
    const bool isUsedRdmaOuter_;

    // 保存所有级别的通信rank关系, CommPlaneVector_[CommPlane][ringIndex]: 第CommPlane级 第ringIndex个环
    std::vector<std::vector<std::vector<RankInfo> > > CommPlaneVector_;
    // 子通信域内当前userrank是否为bridge rank的属性(多环)
    std::vector<bool> isBridgeVector_;

    // 通信域在当前superPod内, 按照serverIdx划分的所有rank信息
    std::map<u32, std::vector<RankInfo> > serverToRank_;
    // 通信域所有rank的信息, 按照superPodId -> RankInfo 的结构划分
    std::map<std::string, std::vector<RankInfo> > superPodToRank_;
    // 记录server内, 本rank和其他rank的连接关系
    std::map<s32, LinkTypeInServer> deviceLinkTypeMap_;

    // 8pring 多环的commouter 顺序
    std::vector<std::vector<u32> > multiOuterOrder_;
    // 整个通信域内rank的信息(直接调用exchanger生成，下标为userrank)
    std::vector<RankInfo> rankVector_;
    // 不同拓扑场景每个server上的设备数
    std::array<u32, static_cast<u32>(TopoType::TOPO_TYPE_RESERVED) > ranksOneNode_;

    NICDeployment nicDeployInner_;
    bool isHeterogComm_;
    bool isDiffAggregation_;
    const void* transportResourceInfoAddr_;
    size_t transportResourceInfoSize_;
    u32 meshAggregationRankSize_;
    u32 serverNum_;
    bool isHaveCpuRank_;
    // 复用Socket时, 复用SocketManager
    std::shared_ptr<HcclSocketManager> reusedSocketManager_;
    s32 deviceLogicId_;

    bool isUsedInterHccsMode_;
    bool useSdidForDeviceId_;
    std::unordered_map<std::string, std::map<u32, HcclIpAddress>> rankDevicePhyIdNicInfoMap_;
    std::vector<u32> ranksPort_;
    bool isSetHDCModeInfo_ { false };
    bool isUseRankPort_{ false };

    std::vector<std::vector<std::map<u32, u32>>> subCommRank2UserRank_;
    std::vector<std::vector<std::map<u32, u32>>> userRank2subCommRank_;
};
}  // namespace hccl

#endif /* COMM_FACTORY_PUB_H */
