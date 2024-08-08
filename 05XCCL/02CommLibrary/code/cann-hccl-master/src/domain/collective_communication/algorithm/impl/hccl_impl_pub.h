/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_IMPL_PUB_H
#define HCCL_IMPL_PUB_H

#include "comm_base_pub.h"
#include "threadManage.h"
#include "heartbeat_pub.h"

namespace hccl {
constexpr u32 OUTER_BRIDGE_RANK_ID = 0;

constexpr s32 PROF_RANKSIZE_OFFSET_OF_PLANEID = 16;
constexpr s32 PROF_RINGINDEX_OFFSET_OF_PLANEID = 28;

constexpr u32 PROF_STAGE_0 = 0;
constexpr u32 PROF_STAGE_1 = 1;
constexpr u32 PROF_STAGE_2 = 2;

constexpr u8 DETERMINISTIC_CONFIG_DISABLE = 0;  // 不开启确定性计算
constexpr u8 DETERMINISTIC_CONFIG_ENABLE = 1;   // 开启确定性计算

using innerStreamInfo_t = struct InnerStreamInfo {
    u32 ringNum; /* 至少有1个ring */
    std::vector<std::shared_ptr<LocalNotify>> ringSignal;
    std::vector<std::shared_ptr<LocalNotify>> ringSignalAux;
    std::vector<Stream> ringStreams;
    std::vector<std::shared_ptr<ThreadManage>> ringThreadsManage;
    std::vector<uint32_t> tidInfo;
    std::vector<Stream> ringDeviceStreams;
    std::vector<std::shared_ptr<LocalNotify>> ringDeviceSignal;
    std::vector<std::shared_ptr<LocalNotify>> ringDeviceSignalAux;

    InnerStreamInfo() : ringNum(1)
    {
    }
};

using tagStreamInfo_t = std::map<std::string, InnerStreamInfo>;

using CommInfo = struct TagCommInfo {
    std::vector<std::unique_ptr<CommBase> > commInner;
    std::vector<std::unique_ptr<CommBase> > commInnerRdma;
    std::vector<std::unique_ptr<CommBase> > commOuter;
    std::vector<std::unique_ptr<CommBase> > commOuterRdma;
    std::vector<std::unique_ptr<CommBase> > commLevel2;
    std::vector<std::unique_ptr<CommBase> > commP2P;
    std::unique_ptr<CommBase> commIntraServer;

    TagCommInfo() : commInner(0), commInnerRdma(0), commOuter(0), commOuterRdma(0), commP2P(0), commIntraServer(nullptr)
    {
    }
};
using tagCommInfo_t = std::map<std::string, CommInfo>;

using HcclAlgoAttr = struct HcclAlgoAttrDef {
    bool isHaveCpuRank;              // 是否有cpu参与通信
    bool inlineReduceSwitchOn;       // 收到数量时同时完成Reduce计算
    bool isUsedRdmaOuter;            // Outer 通信域是否使用RDMA
    bool isUsedInterHccsMode;       // 超节点内节点间是否使用HCCS模式
    std::string identifier;
    std::string collectiveId;
    NICDeployment nicDeployment;
    WorkMode commWorkMode;

    HcclAlgoAttrDef()
        : isHaveCpuRank(false),
        inlineReduceSwitchOn(true),
        isUsedRdmaOuter(false),
        isUsedInterHccsMode(false),
        identifier(""),
        collectiveId(""),
        nicDeployment(NICDeployment::NIC_DEPLOYMENT_DEVICE),
        commWorkMode(WorkMode::HCCL_MODE_NORMAL)
    {}
};

using HcclTopoAttr = struct HcclTopoAttrDef {
    u32 serverNum;                   // 集群中总的服务器数
    u32 devNumInLevel2;                 // 集群中总的超节点数
    u32 moduleNum;                   // 集群中的总的module数
    u32 deviceNumPerServer;          // 服务器上的Device数量
    u32 deviceNumPerAggregation;     // 每个module中的Device数量
    bool multiModuleDiffDeviceNumMode; // 每个module内的设备数是否相等，如果不相同即为多module不同卡模式 （走大RING环）

    u32 meshAggregationRankSize;
    bool isDiffDeviceModule;
    bool isSingleMeshAggregation;
    bool isAllRankSamePlane;         // 通信域所有Rank是否在同一平面

    u32 userRank;                    // 通信域 RankID
    u32 realUserRank;
    u32 userRankSize;                // 通信域的 Rank数量
    std::vector<RankInfo> rankInfoList; // world group内rank的信息, 按照rank id递增依次排列
    std::vector<HbRankInfo> hbRankInfoList;  // world group内rank的信息, 按照rank id递增依次排列, 心跳模块使用

    u32 devicePhyId;
    s32 deviceLogicId;
    bool useSdidForDeviceId;    // 使用SDID作为DeviceId做相关查询操作

    DevType deviceType;
    bool isStandardCard;
    bool is310PDuoCard;

    std::vector<u32> nicList;
    std::unordered_map<u32, u32> pairLinkCounter; // server内所有device间的链路类型计数
    std::unordered_map<u32, std::unordered_map<int, std::vector<int>>> pairLinkInfo; // server内所有device间的链路类型
    bool isSupportRdmaLite;           // 是否支持rdma lite

    HcclTopoAttrDef()
        : serverNum(0),
        devNumInLevel2(0),
        moduleNum(0),
        deviceNumPerServer(0),
        deviceNumPerAggregation(0),
        multiModuleDiffDeviceNumMode(false),
        meshAggregationRankSize(0),
        isDiffDeviceModule(false),
        isSingleMeshAggregation(false),
        isAllRankSamePlane(false),
        userRank(0),
        realUserRank(0),
        userRankSize(0),
        rankInfoList(0),
        devicePhyId(0),
        deviceLogicId(0),
        useSdidForDeviceId(false),
        deviceType(DevType::DEV_TYPE_COUNT),
        isStandardCard(false),
        is310PDuoCard(false),
        nicList(0),
        pairLinkCounter(0),
        pairLinkInfo(0),
        isSupportRdmaLite(false)
    {}
};

}  // namespace hccl

#endif /** HCCL_IMPL_PUB_H */
