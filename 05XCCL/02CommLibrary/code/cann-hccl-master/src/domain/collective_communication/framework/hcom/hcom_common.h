/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOM_COMMOM_H
#define HCOM_COMMOM_H

#include "hccl_comm_pub.h"
namespace hccl {
    class HcclCommBase;
}

enum class RankInfoType {
    RANK_SIZE_IN_GROUP,
    RANK_ID_IN_GROUP,
    WORLD_RANK_ID_BY_GROUP,
    GROUP_RANK_ID_BY_WORLD,
    SERVER_NUM_IN_GROUP
};


constexpr u32 SINGLE_SERVER_NUM = 1;
using HcomOpTagInfo = struct HcomOpTagInfoCtx {
    std::map<std::string, u32> opIndex; // key: (group name) or (identifier), value: op index
};

using HcclGroupParams = struct TagHcclGroupParamsInfo {
    /* * group的基本构建信息，节点数及本节点在group中的编号、
    本节点在worldgroup中的编号、group的所有ranks */
    u32 worldRank;                /* * 用于标识world内不同节点 */
    u32 groupRank;                /* * 用于标识group内不同节点 */
    u32 serverNum;                /* * 用于标识group内服务器总数 */
    u32 totalRanks;              /* * 用于指示group内的节点总数, rank范围[0, totalRanks-1] */
    std::vector<u32> groupRanks;  // 内部存储wordrankid，其下标表示groupid
    HcclCommPtr pSubComm;
    std::shared_ptr<hccl::HcclCommBase> pSubCommBase;
    u32 refCounter = 0;
    bool destroyFlag = false;
};

using HcomInfo = struct HcomInfoTag {
    HcclCommPtr pComm;
    std::shared_ptr<hccl::HcclCommBase> pCommBase;
    void *psComm;
    hccl::HcclCommParams params;
    std::map<std::string, HcclGroupParams> hcomGroupMap;  // 每个group的信息(kname为服务器的server_id,按照服务器区分)
    std::mutex groupParamsLock;
    hccl::RankTable_t rankTable;
    s32 devId;
    bool cloudFlag;  // cloudFlag为0即实验室场景,cloudFlag为1则为云场景
    bool isHcomInit; // 标识是否为pytorch单算子通信域复用场景
    std::mutex backloggedGroupLock;
    std::map<std::string, std::vector<u32>> backloggedGroup;     // 待创建的group
    HcomInfoTag()
        :pComm(nullptr), devId(-1), cloudFlag(false), isHcomInit(false)
    {
    }
};

HcclResult HcomSetGroupTopoInfo(const char *group, uint32_t rankSize);
void HcomUnSetGroupTopoInfo(const char *group);
HcclResult HcomGetCommByGroup(const char *group, std::shared_ptr<hccl::hcclComm> &hcclComm);
void HcomTopoInfoFuncInstall(HcclResult (*p1)(const char *, uint32_t), void (*p2)(const char *));
HcclResult HcomGetTopoDesc(const char *group, HcclTopoDescs *topoDescs, uint32_t topoSize);
s32 HcclGetThreadDeviceId();
void HcomGroupCallbackFuncInstall(HcclResult (*p1)(const std::string &, const std::vector<u32> &),
    bool (*p2)(HcomInfo &), HcclResult (*p3)(const std::string &), HcclResult (*p4)(HcomInfo &));
HcclResult DestroyFlag(const char *group, bool flag);
HcclResult HcomQueryGroupRef(const char *group, u32 &groupRef);
bool HcomCheckrtMemcpyAddrAsync(void);
HcclResult HcomGetbackloggedByGroup(const char *group, std::vector<u32> &groupRanks, s32 &groupSize);

#ifdef __cplusplus
extern "C" {
#endif
bool HcomGetSecAddrCopyFlag(void);
bool HcomFindGroup(const char *group);
HcclResult HcomInitByFile(const char *rankTablePath, const char *identify);
#ifdef __cplusplus
}
#endif
#endif /* HCCL_COMM_PUB_H */
