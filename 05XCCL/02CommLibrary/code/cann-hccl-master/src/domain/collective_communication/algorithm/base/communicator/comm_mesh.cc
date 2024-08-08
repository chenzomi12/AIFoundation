/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "comm_mesh.h"

namespace hccl {
CommMesh::CommMesh(const std::string &collectiveId,
    const u32 userRank, const u32 userRankSize, const u32 rank, const u32 rankSize, const TopoType topoFlag,
    const HcclDispatcher dispatcher, const std::unique_ptr<NotifyPool> &notifyPool,
    std::map<HcclIpAddress, HcclNetDevCtx> &netDevCtxMap,
    const IntraExchanger &exchanger, const std::vector<RankInfo> paraVector,
    const DeviceMem &inputMem, const DeviceMem &outputMem, const bool isUsedRdmaOuter,
    const void* transportResourceInfoAddr, size_t transportResourceInfoSize, const std::string &tag,
    bool isAlltoAllCommMesh, const NICDeployment nicDeployInner,
    const bool useOneDoorbell, const bool isAicpuModeEn,
    const bool isHaveCpuRank, const bool useSdidForDeviceId): CommBase(collectiveId, userRank, userRankSize,
    rank, rankSize, paraVector, topoFlag, dispatcher, notifyPool, netDevCtxMap, exchanger, inputMem, outputMem,
    isUsedRdmaOuter, transportResourceInfoAddr, transportResourceInfoSize, tag, nicDeployInner,
    isAlltoAllCommMesh, useOneDoorbell, isAicpuModeEn, INVALID_UINT, isHaveCpuRank, useSdidForDeviceId)
{}

CommMesh::~CommMesh()
{
}

HcclResult CommMesh::CalcLink()
{
    // 原则:创建本rank与rank+n时的link，本rank为server端;创建本rank与rank-n时的link，本rank为client端;
    u32 dstClientRank = INVALID_VALUE_RANKID;
    u32 dstServerRank = INVALID_VALUE_RANKID;

    u32 backwardDst = rankSize_ - rank_ - HCCL_RANK_OFFSET;

    HcclResult ret = HCCL_SUCCESS;
    // 先创建dst_rank < rank_
    for (u32 clientIndex = 0; clientIndex < rank_; clientIndex++) {
        // rank 作为client
        dstServerRank = rank_ - clientIndex - HCCL_RANK_OFFSET;
        ret = CalcLinksNum(MachineType::MACHINE_CLIENT_TYPE, dstServerRank);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Calc][Link]comm mesh calc links num failed, type[%d], dstServerRank[%u]",
            static_cast<int32_t>(MachineType::MACHINE_CLIENT_TYPE), dstServerRank), ret);
    }

    // 再创建dst_rank > rank_
    for (u32 serverIndex = 0; serverIndex < backwardDst; serverIndex++) {
        // rank 作为server
        dstClientRank = rank_ + serverIndex + HCCL_RANK_OFFSET;
        ret = CalcLinksNum(MachineType::MACHINE_SERVER_TYPE, dstClientRank);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Calc][Link]comm mesh calc links num failed, type[%d], dstClientRank[%u]",
            static_cast<int32_t>(MachineType::MACHINE_SERVER_TYPE), dstClientRank), ret);
    }

    return HCCL_SUCCESS;
}
}  // namespace hccl
