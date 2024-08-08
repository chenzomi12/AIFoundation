/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "comm_halving_doubling.h"

namespace hccl {
CommHalvingDoubling::CommHalvingDoubling(const std::string &collectiveId,
                                         const u32 userRank, const u32 userRankSize,
                                         const u32 rank, const u32 rankSize, const TopoType topoFlag,
                                         const HcclDispatcher dispatcher,
                                         const std::unique_ptr<NotifyPool> &notifyPool,
                                         std::map<HcclIpAddress, HcclNetDevCtx> &netDevCtxMap,
                                         const IntraExchanger &exchanger,
                                         const std::vector<RankInfo> paraVector,
                                         const DeviceMem& inputMem, const DeviceMem& outputMem,
                                         const bool isUsedRdmaOuter,
                                         const void* transportResourceInfoAddr, size_t transportResourceInfoSize,
                                         const std::string &tag, const NICDeployment nicDeployInner,
                                         const u32 subUserRankRoot, HalvingDoublingType halvingDoublingType,
                                         const bool isHaveCpuRank, const bool useSdidForDeviceId)
    : CommBase(collectiveId, userRank, userRankSize, rank, rankSize, paraVector, topoFlag,
               dispatcher, notifyPool, netDevCtxMap, exchanger, inputMem, outputMem, isUsedRdmaOuter,
               transportResourceInfoAddr, transportResourceInfoSize, tag, nicDeployInner, false, false, false,
               INVALID_UINT, isHaveCpuRank, useSdidForDeviceId),
      subUserRankRoot_(subUserRankRoot), halvingDoublingType_(halvingDoublingType)
{
}

CommHalvingDoubling::~CommHalvingDoubling()
{
}

HcclResult CommHalvingDoubling::CalcLink()
{
    u32 subRoot = INVALID_VALUE_RANKID;
    HcclResult ret = HCCL_SUCCESS;
    if (subUserRankRoot_ != INVALID_VALUE_RANKID) {
        CHK_RET(GetRankByUserRank(subUserRankRoot_, subRoot));
    }

    std::vector<bool> linkRelation = ExecutorBase::CalcLinksRelation(rank_, rankSize_, subRoot,
                                                                     halvingDoublingType_);
    if (linkRelation.size() == 0) {
        HCCL_ERROR("[Calc][Link]comm halving doubling calculate link relation failed!");
        return HCCL_E_INTERNAL;
    }

    for (u32 dstRank = 0; dstRank < rankSize_; dstRank++) {
        if (!linkRelation[dstRank]) {
            HCCL_DEBUG("comm halving doubling rank[%u] don't need create link with dst_rank[%u]", rank_, dstRank);
            continue;
        }

        if (rank_ < dstRank) {
            ret = CalcLinksNum(MachineType::MACHINE_SERVER_TYPE, dstRank);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Calc][Link]comm halving doubling calc links num failed, type[%d], "\
                "dstRank[%u]", static_cast<int32_t>(MachineType::MACHINE_SERVER_TYPE), dstRank), ret);
        }

        if (rank_ > dstRank) {
            ret = CalcLinksNum(MachineType::MACHINE_CLIENT_TYPE, dstRank);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Calc][Link]comm halving doubling calc links num failed, type[%d], "\
                "dstRank[%u]", static_cast<int32_t>(MachineType::MACHINE_CLIENT_TYPE), dstRank), ret);
        }
    }

    return HCCL_SUCCESS;
}
bool CommHalvingDoubling::NeedDataReceivedAck()
{
    HCCL_INFO("having-doubling need make a comm supportting DataReceivedAck");
    return true;
}
}  // namespace hccl
