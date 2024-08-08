/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "comm_p2p.h"

namespace hccl {
CommP2P::CommP2P(const std::string &collectiveId, const u32 userRank,
    const u32 userRankSize, const u32 rank, const u32 rankSize, const TopoType topoFlag,
    const HcclDispatcher dispatcher, const std::unique_ptr<NotifyPool> &notifyPool,
    std::map<HcclIpAddress, HcclNetDevCtx> &netDevCtxMap,
    const IntraExchanger &exchanger, const std::vector<RankInfo> paraVector,
    const DeviceMem& inputMem, const DeviceMem& outputMem, const bool isUsedRdmaOuter,
    const void* transportResourceInfoAddr, size_t transportResourceInfoSize,
    const std::string &tag, const u32 dstUserRank, const NICDeployment nicDeployInner,
    const bool isHaveCpuRank, const bool useSdidForDeviceId)
    : CommBase(collectiveId, userRank, userRankSize, rank, rankSize, paraVector, topoFlag, dispatcher, notifyPool,
      netDevCtxMap, exchanger, inputMem, outputMem, isUsedRdmaOuter, transportResourceInfoAddr,
      transportResourceInfoSize, tag, nicDeployInner, false, false, false, INVALID_UINT, isHaveCpuRank,
      useSdidForDeviceId),
      dstUserRank_(dstUserRank)
{
}

CommP2P::~CommP2P()
{
}

HcclResult CommP2P::CalcLink()
{
    u32 dstRank = INVALID_VALUE_RANKID;
    HcclResult ret = GetRankByUserRank(dstUserRank_, dstRank);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Calc][Link]comm p2p dstRank[%u] is invalid", dstRank), HCCL_E_PARA);

    if (paraVector_.size() <= dstRank) {
        HCCL_ERROR("[Calc][Link]comm p2p dstRank[%u] is bigger than para_vector size[%llu]",
            dstRank, paraVector_.size());
        return HCCL_E_PARA;
    }

    if (paraVector_.size() <= rank_) {
        HCCL_ERROR("[Calc][Link]comm p2p rank[%u] is bigger than para_vector size[%llu]", rank_, paraVector_.size());
        return HCCL_E_PARA;
    }

    /* 校验建链可行性:
    服务器间devid满足:基于8取模值相同(devid相等) */
    bool isSupportP2P = false;
    if ((paraVector_[rank_].deviceType == DevType::DEV_TYPE_910) ||
        (paraVector_[rank_].deviceType == DevType::DEV_TYPE_910B) ||
        (paraVector_[rank_].deviceType == DevType::DEV_TYPE_910_73) ||
        (paraVector_[rank_].deviceType == DevType::DEV_TYPE_NOSOC)) {
        isSupportP2P = true;
    } else if (paraVector_[rank_].deviceType == DevType::DEV_TYPE_310P3) {
        isSupportP2P = (paraVector_[rank_].serverId == paraVector_[dstRank].serverId);
    } else {
        HCCL_ERROR("[Calc][Link]invalid device type[%d]", paraVector_[rank_].deviceType);
        return HCCL_E_PARA;
    }
    if (!isSupportP2P) {
        HCCL_ERROR("[Calc][Link]comm p2p rank[%u] devid[%d] serverId[%s] and dstRank[%u] devid[%d] serverId[%s] is "\
            "not support P2P", rank_, paraVector_[rank_].devicePhyId, paraVector_[rank_].serverId.c_str(),\
            dstRank, paraVector_[dstRank].devicePhyId, paraVector_[dstRank].serverId.c_str());
        return HCCL_E_PARA;
    }

    if (rank_ < dstRank) {
        ret = CalcLinksNum(MachineType::MACHINE_SERVER_TYPE, dstRank);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Calc][Link]comm p2p calc links num failed, type[%d], dstRank[%u]",
            static_cast<int32_t>(MachineType::MACHINE_SERVER_TYPE), dstRank), ret);
    } else if (rank_ > dstRank) {
        ret = CalcLinksNum(MachineType::MACHINE_CLIENT_TYPE, dstRank);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Calc][Link]comm p2p calc links num failed, type[%d], dstRank[%u]",
            static_cast<int32_t>(MachineType::MACHINE_CLIENT_TYPE), dstRank), ret);
    } else {
        HCCL_ERROR("[Calc][Link]comm p2p dstRank_[%u] is not support to creat link with itself", dstRank);
        return HCCL_E_PARA;
    }

    return HCCL_SUCCESS;
}
}  // namespace hccl
