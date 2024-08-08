/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include "calc_partial_mesh_transport_req.h"
#include "dtype_common.h"

namespace hccl {
CalcPartialMeshTransportReq::CalcPartialMeshTransportReq(std::vector<std::vector<u32>> &subCommPlaneVector,
    std::vector<bool> &isBridgeVector, u32 userRank)
    : CalcTransportReqBase(subCommPlaneVector, isBridgeVector, userRank)
{
}

CalcPartialMeshTransportReq::~CalcPartialMeshTransportReq()
{
}

HcclResult CalcPartialMeshTransportReq::CalcTransportRequest(const std::string &tag, TransportMemType inputMemType,
    TransportMemType outputMemType, const CommParaInfo &commParaInfo,
    std::vector<SingleSubCommTransport> &commTransport, u32 subUserRankRoot)
{
    // send/recv分别使用一个comm
    u32 ringSize = 2;
    commTransport.resize(ringSize);

    for (u32 ringIndex = 0; ringIndex < ringSize; ringIndex++) {
        if (commParaInfo.commPlane == COMM_LEVEL1 && !isBridgeVector_.empty() && !isBridgeVector_[0]) {
            continue; // 跳出本次循环
        }
        CHK_PRT_RET(subCommPlaneVector_.empty(), HCCL_ERROR("[CalcPartialMeshTransportReq][CalcTransportRequest] "\
            "the vector named subCommPlaneVector_ is empty."), HCCL_E_NOT_FOUND);
        u32 rank = GetSubCollectiveRank(subCommPlaneVector_[0]);
        if (rank == INVALID_VALUE_RANKID) {
            continue;
        }

        u32 rankSize = subCommPlaneVector_[0].size();
        SingleSubCommTransport &subCommTransport = commTransport[ringIndex];
        subCommTransport.transportRequests.resize(rankSize);
        // 只有一张卡时不需要建链
        if (rankSize == HCCL_RANK_SIZE_EQ_ONE) {
            HCCL_INFO("comm base needn't to create links, rankSize_[%u].", rankSize);
            return HCCL_SUCCESS;
        }

        for (u32 rankIndex = 0; rankIndex < rankSize; rankIndex++) {
            TransportRequest &tmpTransport = subCommTransport.transportRequests[rankIndex];
            auto it = commParaInfo.batchSendRecvtargetRanks.find(subCommPlaneVector_[0][rankIndex]);
            if (rankIndex != rank && it != commParaInfo.batchSendRecvtargetRanks.end()) {
                tmpTransport.isValid = true;
                tmpTransport.localUserRank  = userRank_;
                tmpTransport.remoteUserRank = subCommPlaneVector_[0][rankIndex];
                tmpTransport.inputMemType = inputMemType;
                tmpTransport.outputMemType = outputMemType;
                HCCL_INFO("[CommFactory][CalcPartialMeshCommInfo] param_.tag[%s] ringIndex[%u], localRank[%u], "\
                    "remoteRank[%u], inputMemType[%d], outputMemType[%d]", tag.c_str(), ringIndex, userRank_,
                    tmpTransport.remoteUserRank, inputMemType, outputMemType);
            } else {
                tmpTransport.isValid = false;
                continue;
            }
        }
    }
    return HCCL_SUCCESS;
}

}  // namespace hccl