/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "calc_hd_transport_req.h"

namespace hccl {
CalcHDTransportReq::CalcHDTransportReq(std::vector<std::vector<u32>> &subCommPlaneVector,
    std::vector<bool> &isBridgeVector, u32 userRank)
    : CalcTransportReqBase(subCommPlaneVector, isBridgeVector, userRank)
{
}

CalcHDTransportReq::~CalcHDTransportReq()
{
}

HcclResult CalcHDTransportReq::CalcTransportRequest(const std::string &tag, TransportMemType inputMemType,
    TransportMemType outputMemType, const CommParaInfo &commParaInfo,
    std::vector<SingleSubCommTransport> &commTransport, u32 subUserRankRoot)
{
    u32 ringSize = subCommPlaneVector_.size();
    commTransport.resize(ringSize);

    for (u32 ringIndex = 0; ringIndex < ringSize; ringIndex++) {
        if (commParaInfo.commPlane == COMM_LEVEL1 && !isBridgeVector_[ringIndex]) {
            continue; // 跳出本次循环
        }

        u32 rank = GetSubCollectiveRank(subCommPlaneVector_[ringIndex]);
        if (rank == INVALID_VALUE_RANKID) {
            continue;
        }

        u32 rankSize = subCommPlaneVector_[ringIndex].size();
        SingleSubCommTransport &subCommTransport = commTransport[ringIndex];
        subCommTransport.transportRequests.resize(rankSize);
        // 只有一张卡时不需要建链
        if (rankSize == HCCL_RANK_SIZE_EQ_ONE) {
            HCCL_INFO("comm base needn't to create links, rankSize_[%u].", rankSize);
            return HCCL_SUCCESS;
        }
    
        u32 subRoot = INVALID_VALUE_RANKID;
        if (subUserRankRoot != INVALID_VALUE_RANKID) {
            CHK_RET(GetRankByUserRank(subCommPlaneVector_[ringIndex], subUserRankRoot, subRoot));
        }

        std::vector<bool> linkRelation =  ExecutorBase::CalcLinksRelation(rank, rankSize, subRoot,
            HalvingDoublingType::RECURSIVE_HALVING_DOUBLING);

        for (u32 rankIndex = 0; rankIndex < rankSize; rankIndex++) {
            TransportRequest &tmpTransport = subCommTransport.transportRequests[rankIndex];
            if (linkRelation[rankIndex] == true) {
                tmpTransport.isValid = true;
                tmpTransport.localUserRank  = userRank_;
                tmpTransport.remoteUserRank = subCommPlaneVector_[ringIndex][rankIndex];
                tmpTransport.inputMemType = inputMemType;
                tmpTransport.outputMemType = outputMemType;
                HCCL_INFO("[CommFactory][CalcHDCommInfo] param_.tag[%s] ringIndex[%u], localRank[%u], "\
                    "remoteRank[%u], inputMemType[%d], outputMemType[%d]", tag.c_str(), ringIndex, userRank_,
                    tmpTransport.remoteUserRank, inputMemType, outputMemType);
            } else {
                tmpTransport.isValid = false;
            }
        }
        subCommTransport.supportDataReceivedAck = true;
    }
    return HCCL_SUCCESS;
}

}  // namespace hccl