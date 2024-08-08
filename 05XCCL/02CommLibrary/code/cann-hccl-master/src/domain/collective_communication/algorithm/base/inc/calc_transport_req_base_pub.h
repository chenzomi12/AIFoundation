/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CALC_TRANSPORT_REQ_BASE_PUB_H
#define CALC_TRANSPORT_REQ_BASE_PUB_H

#include "hccl/base.h"
#include <hccl/hccl_types.h>
#include <vector>
#include "coll_alg_param.h"
#include "comm_factory_pub.h"

namespace hccl {
class CalcTransportReqBase {
public:
    explicit CalcTransportReqBase(std::vector<std::vector<u32>> &subCommPlaneVector,
        std::vector<bool> &isBridgeVector, u32 userRank);

    virtual ~CalcTransportReqBase();

    virtual HcclResult CalcTransportRequest(const std::string &tag, TransportMemType inputMemType,
        TransportMemType outputMemType, const CommParaInfo &commParaInfo,
        std::vector<SingleSubCommTransport> &commTransport, u32 subUserRankRoot = INVALID_VALUE_RANKID);

protected:
    // 获取本rank在子通信域(多平面)内当前平面的rank号
    const u32 GetSubCollectiveRank(const std::vector<u32> &vecPara) const;
    HcclResult GetRankByUserRank(const std::vector<u32> &vecPara, const u32 userRank, u32 &rank) const;

    const std::vector<std::vector<u32>> &subCommPlaneVector_;
    const std::vector<bool> &isBridgeVector_;
    const u32 userRank_;
};
}  // namespace hccl
#endif /* CALC_TRANSPORT_REQ_BASE_PUB_H */