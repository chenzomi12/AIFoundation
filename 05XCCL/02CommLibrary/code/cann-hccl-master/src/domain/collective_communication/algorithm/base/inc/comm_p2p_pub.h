/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMM_P2P_PUB_H
#define COMM_P2P_PUB_H

#include "comm_base_pub.h"

namespace hccl {
class CommP2P : public CommBase {
public:
    explicit CommP2P(const std::string &collectiveId, const u32 userRank,
                      const u32 userRankSize, const u32 rank, const u32 rankSize, const TopoType topoFlag,
                      const HcclDispatcher dispatcher, const std::unique_ptr<NotifyPool> &notifyPool,
                      std::map<HcclIpAddress, HcclNetDevCtx> &netDevCtxMap,
                      const IntraExchanger &exchanger, const std::vector<RankInfo> paraVector,
                      const DeviceMem& inputMem, const DeviceMem& outputMem, const bool isUsedRdmaOuter,
                      const void* transportResourceInfoAddr, size_t transportResourceInfoSize,
                      const std::string &tag = "",
                      const u32 dstUserRank = INVALID_VALUE_RANKID,
                      const NICDeployment nicDeployInner = NICDeployment::NIC_DEPLOYMENT_DEVICE,
                      const bool isHaveCpuRank = false, const bool useSdidForDeviceId = false);

    ~CommP2P() override;

protected:
    // 计算当前rank与其他rank之间的link个数:server/client两种角色,P2P需要派生类实现
    HcclResult CalcLink() override;

private:
    const u32 dstUserRank_;
};
}  // namespace hccl

#endif /* COMM_P2P_PUB_H */
