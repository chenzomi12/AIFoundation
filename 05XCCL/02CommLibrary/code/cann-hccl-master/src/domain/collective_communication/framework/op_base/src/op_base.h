/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_BASE_H
#define OP_BASE_H

#include <vector>
#include <hccl/hccl.h>
#include <hccl/hccl_types.h>

#include "op_base_pub.h"
#include "hccl_comm_pub.h"
#include "config.h"
#include "../common/src/topo/topoinfo_detect.h"

using HcclOpInfoCtx = struct HcclInfoTag {
    HcclCommPtr pComm;
    hccl::HcclCommParams params;
    hccl::RankTable_t rankTable;
    bool cloudFlag;  // cloudFlag为0即实验室场景,cloudFlag为1则为云场景
    bool isUsed;
    std::mutex opGroupMapMutex;
    std::map<std::string, std::shared_ptr<hccl::hcclComm>> opGroup2CommMap;
    std::map<std::string, std::shared_ptr<hccl::TopoInfoDetect>> hcclCommTopoInfoDetectServer;
    std::map<std::string, std::shared_ptr<hccl::TopoInfoDetect>> hcclCommTopoInfoDetectAgent;
    HcclInfoTag() :isUsed(false) {}
};

HcclOpInfoCtx &GetHcclOpInfoCtx(void);

HcclResult InitOtherInfo(hccl::HcclCommParams &params, const char *rankTable);

HcclResult CallMsprofReportHostApi(hccl::hcclComm* hcclComm, HcclCMDType cmdType, uint64_t beginTime, u64 count,
    HcclDataType dataType, std::string tag);

HcclResult ReduceScatterLoop(const std::string &tag, void *inputPtr, void *outputPtr, const u64 &count,
    HcclDataType dataType, HcclReduceOp op, hccl::hcclComm *hcclComm, rtStream_t stream);

HcclResult HcclGetOpBasedMemSize(const HcclCMDType &opType, u64 &size,
    const HcomCollOpInfo &opInfo);

HcclResult ReduceLoop(const std::string &tag, void *inputPtr, void *outputPtr, const u64 count,
    HcclDataType dataType, HcclReduceOp op, const u32 root, hccl::hcclComm *hcclComm, rtStream_t stream);

HcclResult HcclAlltoAllVC(const void *sendBuf, const void *sendCountMatrix, HcclDataType sendType,
    const void *recvBuf, HcclDataType recvType, HcclComm comm, rtStream_t stream);

HcclResult HcclGatherAlltoAllV(HcomGatherAllToAllVParams params, HcclComm comm, aclrtStream stream);

HcclResult RunGather(u64 *sendCounts, u64 *sdispls, void *sendDevBuf, GatherPara &gatherPara);

void GatherMemCopyThread(void *baseAddr, u64 offset, std::vector<u64> &addrInfo, OpBaseMemPara memCpyPara);

HcclResult SetDefaultQosConfig(hccl::hcclComm *hcclComm);

HcclResult HcclGetCommAll(uint32_t ndev, int32_t *devices, HcclComm *comms);

HcclResult GetDeviceComm(uint32_t ndev, const HcclRootInfo &rootHandle, const s32 rank, const s32 logicDeviceId,
    HcclComm &comm);

HcclResult SetOverFlowAddr(hccl::hcclComm *hcclComm);

HcclResult HcclGetCommHandle(const char *commName, std::shared_ptr<hccl::hcclComm> &comm);

HcclResult GetPairWiseList(HcclSendRecvItem *sendRecvInfo, u32 itemNum, u32 rankSize, u32 rankId,
    std::vector<HcclSendRecvItem *> &orderedList);
HcclResult CheckScatterInputPara(uint64_t recvCount, HcclComm comm, void *recvBuf);

HcclResult HcclCreateComResourceByComm(HcclComm comm, u32 streamMode, bool isOpbaseMode,
    void** commContext);

HcclResult HcclDeviceRefresh(void);

HcclResult HcclSetIfProfile(void);
#endif  // OP_BASE_H