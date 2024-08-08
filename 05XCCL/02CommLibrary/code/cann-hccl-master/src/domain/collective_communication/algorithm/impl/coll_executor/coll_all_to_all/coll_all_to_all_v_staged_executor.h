/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef COLL_RUN_ALLTOALLV_TWO_LEVEL_PIPELINE_H
#define COLL_RUN_ALLTOALLV_TWO_LEVEL_PIPELINE_H
#include "coll_all_to_all_executor.h"
namespace hccl {
class CollRunAlltoAllVStaged : public CollAlltoAllExecutor {

public:
    CollRunAlltoAllVStaged(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher);
    ~CollRunAlltoAllVStaged() = default;

    bool CheckNeedRecreateComm(u64 lastScratchMemSize) override;
    HcclResult CheckNeedCreateVirtualLinks(AlgResourceRequest &resourceRequest) override;
    HcclResult ParallelTaskLoaderProcess(const std::string &tag, Stream &stream, SubCommInfo &outerCommInfo,
        std::vector<Stream> &ringStreams);

private:

    HcclResult CalcStreamNum(u32& streamNum) override;
    void CalcWorkSpaceMemSize(const AlltoAllUserRankInfo &userRankInfo,
        const std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo, u64 &workspaceMemSize,
        u32 meshAggregationRankSize);
    HcclResult CalcScratchMemSize(u64& scratchMemSize) override;

    HcclResult CalcLevel0CommInfo(TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcLevel1CommInfo(TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcLevel2CommInfo(TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalStagedAlltoallVCommInfo(TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport);
    HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult KernelRun(const OpParam &param, ExecMem &execMem) override;

    HcclResult PrepareAlltoAllVStaged1(DeviceMem &sendBuf, DeviceMem &recvBuf, DeviceMem &scratchMem,
        std::map<u32, std::list<OneSendRecvAddrInfo>> &sendAddrInfosIntra,
        std::map<u32, std::list<OneSendRecvAddrInfo>> &recvAddrInfosIntra,
        Stream &stream, const std::string &tag, std::unique_ptr<AlltoAllVStagedBase> &alltoallOuter,
        ExecMem &execMem);
    void CalcInterMeshAggregationRecvRemoteOffset(const AlltoAllUserRankInfo &userRankInfo,
        const std::vector<SendRecvInfo> &allSendRecvInfo, u32 index, u64 &remoteOffset, u32 meshAggregationRankSize);
    void CalcInterMeshAggregationAlltoAllMemInfo(
        const AlltoAllUserRankInfo &userRankInfo, const std::vector<SendRecvInfo> &allSendRecvInfo,
        std::map<u32, std::list<OneSendRecvAddrInfo>> &sendAddrInfosInter,
        std::map<u32, std::list<OneSendRecvAddrInfo>> &recvAddrInfosInter,
        u32 meshAggregationRankSize);
    HcclResult PrepareAlltoAllVStaged2(DeviceMem &recvBuf, DeviceMem &scratchMem,
        std::map<u32, std::list<OneSendRecvAddrInfo>> &sendAddrInfosInter,
        std::map<u32, std::list<OneSendRecvAddrInfo>> &recvAddrInfosInter,
        Stream &stream, const std::string &tag, std::unique_ptr<AlltoAllVStagedBase> &alltoallInner,
        ExecMem &execMem);
};

} // namespace hccl

#endif