/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_scatter_comm_executor.h"

namespace hccl {
CollScatterCommExecutor::CollScatterCommExecutor(const HcclDispatcher dispatcher,
                                std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollScatterExecutor(dispatcher, topoMatcher)
{
}

HcclResult CollScatterCommExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcCombinedCommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollScatterCommExecutor::CalcCombinedCommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaInfo(COMM_COMBINE, CommType::COMM_TAG_MAX);
    if (UseInterServerNHRAlgo(algType_)) {
        commParaInfo.commType = CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING;
    } else if (UseInterServerNBAlgo(algType_)) {
        commParaInfo.commType = CommType::COMM_TAG_NONUNIFORM_BRUCK;
    } else {
        commParaInfo.commType = CommType::COMM_TAG_RING_INNER;
    }
    CHK_RET(CalcCommPlaneInfo(tag_, commParaInfo, opTransport[COMM_COMBINE], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollScatterCommExecutor::CalcStreamNum(u32& streamNum)
{
    // 只传递从流数量
    streamNum = 0;
    HCCL_INFO("[CollScatterCommExecutor][CalcStreamNum]tag[%s] streamNum_ is [%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollScatterCommExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    DeviceMem& inputMem = execMem.inputMem;
    DeviceMem& outputMem = execMem.outputMem;
    u64 count = execMem.count;
    auto root = param.root;
    auto dataType = param.DataDes.dataType;
    Stream& stream = const_cast<Stream&>(param.stream);
    u32 userRank = topoAttr_.userRank;

    u32 commIndex = COMM_INDEX_0;
    // 统一走server间
    CHK_RET(CheckCommSize(COMM_COMBINE, commIndex + 1));
    SubCommInfo combinedCommInfo = GetSubCommInfo(COMM_COMBINE, 0);

    CHK_RET(KernelRunInner(inputMem, count, dataType, commIndex, root, userRank, COMM_COMBINE, stream));

    // 将scratchMem赋值给outputMem
    u8 *inputMemPtr = static_cast<u8 *>(inputMem.ptr());
    DeviceMem resultMem(inputMemPtr + outputMem.size() * combinedCommInfo.localRank, outputMem.size());
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, outputMem, resultMem, stream));
    return HCCL_SUCCESS;
}

REGISTER_EXEC("ScatterCommExecutor", ScatterComm, CollScatterCommExecutor);

}
