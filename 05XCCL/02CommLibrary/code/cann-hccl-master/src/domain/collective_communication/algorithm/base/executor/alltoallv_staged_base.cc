/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alltoallv_staged_base.h"
#include "log.h"

namespace hccl {
using namespace std;

AlltoAllVStagedBase::AlltoAllVStagedBase(const HcclDispatcher dispatcher, Stream &stream)
    : dispatcher_(dispatcher), mainStream_(stream) {}

HcclResult AlltoAllVStagedBase::LocalCopy(u32 rank)
{
    if (sendAddrInfo_.find(rank) == sendAddrInfo_.end()) {
        HCCL_ERROR("[AlltoAllVStaged][LocalCopy] Invalid Rank[%u]", rank);
        return HCCL_E_INTERNAL;
    }

    for (auto it = sendAddrInfo_[rank].begin(); it != sendAddrInfo_[rank].end(); it++) {
        u8 *dstAddr = static_cast<u8 *>(recvMem_.ptr()) + it->remoteOffset;
        u64 destMax = it->remoteLength;
        u8 *srcAddr = static_cast<u8 *>(sendMem_.ptr()) + it->localOffset;
        u64 size = it->localLength;
        DeviceMem dst = DeviceMem::create(dstAddr, destMax);
        DeviceMem src = DeviceMem::create(srcAddr, size);

        HCCL_DEBUG(
            "[AlltoAllVStagedPairwise][LocalCopy]: Rank[%u] destAddr[%p], destMax[%llu], srcAddr[%p], size[%llu]", rank,
            dstAddr, destMax, srcAddr, size);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, mainStream_));
    }

    return HCCL_SUCCESS;
}
} // namespace hccl