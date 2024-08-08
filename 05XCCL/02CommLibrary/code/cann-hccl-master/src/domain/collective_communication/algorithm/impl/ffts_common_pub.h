/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FFTS_COMMON_PUB_H
#define FFTS_COMMON_PUB_H

#include <cstdint>
#include <map>
#include <vector>
#include <memory>

#include "base.h"
#include "adapter_rts_common.h"
#include "hccl_common.h"
#include "device_capacity.h"

namespace hccl {
enum class ReduceType {
    INLINE_REDUCE = 0,
    TBE_REDUCE
};

enum class CopyPattern {
    ZCOPY = 0,
    BCOPY
};

using HcclOpMetaInfo = struct HcclOpMetaInfoDef {
    HcclCMDType opType = HcclCMDType::HCCL_CMD_INVALID;
    bool isRootRank = false;
    bool isSmallCount = false;
    uint32_t rootRank = INVALID_UINT;
    ReduceType reduceType = ReduceType::INLINE_REDUCE;
    CopyPattern copyPattern = CopyPattern::BCOPY;
    u64 alltoallvSendDataSize = 0;
    u64 alltoallvcSendDataSize = 0;
    u32 piplineSliceNum = 1;
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    u32 algolevel1Type = 0;
    bool hugeData = false;
    u64 sliceNum = 1;
    u32 dataSplit = 0;
    bool isAivMode = false;
    bool isEnableCache = true;

    static bool CheckEnableCache(const HcclOpMetaInfoDef &opMetaInfo)
    {
        if (opMetaInfo.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV || opMetaInfo.opType == HcclCMDType::HCCL_CMD_SEND ||
            opMetaInfo.opType == HcclCMDType::HCCL_CMD_RECEIVE || opMetaInfo.hugeData) {
            return false;
        }

        if (opMetaInfo.alltoallvSendDataSize > RDMA_SEND_MAX_SIZE ||
            opMetaInfo.alltoallvcSendDataSize > RDMA_SEND_MAX_SIZE) {
            return false;
        }
        return !((opMetaInfo.opType == HcclCMDType::HCCL_CMD_ALLTOALLV ||
            opMetaInfo.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) &&
            (opMetaInfo.copyPattern == CopyPattern::BCOPY));
    }

    static HcclOpMetaInfoDef GetOneForAllReduce(u32 algolevel1Type = 0,
        HcclDataType dataType = HCCL_DATA_TYPE_RESERVED, ReduceType reduceType = ReduceType::INLINE_REDUCE,
        bool isSmallCount = false, u32 piplineSliceNum = 1, bool hugeData = false,
        CopyPattern copyPattern = CopyPattern::BCOPY, u64 sliceNum = 1,
        bool isAivMode = false)
    {
        HcclOpMetaInfoDef meta;
        meta.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
        meta.copyPattern = copyPattern;
        meta.reduceType = reduceType;
        meta.dataType = dataType;
        meta.isSmallCount = isSmallCount;
        meta.piplineSliceNum = piplineSliceNum;
        meta.algolevel1Type = algolevel1Type;
        meta.hugeData = hugeData;
        meta.sliceNum = sliceNum;
        meta.isAivMode = isAivMode;
        meta.isEnableCache = CheckEnableCache(meta);
        return meta;
    }

    static HcclOpMetaInfoDef GetOneForAllGather(u32 algolevel1Type = 0, bool hugeData = false,
        CopyPattern copyPattern = CopyPattern::BCOPY)
    {
        HcclOpMetaInfoDef meta;
        meta.opType = HcclCMDType::HCCL_CMD_ALLGATHER;
        meta.copyPattern = copyPattern;
        meta.algolevel1Type = algolevel1Type;
        meta.hugeData = hugeData;
        meta.isEnableCache = CheckEnableCache(meta);
        return meta;
    }

    static HcclOpMetaInfoDef GetOneForBroadcast(bool isRootRank, uint32_t rootRank,
        bool hugeData = false, bool isSmallCount = false)
    {
        HcclOpMetaInfoDef meta;
        meta.opType = HcclCMDType::HCCL_CMD_BROADCAST;
        meta.isSmallCount = isSmallCount;             // 是否小数据
        meta.isRootRank = isRootRank;
        meta.rootRank = rootRank;
        meta.hugeData = hugeData;
        meta.isEnableCache = CheckEnableCache(meta);
        return meta;
    }

    static HcclOpMetaInfoDef GetOneForScatter(uint32_t rootRank, bool hugeData = false)
    {
        HcclOpMetaInfoDef meta;
        meta.opType = HcclCMDType::HCCL_CMD_SCATTER;
        meta.rootRank = rootRank;
        meta.hugeData = hugeData;
        meta.isEnableCache = CheckEnableCache(meta);
        return meta;
    }

    static HcclOpMetaInfoDef GetOneForReduceScatter(
        u32 algolevel1Type = 0, HcclDataType dataType = HCCL_DATA_TYPE_RESERVED,
        ReduceType reduceType = ReduceType::INLINE_REDUCE, bool hugeData = false,
        bool isSmallCount = false, CopyPattern copyPattern = CopyPattern::BCOPY)
    {
        HcclOpMetaInfoDef meta;
        meta.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;
        meta.reduceType = reduceType;
        meta.dataType = dataType;
        meta.algolevel1Type = algolevel1Type;
        meta.hugeData = hugeData;
        meta.isSmallCount = isSmallCount; // 是否小数据
        meta.copyPattern = copyPattern;
        meta.isEnableCache = CheckEnableCache(meta);
        return meta;
    }

    static HcclOpMetaInfoDef GetOneForAllToAllV(CopyPattern copyPattern, u64 dataSize, bool hugeData = false)
    {
        HcclOpMetaInfoDef meta;
        meta.opType = HcclCMDType::HCCL_CMD_ALLTOALLV;
        meta.copyPattern = copyPattern;
        meta.alltoallvSendDataSize = dataSize;
        meta.hugeData = hugeData;
        meta.isEnableCache = CheckEnableCache(meta);
        return meta;
    }

    static HcclOpMetaInfoDef GetOneForAllToAllVC(CopyPattern copyPattern, u64 dataSize, bool hugeData = false)
    {
        HcclOpMetaInfoDef meta;
        meta.opType = HcclCMDType::HCCL_CMD_ALLTOALLVC;
        meta.copyPattern = copyPattern;
        meta.alltoallvcSendDataSize = dataSize;
        meta.hugeData = hugeData;
        meta.isEnableCache = CheckEnableCache(meta);
        return meta;
    }

    static HcclOpMetaInfoDef GetOneForSend()
    {
        HcclOpMetaInfoDef meta;
        meta.opType = HcclCMDType::HCCL_CMD_SEND;
        meta.isEnableCache = CheckEnableCache(meta);
        return meta;
    }

    static HcclOpMetaInfoDef GetOneForRecieve()
    {
        HcclOpMetaInfoDef meta;
        meta.opType = HcclCMDType::HCCL_CMD_RECEIVE;
        meta.isEnableCache = CheckEnableCache(meta);
        return meta;
    }

    static HcclOpMetaInfoDef GetOneForBatchSendRecv()
    {
        HcclOpMetaInfoDef meta;
        meta.opType = HcclCMDType::HCCL_CMD_BATCH_SEND_RECV;
        meta.isEnableCache = CheckEnableCache(meta);
        return meta;
    }

    static HcclOpMetaInfoDef GetOneForReduce(bool isRootRank, uint32_t rootRank, u32 algolevel1Type = 0,
        HcclDataType dataType = HCCL_DATA_TYPE_RESERVED, ReduceType reduceType = ReduceType::INLINE_REDUCE,
        bool hugeData = false, CopyPattern copyPattern = CopyPattern::BCOPY)
    {
        HcclOpMetaInfoDef meta;
        meta.opType = HcclCMDType::HCCL_CMD_REDUCE;
        meta.isRootRank = isRootRank;
        meta.rootRank = rootRank;
        meta.reduceType = reduceType;
        meta.dataType = dataType;
        meta.algolevel1Type = algolevel1Type;
        meta.hugeData = hugeData;
        meta.copyPattern = copyPattern;
        meta.isEnableCache = CheckEnableCache(meta);
        return meta;
    }

    std::string GetCacheKey() const
    {
        std::string isRootRankStr = isRootRank ? "1" : "0";
        std::string isSmallCountStr = isSmallCount ? "1" : "0";
        std::string dataSplitStr = dataSplit ? "1" : "0";
        std::string isAivModeStr = isAivMode ? "1" : "0";
        return std::to_string(static_cast<int>(opType)) + isRootRankStr + std::to_string(static_cast<int>(reduceType)) +
               std::to_string(rootRank) + std::to_string(sliceNum) + std::to_string(static_cast<int>(dataType)) +
               isSmallCountStr + std::to_string(piplineSliceNum) + std::to_string(algolevel1Type) +
               std::to_string(static_cast<int>(copyPattern)) + dataSplitStr + isAivModeStr;
    }
};
}
#endif // !FFTS_COMMON_H