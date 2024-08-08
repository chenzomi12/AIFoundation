/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXECUTOR_BASE_PUB_H
#define EXECUTOR_BASE_PUB_H

#include <cstring>
#include <vector>
#include <memory>
#include <list>
#include "hccl/base.h"
#include "externalinput_pub.h"
#include "mem_device_pub.h"
#include "stream_pub.h"
#include "transport_pub.h"
#include "adapter_pub.h"
#include "dispatcher.h"
#include "local_notify.h"
#include "ffts_common_pub.h"

namespace hccl {
constexpr s32 HCCL_EXEC_STAGE_NOT_SET = -1;
constexpr s32 HCCL_EXEC_STEP_NOT_SET = -1;
constexpr s32 HCCL_EXEC_PLANE_NOT_SET = -1;
constexpr u64 ZERO_SLICE = 0;
constexpr u32 TWO_RANK_SIZE = 2;
constexpr u32 HCCL_SPLIT_FLAG = 2;
constexpr u32 DMA_REDUCE_TWO_OFFSET = 2;
constexpr u32 DMA_REDUCE_THREE_OFFSET = 3;
constexpr u64 HCCL_CHUNK_SIZE = 1024 * 1024 * 1024; // 1024*1024*1024的size
constexpr u64 HCCL_MIN_PIPLINE_SLICE_ALIGN = 512;
constexpr u64 HCCL_MIN_SLICE_ALIGN_910B = 16384;
constexpr u64 HCCL_MIN_SLICE_ALIGN_910_73 = 16384;
constexpr u64 HCCL_SDMA_RDMA_SPLIT_SIZE = 67108864;
constexpr u64 HCCL_MIN_SLICE_ALIGN_ONCHIP = 512;
constexpr u64 HCCL_MIN_SLICE_ALIGN = 128;
constexpr u64 HCCL_NIC_MAX_NUM = 8;

enum class SliceType {
    SLICE_TYPE_TX,
    SLICE_TYPE_RX
};

enum class HalvingDoublingType {
    BINARY_BLOCK_HALVING_DOUBLING,
    RECURSIVE_HALVING_DOUBLING,
    RESERVED_ALGORITHM_TYPE
};

using SliceType = enum SliceType;
struct Slice {
    u64 offset{0}; // Slice相对于input/output的偏移字节数，gather类操作取output，scatter类操作取input
    u64 size{0};    // Slice的数据大小，单位：字节
};

enum class RunStage {
    RUN_PREPARE,
    RUN_REDUCE_SCATTER,
    RUN_ALLGATHER,
    RUN_ALLREDUCE,
    RUN_DEFAULT
};

class ExecutorBase {
public:
    explicit ExecutorBase(const HcclDispatcher dispatcher);
    virtual ~ExecutorBase();

    virtual HcclResult RunAsync(const u32 rank, const u32 rankSize,
        const std::vector<std::shared_ptr<Transport> > &links);
    virtual HcclResult RunAsyncStaged(const u32 rank, const u32 rankSize,
        const std::vector<std::shared_ptr<Transport> > &links, RunStage stage);
    HcclResult Prepare(DeviceMem &inputMem, DeviceMem &outputMem, DeviceMem &scratchMem, const u64 count,
                         const HcclDataType dataType, const Stream &stream,
                         const HcclReduceOp reductionOp = HCCL_REDUCE_RESERVED,
                         const u32 root = INVALID_VALUE_RANKID,
                         const std::vector<Slice> &slices = std::vector<Slice>(ZERO_SLICE),
                         const u64 baseOffset = 0, std::vector<u32> nicRankList = {0, 1, 2, 3, 4, 5, 6, 7});

    HcclResult Prepare(DeviceMem &inputMem, DeviceMem &scratchMem, const u64 count,
                         const HcclDataType dataType,
                         const Stream &stream, const HcclReduceOp reductionOp = HCCL_REDUCE_RESERVED,
                         const u32 root = INVALID_VALUE_RANKID,
                         const std::vector<Slice> &slices = std::vector<Slice>(ZERO_SLICE),
                         const u64 baseOffset = 0, std::vector<u32> nicRankList = {0, 1, 2, 3, 4, 5, 6, 7});
    HcclResult Sum(const std::vector<Slice> &inputSlices, u32 start, u32 num, u64 &sizeOut);
    HcclResult RegisterProfiler(s32 planeId, s32 stage, s32 step, const Stream &stream);
    static HcclResult ExecEmptyTask(DeviceMem &inputMem, DeviceMem &outputMem, Stream &stream,
        const HcclDispatcher dispatcher);
    HcclResult CheckConcurrentDirectParameters(const u32 rank, const u32 rankSize, const std::vector<LINK> &links);
    u32 DataUnitSize(HcclDataType dataType) const
    {
        if (dataType >= HCCL_DATA_TYPE_RESERVED) {
            HCCL_ERROR("[ExecutorBase][DataUnitSize]data type[%s] out of range[%d, %d]",
                GetDataTypeEnumStr(dataType).c_str(), HCCL_DATA_TYPE_INT8, HCCL_DATA_TYPE_RESERVED - 1);
            return 0;
        }

        return SIZE_TABLE[dataType];
    }

    static std::vector<bool> CalcLinksRelation(const u32 rank, const u32 rankSize, const u32 rootRank = 0,
        HalvingDoublingType algorithmType = HalvingDoublingType::RECURSIVE_HALVING_DOUBLING);

    static HcclResult PrepareSliceData(u64 dataCount, u32 unitSize, u32 sliceNum, u64 piplineOffset,
        std::vector<Slice> &dataSlice);
    static HcclResult PrepareSliceMeshStreams(const std::vector<Slice> &rankSegsSlice, u32 streamCount,
        std::vector<std::vector<Slice>> &mutliStreamsSlices);

    static inline u64 RoundUpWithDivisor(u64 value, u64 divisor)
    {
        if (value == 0) {
            return divisor;
        }
        // divisor必须大于等于1, 返回value向上取divisor的整数倍的值
        return ((value + (divisor - 1)) / divisor) * divisor;
    }
    inline u64 ByteOffset(u64 countOffset) const
    {
        return countOffset * DataUnitSize(dataType_);
    }
    inline u64 SliceOffset(u32 sliceIndex, u64 countPerSlice) const
    {
        return sliceIndex * countPerSlice * DataUnitSize(dataType_);
    }
    inline void CloseBarrier()
    {
        barrierSwitchOn_ = false;
    }

protected:
    HcclResult ExecuteBarrier(const std::shared_ptr<Transport> &preLink, const std::shared_ptr<Transport> &aftLink);
    HcclResult ExecuteBarrier(const std::shared_ptr<Transport> &preLink,
        const std::shared_ptr<Transport> &aftLink, Stream &stream);
    HcclResult ExecuteBarrier(std::shared_ptr<Transport> link, Stream &stream);
    HcclResult ExecuteRxSync(std::shared_ptr<Transport> link, UserMemType srcMemType, u64 srcOffset,
        void *dst, u64 len, Stream &stream) const;
    HcclResult ExecuteTxSync(std::shared_ptr<Transport> link, UserMemType dstMemType, u64 dstOffset,
        void *src, u64 len, Stream &stream) const;
    virtual HcclResult PrepareRunAsync(const u32 rank, const u32 rankSize,
        const std::vector<std::shared_ptr<Transport> > &links);
    const HcclDispatcher dispatcher_;
    std::vector<Slice> slicesDummy_;
    std::vector<Slice> &slices_;
    DeviceMem inputMem_;   /* * 输入memory */
    DeviceMem outputMem_;  /* * 输出memory */
    DeviceMem scratchMem_; /* * 草稿memory */

    u64 count_; //  需处理的每块memory数据总个数
    u64 dataBytes_; //  数据所占的字节数
    HcclDataType dataType_;
    HcclReduceOp reductionOp_;
    u32 root_;

    // Added on Mar.24th, for profiling template
    StepData profilerInput_;
    u64 baseOffset_;

    Stream stream_;

    // 用于chunk算法
    std::vector<u32> nicRankList_;
    std::vector<std::vector<u32>> rankSliceLists_;
    bool barrierSwitchOn_;
private:
    static void CalcBinaryBlockParams(u32 rank, u32 rankSize, u32 &stepsInBlock, u32 &lowerBlockSize,
        u32 &myBlockSize, u32 &rankInMyBlock, u32 &myBlockOffset, u32 &higherBlockSize);
    static HcclResult CalcBinaryBlockHalvingDoubleLinkReleation(u32 rank,  u32 rankSize,
                                                                      std::vector<bool> &linkRelation);

    static void CalcLinkInBlock(u32 blockSize, u32 rankInBlock, std::list<u32> &linkRankIndexInBlock);
    static void CalcLinkBetweenParts(u32 part1Size, std::list<u32> &linkRankIndexInBlock,
                                             std::list<u32> &linkRankIndex, bool oddRank);
    static void CalcRecursiveHalvingDobuleLinkReleation(u32 rank, u32 rankSize, u32 rootRank,
                                                                   std::vector<bool> &linkRelation);
    static void CalcRecursiveHdLinkRelationForFirstScene(u32 rank,
        u32 part1Size, u32 blockSize, std::vector<bool> &linkRelation);
    static void CalcRecursiveHdLinkRelationForSecondScene(u32 rank,
        u32 part1Size, u32 blockSize, std::vector<bool> &linkRelation);
};
}  // namespace hccl

#endif /* EXECUTOR_BASE_PUB_H */
