/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_COMMON_EXECUTOR_H
#define COLL_COMMON_EXECUTOR_H

#include "coll_native_executor_base.h"
#include "coll_alg_exec_registry.h"

namespace hccl {
class CollCommExecutor : public CollNativeExecutorBase {
public:
    CollCommExecutor(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher);
    ~CollCommExecutor() = default;

    // CCL Op Share
    HcclResult MultiRingAllReduce(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
                                    const u64 count, const HcclDataType dataType,
                                    const HcclReduceOp reductionOp,
                                    const std::vector<std::vector<Slice>> &multRingsSliceZero, Stream stream,
                                    s32 profStage, const u64 baseOffset = 0);
    HcclResult MultiRingReduceScatter(const std::string &tag, DeviceMem inputMem, DeviceMem outputMem, const u64 count,
        const HcclDataType dataType, const HcclReduceOp reductionOp,
        const std::vector<std::vector<Slice>> multRingsSliceZero, Stream stream,
        s32 profStage, const u64 baseOffset = 0, const HcomCollOpInfo *opInfo = nullptr,
        const std::vector<std::vector<Slice>> multRingsUserMemSlice = std::vector<std::vector<Slice>> (0));

    HcclResult MultiRingReduceScatterConcurrent(const std::string &tag, DeviceMem inputMem,
        DeviceMem outputMem, const u64 count, const HcclDataType dataType,
        const HcclReduceOp reductionOp,
        const std::vector<std::pair<bool, std::vector<Slice>>> multRingsSliceZero, Stream stream,
        s32 profStage, const u64 baseOffset, const HcomCollOpInfo *opInfo = nullptr);

    HcclResult MultiRingAllGather(const std::string &tag, DeviceMem inputMem, DeviceMem outputMem, const u64 count,
        const HcclDataType dataType,
        const std::vector<std::vector<Slice> > multRingsSliceZero, Stream stream,
        s32 profStage, const u64 baseOffset = 0, const HcomCollOpInfo *opInfo = nullptr,
        const std::vector<std::vector<Slice>> multRingsUserMemSlice = std::vector<std::vector<Slice>> (0));

    HcclResult MultiRingAllGatherConcurrent(const std::string &tag, DeviceMem inputMem, DeviceMem outputMem,
        const u64 count, const HcclDataType dataType,
        const std::vector<std::pair<bool, std::vector<Slice>>> multRingsSliceZero, Stream stream,
        s32 profStage, const u64 baseOffset = 0, const HcomCollOpInfo *opInfo = nullptr);

    HcclResult MultiRingMultiRootScatter(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
        const u64 count, const HcclDataType dataType, const std::vector<std::vector<Slice>> &multRingsSliceZero,
        u32 root, Stream stream, const u64 baseOffset);

    HcclResult MultiStreamReduceScatterMesh(const std::string &tag, DeviceMem inputMem, DeviceMem outputMem,
                                                  const u64 count, const HcclDataType dataType,
                                                  const HcclReduceOp reductionOp,
                                                  const std::vector<std::vector<Slice>>& multStreamsSlice,
                                                  Stream stream,
                                                  const CommPlane commLevelIndex,
                                                  const u64 baseOffset = 0);

    HcclResult MultiRingGather(const std::string &tag, DeviceMem inputMem, DeviceMem outputMem, const u64 count,
                                const HcclDataType dataType, const std::vector<std::vector<Slice>> multRingsSliceZero, 
                                HcclReduceOp op, u32 root, Stream stream, s32 profStage);

    HcclResult MultiStreamReduceScatterMeshAtomic(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
                                                  const u64 count, const HcclDataType dataType,
                                                  const HcclReduceOp reductionOp,
                                                  const std::vector<Slice> &dataSlice,
                                                  Stream &stream,
                                                  const CommPlane commLevelIndex,
                                                  const u64 baseOffset = 0, HcomCollOpInfo *opInfo = nullptr);
    HcclResult PrepareReduceScatterSliceData(u64 dataCount, u32 unitSize, u32 sliceNum, std::vector<Slice> &dataSlice);

    HcclResult MultiRingScatter(const std::string &tag, DeviceMem inputMem, DeviceMem outputMem, const u64 count,
                                const HcclDataType dataType, const std::vector<std::vector<Slice> > multRingsSliceZero,
                                u32 root, Stream stream, const HcomCollOpInfo *opInfo);
    std::vector<std::vector<u32>> GetRingsOrderByTopoType(u32 ranksSize, TopoType topoType, std::vector<u32> &nicList);
    HcclResult MutliSegSlicePrepare(const std::vector<Slice> &dataSegsSlice,
        std::vector<std::vector<Slice> >& mutliSegsSlices, u32 ringCount);
    HcclResult MutliSegSlicePrepareAvoidCceRewrite(const std::vector<Slice> &dataSegsSlice,
        std::vector<std::vector<Slice> >& mutliSegsSlices, u32 ringCount) const;
    void NicSendSizeCal(const std::vector<std::vector<Slice>> &mutliSegsSlices, u32 ringCount, u32 chunkSize,
        const std::vector<u32> &nicList, const std::string &tag);
    std::vector<std::vector<Slice> > PrepareMultiRingSlice(const std::vector<Slice> &dataSegsSlice,
        const std::string &tag, bool avoidCceRewrite = false, std::vector<u32> nicList = {0, 1, 2, 3, 4, 5, 6, 7});
    u32 RefreshCommIdx(u32 commIndex, std::vector<u32> nicList, u32 devicePhyId);

    bool Is2U2PInfer();
    bool Is910BSingleMesh();
    bool NeedCreateSingleMeshPlane(const bool isInlineReduce);
    bool SingleMeshInlineReduce(void *inputPtr, void *outputPtr, HcclDataType dataType, HcclReduceOp op);
    bool IsMultiMeshInlineReduce(void *inputPtr, void *outputPtr, HcclDataType dataType, HcclReduceOp op);

    u64 GetReduceAttr(DeviceMem &inputMem, DeviceMem &outputMem, HcclDataType dataType, HcclReduceOp op);
    HcclResult PrepareInnerCommInfo(u32 &segmentIdx, u32 &commIndex, u64 &hdSize,
                                          const SubCommInfo &commInfo,
                                          const std::vector<std::vector<Slice> > &multRingsSliceZero,
                                          const std::string &tag);
protected:
    HcclResult GetSubStreamInfoOnOneRing(const innerStreamInfo_t &streamInfo, const u32 ringIndex,
                                         std::vector<Stream>                       &subStreamsInOneRing,
                                         std::vector<std::shared_ptr<LocalNotify>> &mainSignalsInOneRing,
                                         std::vector<std::shared_ptr<LocalNotify>> &subSignalsInOneRing);
    HcclResult CalUserMemSlices(const HcclDataType dataType, const HcomCollOpInfo *opInfo,
                                const std::vector<Slice> &singleRingSliceZero, u32 ringIndex,
                                const std::vector<std::vector<u32>> &multiRingsOrder,
                                std::vector<Slice>                  &userMemSlices);
    HcclResult GetRankOrder(const std::vector<std::vector<u32>> &multiRingsOrder, u32 ringIndex,
                            std::vector<u32> &rankOrder);
    HcclResult SetRingNics(const std::string &tag, const std::vector<std::vector<u32>> &ringNics);
    HcclResult GetRingNics(const std::string &tag, std::vector<std::vector<u32>> &ringNics);
    HcclResult SetNicSendSize(const std::string &tag, std::vector<u64> &sizeList);
    HcclResult GetStreamThreadManage(const std::string &tag, u32 streamNum,
                                     std::vector<std::shared_ptr<ThreadManage>> &threadManager);
    std::mutex ringNicListLock_;
    std::map<std::string, std::vector<std::vector<u32>>> ringNicList_;
    std::mutex nicSendSizeListLock_;
    std::map<std::string, std::vector<u64>> nicSendSizeList_;
    std::mutex threadManageMapLock_;
    std::map<std::string, std::vector<std::shared_ptr<ThreadManage>>> threadManageMap_;
};
} // namespace hccl

#endif /** __COLL_COMMON_EXECUTOR_H__ */