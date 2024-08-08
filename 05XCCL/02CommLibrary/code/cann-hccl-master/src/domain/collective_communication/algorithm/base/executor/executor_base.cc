/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sal_pub.h"
#include "executor_base.h"

namespace hccl {
ExecutorBase::ExecutorBase(const HcclDispatcher dispatcher)
    : dispatcher_(dispatcher),
      slices_(slicesDummy_), count_(0), dataBytes_(0), dataType_(HCCL_DATA_TYPE_RESERVED),
      reductionOp_(HCCL_REDUCE_RESERVED), root_(INVALID_VALUE_RANKID),
      baseOffset_(0), barrierSwitchOn_(true)
{
}

ExecutorBase::~ExecutorBase()
{
    slices_.clear();
}

// prepare函数给需要进行集合通信操作进行参数赋值
HcclResult ExecutorBase::Prepare(DeviceMem &inputMem, DeviceMem &outputMem, DeviceMem &scratchMem,
                                 const u64 count,
                                 const HcclDataType dataType, const Stream &stream,
                                 const HcclReduceOp reductionOp,
                                 const u32 root, const std::vector<Slice> &slices, const u64 baseOffset,
                                 std::vector<u32> nicRankList)
{
    // 部分集合通信操作允许input_mem/output_mem为空

    HCCL_DEBUG("ExecutorBase prepare start");

    /* * 参数保存 */
    inputMem_ = inputMem;
    outputMem_ = outputMem;
    scratchMem_ = scratchMem;
    stream_ = stream;
    count_ = count;
    dataType_ = dataType;
    dataBytes_ = count * DataUnitSize(dataType);
    reductionOp_ = reductionOp;
    root_ = root;

    /* 相对用户基地址偏移 */
    baseOffset_ = baseOffset;

    if (slices.size() > 0) {
        slices_.resize(slices.size());
        slices_ = slices;
    }

    nicRankList_.assign(nicRankList.begin(), nicRankList.end());
    // 不带入该参数，代表数据均分，直接用count赋值
    HCCL_DEBUG("ExecutorBase prepare end");
    return HCCL_SUCCESS;
}

HcclResult ExecutorBase::Prepare(DeviceMem &inputMem, DeviceMem &scratchMem, const u64 count,
                                 const HcclDataType dataType, const Stream &stream,
                                 const HcclReduceOp reductionOp,
                                 const u32 root, const std::vector<Slice> &slices, const u64 baseOffset,
                                 std::vector<u32> nicRankList)
{
    // 部分集合通信操作允许input_mem/output_mem为空
    CHK_PTR_NULL(stream.ptr());

    HCCL_DEBUG("ExecutorBase prepare start");

    /* * 参数保存 */
    inputMem_ = inputMem;
    outputMem_ = inputMem;
    scratchMem_ = scratchMem;
    stream_ = stream;
    count_ = count;
    dataType_ = dataType;
    dataBytes_ = count * DataUnitSize(dataType);
    reductionOp_ = reductionOp;
    root_ = root;

    /* 相对用户基地址偏移 */
    baseOffset_ = baseOffset;

    if (slices.size() > 0) {
        slices_.resize(slices.size());
        slices_ = slices;
    }

    nicRankList_.assign(nicRankList.begin(), nicRankList.end());
    // 不带入该参数，代表数据均分，直接用count赋值
    HCCL_DEBUG("ExecutorBase prepare end");
    return HCCL_SUCCESS;
}

HcclResult ExecutorBase::RegisterProfiler(s32 planeId, s32 stage, s32 step, const Stream &stream)
{
    profilerInput_.streamID = stream.id();
    profilerInput_.planeID = planeId;
    profilerInput_.stage = stage;
    profilerInput_.step = step;
    return HCCL_SUCCESS;
}

HcclResult ExecutorBase::RunAsync(const u32 rank, const u32 rankSize,
    const std::vector<std::shared_ptr<Transport> > &links)
{
    (void)rank;
    (void)rankSize;
    (void)links;
    return HCCL_SUCCESS;
}

HcclResult ExecutorBase::RunAsyncStaged(const u32 rank, const u32 rankSize,
    const std::vector<std::shared_ptr<Transport> > &links, RunStage stage)
{
    (void)rank;
    (void)rankSize;
    (void)links;
    (void)stage;
    return HCCL_SUCCESS;
}

void ExecutorBase::CalcBinaryBlockParams(u32 rank, u32 rankSize, u32 &stepsInBlock, u32 &lowerBlockSize,
    u32 &myBlockSize, u32 &rankInMyBlock, u32 &myBlockOffset, u32 &higherBlockSize)
{
    u32 offset = rankSize;
    u32 blockSize = 1;

    u32 currentBlockSize = 0;
    u32 preBlockSize = 0;
    while (offset != 0) {
        if ((rankSize & blockSize) != 0) {
            preBlockSize = currentBlockSize;
            currentBlockSize = blockSize;
            if (blockSize == 0) {
                HCCL_ERROR("[Calc][BinaryBlockParams]calculate_binary_block_paras: blockSize[%u] is zero",
                    blockSize);
                break;
            }
            offset -= blockSize;
            if (myBlockSize != 0) {
                higherBlockSize = currentBlockSize;
                break;
            }
            if (offset <= rank) {
                myBlockOffset = offset;
                myBlockSize = currentBlockSize;
                lowerBlockSize = preBlockSize;
            }
        }
        blockSize <<= 1;
    }
    stepsInBlock = SalLog2(myBlockSize);
    if (myBlockSize != 0) {
        rankInMyBlock = rank % myBlockSize;
    }
}
std::vector<bool> ExecutorBase::CalcLinksRelation(const u32 rank, const u32 rankSize, const u32 rootRank,
                                                  HalvingDoublingType algorithmType)
{
    HCCL_INFO("Calculate links relation: Rank[%u], RankSize[%u], RootRank[%u], HDType[%d]",
        rank, rankSize, rootRank, algorithmType);
    std::vector<bool> linkRelation(rankSize, false);

    HcclResult ret;
    switch (algorithmType) {
        case HalvingDoublingType::RECURSIVE_HALVING_DOUBLING:
            CalcRecursiveHalvingDobuleLinkReleation(rank, rankSize, rootRank, linkRelation);
            break;
        case HalvingDoublingType::BINARY_BLOCK_HALVING_DOUBLING:
        default:
            ret = CalcBinaryBlockHalvingDoubleLinkReleation(rank, rankSize, linkRelation);
            if (ret == HCCL_E_PARA) {
                HCCL_ERROR("[Calc][LinksRelation]errNo[0x%016llx] Calculation binary block parameter error",
                    HCCL_ERROR_CODE(HCCL_E_PARA));
                for (u32 i = 0; i < rankSize; i++) {
                    linkRelation[i] = false;
                }
                return linkRelation;
            }
            break;
    }

    // 打印建链关系
    std::string strLinkRelation;
    u32 index = 0;
    for (auto link : linkRelation) {
        if (link) {
            strLinkRelation.append(std::to_string(index));
        }
        if (index < linkRelation.size() - 1) {
            strLinkRelation.append(", ");
        }
        index++;
    }
    HCCL_DEBUG("Rank[%u] has link to these Ranks: %s", rank, strLinkRelation.c_str());

    return linkRelation;
}

// 将数据均分，最小单位是128
HcclResult ExecutorBase::PrepareSliceData(u64 dataCount, u32 unitSize, u32 sliceNum, u64 piplineOffset,
    std::vector<Slice>& dataSlice)
{
    Slice temp;
    u64 totalSize = dataCount * unitSize;
    dataSlice.clear();
    dataSlice.reserve(sliceNum);
    CHK_PRT_RET((sliceNum == 0), HCCL_ERROR("[Prepare][SliceData]data slice prepare, sliceNum is 0"), HCCL_E_PARA);
    u64 tempPerSlice = (totalSize + sliceNum - 1) / sliceNum; /* 1是为了向上取整 */
    u64 sizePerSlice = RoundUpWithDivisor(tempPerSlice, HCCL_MIN_SLICE_ALIGN);
    HCCL_DEBUG("total_size:%llu sliceNum:%u temp_per_ring:%llu size_per_ring:%llu", totalSize, sliceNum, tempPerSlice,
        sizePerSlice);
    u64 residueSize = totalSize;
    u32 i = 0;
    while (residueSize > 0) {
        u64 sliceSize = sizePerSlice < residueSize ? sizePerSlice : residueSize;
        temp.size = sliceSize;
        temp.offset = totalSize - residueSize + piplineOffset;
        i++;
        CHK_PRT_RET((sliceSize <= 0), HCCL_ERROR("[Prepare][SliceData]data_slice_prepare sliceSize[%llu]", sliceSize),
            HCCL_E_PARA);
        residueSize -= sliceSize;
        dataSlice.push_back(temp);
    }
    while (i < sliceNum) {
        temp.size = 0;
        temp.offset = totalSize + piplineOffset;
        i++;
        dataSlice.push_back(temp);
    }
    return HCCL_SUCCESS;
}

// 数据切分到每个stream上，最小单位是128
HcclResult ExecutorBase::PrepareSliceMeshStreams(const std::vector<Slice> &rankSegsSlice, u32 streamCount,
    std::vector<std::vector<Slice>>& mutliStreamsSlices)
{
    std::vector<u64> rankStreamSize;
    std::vector<u64> rankResidueSize;
    rankStreamSize.reserve(rankSegsSlice.size());
    rankResidueSize.reserve(rankSegsSlice.size());
    mutliStreamsSlices.clear();
    mutliStreamsSlices.reserve(streamCount);
    if (streamCount == 0) {
        HCCL_ERROR("[Prepare][SliceMeshStreams]data slice mesh prepare, streamCount is 0");
        return HCCL_E_PARA;
    }
    for (u32 rankId = 0; rankId < rankSegsSlice.size(); rankId++) {
        u64 rankDataSize = rankSegsSlice[rankId].size;
        u64 sizePerStream = (rankDataSize + streamCount - 1) / streamCount;
        u64 sizeAlgin = ExecutorBase::RoundUpWithDivisor(sizePerStream, HCCL_MIN_SLICE_ALIGN);
        rankStreamSize.push_back(sizeAlgin);
        rankResidueSize.push_back(rankDataSize);
    }

    for (u32 streamIndex = 0; streamIndex < streamCount; streamIndex++) {
        std::vector<Slice> singleStreamSlices;
        singleStreamSlices.reserve(rankSegsSlice.size());
        for (u32 rankId = 0; rankId < rankSegsSlice.size(); rankId++) {
            Slice rankSliceTemp;
            u64 rankDataResidue = rankResidueSize[rankId];
            u64 singleStreamSize = 0;
            if (rankDataResidue > 0) {
                singleStreamSize = rankStreamSize[rankId] < rankDataResidue ? rankStreamSize[rankId] : rankDataResidue;
                rankSliceTemp.offset = rankSegsSlice[rankId].offset + rankSegsSlice[rankId].size - rankDataResidue;
                rankSliceTemp.size = singleStreamSize;
                rankResidueSize[rankId] -= singleStreamSize;
            } else {
                rankSliceTemp.offset = rankSegsSlice[rankId].offset;
                rankSliceTemp.size = 0;
            }
            singleStreamSlices.push_back(rankSliceTemp);
        }
        mutliStreamsSlices.push_back(singleStreamSlices);
    }
    return HCCL_SUCCESS;
}

HcclResult ExecutorBase::CalcBinaryBlockHalvingDoubleLinkReleation(u32 rank, u32 rankSize,
    std::vector<bool> &linkRelation)
{
    u32 stepsInBlock = 0;
    u32 myBlockSize = 0;
    u32 rankInMyBlock = INVALID_VALUE_RANKID;
    u32 myBlockOffset = 0;
    u32 higherBlockSize = 0;
    u32 lowerBlockSize = 0;
    CalcBinaryBlockParams(rank, rankSize, stepsInBlock, lowerBlockSize, myBlockSize, rankInMyBlock,
        myBlockOffset, higherBlockSize);
    if (lowerBlockSize == 0) {
        HCCL_ERROR("[Calc][BinaryBlockHalvingDoubleLinkReleation]lowerBlockSize size is zero.");
        return HCCL_E_PARA;
    }
    for (u32 i = 0; i < rankSize; i++) {
        linkRelation[i] = false;
    }
    u32 dstRankBitmask = 1;
    for (u32 i = 0; i < stepsInBlock; i++) {
        u32 dstRankBitmaskTemp = dstRankBitmask;
        u32 dstRank = rank ^ dstRankBitmaskTemp;
        linkRelation[dstRank] = true;
        dstRankBitmask <<= 1;
    }
    if (lowerBlockSize != 0) {
        u32 divBlockSize = myBlockSize / lowerBlockSize;
        u32 dstRank = myBlockOffset + myBlockSize + rankInMyBlock / divBlockSize;
        linkRelation[dstRank] = true;
    }
    if (higherBlockSize != 0) {
        u32 segments = higherBlockSize / myBlockSize;  // 和高阶block的rank数差n倍，那么本rank就要向高阶的n个rank发送
        u32 dstRank = (myBlockOffset - higherBlockSize) + rankInMyBlock * segments;
        for (u32 i = 0; i < segments; i++) {
            linkRelation[dstRank] = true;
            dstRank++;
        }
    }
    return HCCL_SUCCESS;
}

//  用于recursive halving doubling
void ExecutorBase::CalcLinkInBlock(u32 blockSize, u32 rankInBlock, std::list<u32> &linkRankIndexInBlock)
{
    u32 blockSizeHalving = blockSize / 2;       //  每个循环除2计算当前block的折半rank数
    u32 rankInTempBlock = rankInBlock;
    while (blockSizeHalving >= 1) {
        if (rankInTempBlock < blockSizeHalving) {
            linkRankIndexInBlock.push_back(rankInBlock + blockSizeHalving);
        } else {
            linkRankIndexInBlock.push_back(rankInBlock - blockSizeHalving);
            rankInTempBlock -= blockSizeHalving;
        }
        blockSizeHalving = blockSizeHalving / 2; //  每个循环除2计算当前block的折半rank数
    }
}

//  用于recursive halving doubling
void ExecutorBase::CalcLinkBetweenParts(u32 part1Size, std::list<u32> &linkRankIndexInBlock,
                                        std::list<u32> &linkRankIndex, bool oddRank)
{
    for (auto it : linkRankIndexInBlock) {
        if (it < (part1Size / 2)) {                   //  属于part1,除2计算part1中的rank范围
            if (oddRank) {
                linkRankIndex.push_back(it * 2 + 1); //  乘2加1得到part1中奇数rank
            } else {
                linkRankIndex.push_back(it * 2);      //  乘2得到part1中偶数rank
            }
        } else {
            linkRankIndex.push_back(part1Size / 2 + it); //  不属于part1,除2得到part1中rank范围
        }
    }
}

//  用于recursive halving doubling
void ExecutorBase::CalcRecursiveHalvingDobuleLinkReleation(u32 rank, u32 rankSize, u32 rootRank,
                                                           std::vector<bool> &linkRelation)
{
    u32 exponent = 0;

    if (rootRank == INVALID_VALUE_RANKID) { // all reduce 走这个分支
        rootRank = 0;
    }

    u32 base = 1;
    while ((base << exponent) <= rankSize) {
        exponent++;
    }
    if (exponent > 0) {
        exponent--;
    }
    u32 blockSize = base << exponent;
    u32 part1Size = (rankSize - blockSize) * 2; // part1的大小为总rankSize减去blockSize再乘2

    //  情况1、情况3的建链方式
    if (rootRank >= part1Size || rootRank % 2 == 0) { // 除2判断是否为偶数
        CalcRecursiveHdLinkRelationForFirstScene(rank, part1Size, blockSize, linkRelation);
    } else { //  情况2的建链方式，rootRank<part1Size && 1==rootRank%2
        CalcRecursiveHdLinkRelationForSecondScene(rank, part1Size, blockSize, linkRelation);
    }
}

void ExecutorBase::CalcRecursiveHdLinkRelationForFirstScene(u32 rank,
    u32 part1Size, u32 blockSize, std::vector<bool> &linkRelation)
{
    if (rank < part1Size && rank % 2 == 0) { // 除2判断是否为偶数
        std::list<u32> linkRankIndex;
        std::list<u32> linkRankIndexInBlock;
        u32 rankInBlock = rank / 2;          // 除2计算block内的rank号
        CalcLinkInBlock(blockSize, rankInBlock, linkRankIndexInBlock);
        CalcLinkBetweenParts(part1Size, linkRankIndexInBlock, linkRankIndex, false);
        linkRankIndex.push_back(rank + 1);   // 加1得到旁边的那个rank
        for (auto it : linkRankIndex) {
            linkRelation[it] = true;
        }
    } else if (rank < part1Size && rank % 2 == 1) { // 除2判断是否为奇数
        if (rank - 1 < linkRelation.size() && rank - 1 >= 0) {
            linkRelation[rank - 1] = true;    //  只有旁边的那个rank
        }
    } else {                                //  rank大于等于part1Size
        std::list<u32> linkRankIndexInBlock;
        u32 rankInBlock = rank - part1Size / 2; // 除2计算part1在block内的rank范围
        std::list<u32> linkRankIndex;
        CalcLinkInBlock(blockSize, rankInBlock, linkRankIndexInBlock);
        CalcLinkBetweenParts(part1Size, linkRankIndexInBlock, linkRankIndex, false);
        for (auto it : linkRankIndex) {
            linkRelation[it] = true;
        }
    }
}

void ExecutorBase::CalcRecursiveHdLinkRelationForSecondScene(u32 rank,
    u32 part1Size, u32 blockSize, std::vector<bool> &linkRelation)
{
    if (rank < part1Size && rank % 2 == 1) {  // 除2判断是否为奇数
        std::list<u32> linkRankIndex;
        std::list<u32> linkRankIndexInBlock;
        u32 rankInBlock = (rank - 1) / 2;     // 减1再除2计算在block内的rank
        CalcLinkInBlock(blockSize, rankInBlock, linkRankIndexInBlock);
        CalcLinkBetweenParts(part1Size, linkRankIndexInBlock, linkRankIndex, true);
        linkRankIndex.push_back(rank - 1);   // 减1得到旁边的那个rank
        for (auto it : linkRankIndex) {
            linkRelation[it] = true;
        }
    } else if (rank < part1Size && rank % 2 == 0) { // 除2判断是否为偶数
        if (rank + 1 < linkRelation.size()) {
            linkRelation[rank + 1] = true;    //  只有旁边的那个rank
        }
    } else {                                //  rank大于等于part1Size
        std::list<u32> linkRankIndexInBlock;
        u32 rankInBlock = rank - part1Size / 2; // 除2计算part1在block内的rank范围
        std::list<u32> linkRankIndex;
        CalcLinkInBlock(blockSize, rankInBlock, linkRankIndexInBlock);
        CalcLinkBetweenParts(part1Size, linkRankIndexInBlock, linkRankIndex, true);
        for (auto it : linkRankIndex) {
            linkRelation[it] = true;
        }
    }
}

HcclResult ExecutorBase::ExecuteBarrier(const std::shared_ptr<Transport> &preLink,
                                        const std::shared_ptr<Transport> &aftLink)
{
    return ExecuteBarrier(preLink, aftLink, stream_);
}

HcclResult ExecutorBase::ExecuteBarrier(const std::shared_ptr<Transport> &preLink,
    const std::shared_ptr<Transport> &aftLink, Stream &stream)
{
    // 同步与preLink保证数据收发已结束
    CHK_RET(preLink->TxAck(stream));

    CHK_RET(aftLink->RxAck(stream));

    // 同步与aftLink保证数据收发已结束
    CHK_RET(aftLink->TxDataSignal(stream));

    CHK_RET(preLink->RxDataSignal(stream));

    return HCCL_SUCCESS;
}

HcclResult ExecutorBase::ExecuteBarrier(std::shared_ptr<Transport> link, Stream &stream)
{
    CHK_RET(link->TxAck(stream));

    CHK_RET(link->RxAck(stream));

    CHK_RET(link->TxDataSignal(stream));

    CHK_RET(link->RxDataSignal(stream));

    return HCCL_SUCCESS;
}
HcclResult ExecutorBase::Sum(const std::vector<Slice> &inputSlices, u32 start, u32 num, u64 &sizeOut)
{
    u64 totalSize = 0;
    // 判断不是<=因为访问vector前会先进行num--
    CHK_PRT_RET(inputSlices.size() < start + num, HCCL_ERROR("[ExecutorBase][Sum]recursive Halving Doubling sum "\
        "error.para: size[%llu], start[%u], num[%u]", inputSlices.size(), start, num), HCCL_E_PARA);
    while (num > 0) {
        num--;
        totalSize += inputSlices[start + num].size;
    }
    sizeOut = totalSize;
    return HCCL_SUCCESS;
}
HcclResult ExecutorBase::ExecuteRxSync(std::shared_ptr<Transport> link, UserMemType srcMemType, u64 srcOffset,
    void *dst, u64 len, Stream &stream) const
{
    HcclResult ret = link->TxAsync(srcMemType, srcOffset, dst, 0, stream);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[ExecutorBase][ExecuteRxSync]ExecuteRxSync: tx async size[%llu] "\
        "failed", len), ret);
    ret = link->RxAsync(srcMemType, srcOffset, dst, len, stream);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[ExecutorBase][ExecuteRxSync]ExecuteRxSync: rx async with rcvMem[%p] "\
        "offset[%llu] size[%llu] failed", dst, srcOffset, len), ret);
    ret = link->DataReceivedAck(stream);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[ExecutorBase][ExecuteRxSync]ExecuteRxSync: data received ack failed"), ret);
    return HCCL_SUCCESS;
}
HcclResult ExecutorBase::ExecuteTxSync(std::shared_ptr<Transport> link, UserMemType dstMemType, u64 dstOffset,
    void *src, u64 len, Stream &stream) const
{
    HcclResult ret = link->TxAsync(dstMemType, dstOffset, src, len, stream);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[ExecutorBase][ExecuteTxSync]ExecuteTxSync: tx async sendMem[%p] "\
        "offset[%llu] size[%llu] failed", src, dstOffset, len), ret);
    // 接收应答
    ret = link->RxAsync(dstMemType, dstOffset, src, 0, stream);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[ExecutorBase][ExecuteTxSync]ExecuteTxSync: rx async size[%llu] "\
        "failed", len), ret);
    ret = link->DataReceivedAck(stream);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[ExecutorBase][ExecuteTxSync]ExecuteTxSync: data received ack failed"), ret);
    return HCCL_SUCCESS;
}

HcclResult ExecutorBase::PrepareRunAsync(const u32 rank, const u32 rankSize,
    const std::vector<std::shared_ptr<Transport> > &links)
{
    (void)rank;
    (void)rankSize;
    (void)links;
    return HCCL_SUCCESS;
}

HcclResult ExecutorBase::ExecEmptyTask(DeviceMem &inputMem, DeviceMem &outputMem, Stream &stream,
    const HcclDispatcher dispatcher)
{
    DeviceMem emptySrcMem = DeviceMem::create(inputMem.ptr(), 0);
    DeviceMem emptyDstMem = DeviceMem::create(outputMem.ptr(), 0);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher, emptyDstMem, emptySrcMem, stream));
    return HCCL_SUCCESS;
}

HcclResult ExecutorBase::CheckConcurrentDirectParameters(const u32 rank, const u32 rankSize,
                                                         const std::vector<LINK> &links)
{
    // 判断stream, dispatcher是否为空
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());

    // inputMem_ == outputMem_ 是允许的, 因为ring的时候收的slice和发的slice不是同一片
    // reduce scatter用inputMem_，allgather用outputMem_
    if (!outputMem_ || !inputMem_) {
        HCCL_ERROR("[ExecutorBase] rank[%u] run_async inputmem or outputmem is null", rank);
        return HCCL_E_PTR;
    }
    HCCL_INFO("ExecutorBase run: rank[%u] ranksize[%u] inputMem[%p] outputMem[%p] count[%llu]", rank, rankSize,
              inputMem_.ptr(), outputMem_.ptr(), count_);

    // 判断links数量是否正确
    CHK_PRT_RET(links.size() < rankSize,
                HCCL_ERROR("[ExecutorBase] rank[%u] link size[%u] is less than "
                           "rank size[%u]",
                           rank, links.size(), rankSize),
                HCCL_E_PARA);

    // 校验DataUnitSize
    if (DataUnitSize(dataType_) == 0) {
        HCCL_ERROR("[ExecutorBase] rank[%u] unit data size is zero", rank);
        return HCCL_E_INTERNAL;
    }

    HCCL_INFO("ExecutorBase finished to CheckParameters");
    return HCCL_SUCCESS;
}
}  // namespace hccl
