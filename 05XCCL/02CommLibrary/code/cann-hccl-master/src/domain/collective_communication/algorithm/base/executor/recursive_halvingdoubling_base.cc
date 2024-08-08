/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "recursive_halvingdoubling_base.h"

namespace hccl {
RecursiveHalvingDoublingBase::RecursiveHalvingDoublingBase(const HcclDispatcher dispatcher)
    : ExecutorBase(dispatcher), blockSize_(0), part1Size_(0), round_(0)
{
}

RecursiveHalvingDoublingBase::~RecursiveHalvingDoublingBase()
{
}

HcclResult RecursiveHalvingDoublingBase::CalcPartOneSizeAndBlockSize(const u32 rankSize)
{
    round_ = 0;
    u32 base = 1;
    const u32 minExponent = 1;
    while ((base << round_) <= rankSize) {
        round_++;
    }
    if (round_ >= minExponent) {
        round_ = round_ - minExponent;
    }
    blockSize_ = base << round_;
    part1Size_ = (rankSize - blockSize_) * 2; //  第一部分是rank数减block数乘2
    return HCCL_SUCCESS;
}

HcclResult RecursiveHalvingDoublingBase::BuildSubLinks(const std::vector<LINK> &links, std::vector<LINK> &subLinks,
                                                       u32 rankSize) const
{
    std::vector<LINK>::const_iterator iter = links.begin();
    subLinks.resize(blockSize_);

    for (u32 i = 0; i < rankSize; i++) {
        if (i < part1Size_ && (i % 2) == 1) {   // 模2余1代表当前rank在part1的奇数位置上，不参与block内的建链
            continue;
        } else if (i < part1Size_ && (i % 2) == 0) {  // 模2余0代表当前rank在part1的偶数位置上
            std::vector<LINK>::const_iterator niter = std::next(iter, i);
            if (niter != links.end()) {
                subLinks[i / 2] = *niter;              // 除2计算出在block内的rank号
            }
        } else {
            std::vector<LINK>::const_iterator niter = std::next(iter, i);
            if (niter != links.end()) {
                subLinks[i - part1Size_ / 2] = *niter; // rank在part2中，用原始rank减part1除2，计算出在block内的rank号
            }
        }
    }

    return HCCL_SUCCESS;
}

HcclResult RecursiveHalvingDoublingBase::CalculateSlices(u64 dataBytes) const
{
    CHK_PRT_RET(blockSize_ == 0, HCCL_ERROR("[Calculate][Slices]blocksize_ error"), HCCL_E_INTERNAL);

    slices_.resize(blockSize_);

    u64 bytesPerSlice = dataBytes / blockSize_;
    u64 reminder = dataBytes % blockSize_;
    if (reminder != 0) {
        bytesPerSlice++;
    }
    bytesPerSlice = ((bytesPerSlice + (HCCL_MIN_SLICE_ALIGN - 1)) / HCCL_MIN_SLICE_ALIGN) * HCCL_MIN_SLICE_ALIGN;

    u64 bytesLeft = dataBytes;
    u32 i = 0;
    while (bytesLeft > 0) {
        slices_[i].size = bytesPerSlice < bytesLeft ? bytesPerSlice : bytesLeft;
        slices_[i].offset = dataBytes - bytesLeft;

        bytesLeft -= slices_[i].size;
        i++;
    }

    while (i < blockSize_) {
        slices_[i].offset = dataBytes;
        slices_[i].size = 0;
        i++;
    }

    return HCCL_SUCCESS;
}
} // hccl
