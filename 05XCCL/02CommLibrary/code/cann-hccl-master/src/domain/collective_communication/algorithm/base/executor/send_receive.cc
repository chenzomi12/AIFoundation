/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "send_receive.h"

namespace hccl {
SendReceive::SendReceive(
    const HcclDispatcher dispatcher,
    const std::shared_ptr<Transport> &link,
    const u32 peerRank,
    const u64 chunkNum)
    : ExecutorBase(dispatcher),
      transLink_(link),
      peerRank_(peerRank),
      chunkSize_(chunkNum)
{
}

SendReceive::~SendReceive()
{
}

HcclResult SendReceive::SendPrepare(
    const DeviceMem &inputMem,
    const u32 destRank,
    const Stream &stream)
{
    /* 参数赋值 */
    inputMem_ = inputMem;
    stream_ = stream;
    peerRank_ = destRank;
    return HCCL_SUCCESS;
}

HcclResult SendReceive::ReceivePrepare(
    const DeviceMem &outputMem,
    const u32 srcRank,
    const Stream &stream)
{
    /* 参数赋值 */
    outputMem_ = outputMem;
    stream_ = stream;
    peerRank_ = srcRank;

    return HCCL_SUCCESS;
}

HcclResult SendReceive::SendRunAsync()
{
    if (!inputMem_) {
        HCCL_ERROR("[SendReceive][SendRunAsync]SendRunAsync inputmem is null");
        return HCCL_E_PTR;
    }

    CHK_SMART_PTR_NULL(transLink_);
    u64 sizePerRound = 0;
    u64 sizePerSlice = chunkSize_;
    u64 length = inputMem_.size();
    u64 offset = 0;
    for (u64 sizeResidue = length; sizeResidue > 0; sizeResidue -= sizePerRound) {
        HcclResult ret = transLink_->TxAck(stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[SendReceive][SendRunAsync]tx ack run failed"), ret);
        ret = transLink_->RxAck(stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[SendReceive][SendRunAsync]rx ack run failed"), ret);

        offset += sizePerRound;
        sizePerRound = (sizeResidue > sizePerSlice) ? sizePerSlice : sizeResidue;
        void* localAddr = static_cast<u8 *>(inputMem_.ptr()) + offset;
        HCCL_DEBUG("tx async inputmem's offset[%llu] size[%llu]", offset, sizePerRound);

        ret = transLink_->TxAsync(UserMemType::OUTPUT_MEM, offset, localAddr, sizePerRound, stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[SendReceive][SendRunAsync]tx async offset[%llu] "\
            "size[%llu] failed", offset, sizePerRound), ret);

        ret = transLink_->RxAsync(UserMemType::OUTPUT_MEM, 0, nullptr, 0, stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[SendReceive][ReceiveRunAsync]tx async failed"), ret);

        ret = transLink_->RxWaitDone(stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[SendReceive][SendRunAsync]RxWaitDone failed"), ret);
        ret = transLink_->TxWaitDone(stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[SendReceive][SendRunAsync]TxWaitDone failed"), ret);
    }
    return HCCL_SUCCESS;
}

HcclResult SendReceive::ReceiveRunAsync()
{
    if (!outputMem_) {
        HCCL_ERROR("[SendReceive][ReceiveRunAsync]ReceiveRunAsync outputmem is null");
        return HCCL_E_PTR;
    }
    CHK_SMART_PTR_NULL(transLink_);

    u64 sizePerRound = 0;
    u64 sizePerSlice = chunkSize_;
    u64 length = outputMem_.size();
    u64 offset = 0;
    for (u64 sizeResidue = length; sizeResidue > 0; sizeResidue -= sizePerRound) {
        HcclResult ret = transLink_->TxAck(stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[SendReceive][ReceiveRunAsync]tx ack failed"), ret);
        ret = transLink_->RxAck(stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[SendReceive][ReceiveRunAsync]rx ack failed"), ret);

        offset += sizePerRound;
        sizePerRound = (sizeResidue > sizePerSlice) ? sizePerSlice : sizeResidue;
        void* localAddr = static_cast<u8 *>(outputMem_.ptr()) + offset;
        HCCL_DEBUG("rx async outputmem's offset[%llu] size[%llu]", offset, sizePerRound);

        ret = transLink_->RxAsync(UserMemType::OUTPUT_MEM, offset, localAddr, sizePerRound, stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[SendReceive][ReceiveRunAsync]rx async with offset[%llu] "\
            "size[%llu] failed", offset, sizePerRound), ret);

        ret = transLink_->TxAsync(UserMemType::OUTPUT_MEM, 0, nullptr, 0, stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[SendReceive][ReceiveRunAsync]tx async failed"), ret);
    }
    return HCCL_SUCCESS;
}

HcclResult SendReceive::BatchSendRunAsync()
{
    if (!inputMem_) {
        HCCL_ERROR("[SendReceive][BatchSendRunAsync] BatchSendRunAsync inputmem is null");
        return HCCL_E_PTR;
    }

    CHK_SMART_PTR_NULL(transLink_);
    u64 sizePerRound = 0;
    u64 sizePerSlice = chunkSize_;
    u64 length = inputMem_.size();
    u64 offset = 0;
    for (u64 sizeResidue = length; sizeResidue > 0; sizeResidue -= sizePerRound) {
        HcclResult ret = transLink_->TxPrepare(stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[SendReceive][BatchSendRunAsync]tx ack run failed"), ret);

        offset += sizePerRound;
        sizePerRound = (sizeResidue > sizePerSlice) ? sizePerSlice : sizeResidue;
        void* localAddr = static_cast<u8 *>(inputMem_.ptr()) + offset;
        HCCL_DEBUG("tx async inputmem's offset[%llu] size[%llu]", offset, sizePerRound);

        ret = transLink_->TxData(UserMemType::OUTPUT_MEM, offset, localAddr, sizePerRound, stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[SendReceive][BatchSendRunAsync]tx async offset[%llu] "\
            "size[%llu] failed", offset, sizePerRound), ret);

        ret = transLink_->TxDone(stream_);
  
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[SendReceive][BatchSendRunAsync]TxWaitDone failed"), ret);
    }
    return HCCL_SUCCESS;
}

HcclResult SendReceive::BatchReceiveRunAsync()
{
    if (!outputMem_) {
        HCCL_ERROR("[SendReceive][ReceiveRunAsync]ReceiveRunAsync outputmem is null");
        return HCCL_E_PTR;
    }
    CHK_SMART_PTR_NULL(transLink_);

    u64 sizePerRound = 0;
    u64 sizePerSlice = chunkSize_;
    u64 length = outputMem_.size();
    u64 offset = 0;
    for (u64 sizeResidue = length; sizeResidue > 0; sizeResidue -= sizePerRound) {
        HcclResult ret = transLink_->RxPrepare(stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[SendReceive][BatchReceiveRunAsync]rx ack failed"), ret);

        offset += sizePerRound;
        sizePerRound = (sizeResidue > sizePerSlice) ? sizePerSlice : sizeResidue;
        void* localAddr = static_cast<u8 *>(outputMem_.ptr()) + offset;
        HCCL_DEBUG("rx async outputmem's offset[%llu] size[%llu]", offset, sizePerRound);

        ret = transLink_->RxData(UserMemType::INPUT_MEM, offset, localAddr, sizePerRound, stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[SendReceive][BatchReceiveRunAsync]rx async with offset[%llu] "\
            "size[%llu] failed", offset, sizePerRound), ret);

        ret = transLink_->RxDone(stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[SendReceive][BatchReceiveRunAsync]TxDataSignal offset[%llu]"\
            "size[%llu] failed", offset, sizePerRound), ret);
    }
    return HCCL_SUCCESS;
}
} // namespace hccl

