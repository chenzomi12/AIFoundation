/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#ifndef REDUCE_SCATTER_HD_STAGE_PUB_H
#define REDUCE_SCATTER_HD_STAGE_PUB_H
 
#include "executor_base_pub.h"
#include "mem_host_pub.h"
#include "mem_device_pub.h"
#include "stream_pub.h"
#include "comm_base_pub.h"
namespace hccl {
class ReduceScatterHDStage : public ExecutorBase {
public:
    explicit  ReduceScatterHDStage(const HcclDispatcher dispatcher, const u64 reduceAttrBitMap,
        std::vector<Stream> &meshStreams, std::vector<std::shared_ptr<LocalNotify>> &meshSignal,
        std::vector<std::shared_ptr<LocalNotify>> &meshSignalAux, u32 userRank,
        const HcomCollOpInfo *opInfo);
    ~ ReduceScatterHDStage() override;
 
    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links) override;
 
protected:
private:
    HcclResult MainRecordSub(u32 streamNum);
    HcclResult SubWaitMain(u32 streamNum);
    HcclResult MainWaitSub(u32 streamNum);
    HcclResult SubRecordMain(u32 streamNum);
    HcclResult PrepareSliceData(u32 rankSize);
    HcclResult RunReduceScatterStage(u32 rank, u32 rankSize, const std::vector<LINK> &links);
    HcclResult RunReduceScatterStage1st(u32 rank, u32 rankSize, u32 peer, const std::vector<LINK> &links);
    HcclResult RunReduceScatterStage2nd(u32 rank, u32 rankSize, u32 peer2nd, u32 peer3rd,
        const std::vector<LINK> &links);
    HcclResult RunReduceScatterStageFinal(u32 rank, u32 rankSize, u32 peer, const std::vector<LINK> &links);
    inline u32 BackwardRank(u32 rank, u32 rankSize, u32 step) const
    {
        if (rankSize == 0) {
            return 0;
        }
        return (rank + rankSize - step) % rankSize;
    }
    const u64 reduceAttr_;
    const u32 base = 2;
    u32 userRank_;
    std::vector<Stream> meshStreams_;               /* * 多steam* */
    const std::vector<std::shared_ptr<LocalNotify>> &meshSignal_;    /* 每个ring创建一个signal */
    const std::vector<std::shared_ptr<LocalNotify>> &meshSignalAux_; /* 从stream wait，主steam record */
    const HcomCollOpInfo *opInfo_;
    std::map<u32, std::vector<Slice>> sliceMap;
};
} // namespace hccl
#endif /* REDUCE_SCATTER_HD_STAGE_H */