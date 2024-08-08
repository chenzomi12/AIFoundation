/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SENDER_PUB_H
#define SENDER_PUB_H

#include <hccl/hccl_types.h>

#include "hccl_common.h"
#include "mem_device_pub.h"
#include "stream_pub.h"
#include "transport_pub.h"
#include "adapter_pub.h"

namespace hccl {
struct SenderMemoryInfo {
    u64 dstOffset;
    DeviceMem src;
};

class Sender {
public:
    Sender(const HcclDataType dataType, const HcclReduceOp reductionOp, const u64 reduceAttribute = 0x0);

    ~Sender();

    /**********************************************************************
     功能描述  : reduce操作的发送方内存发送操作
     输入参数  : const std::unique_ptr<Transport>& link: rank链接
               s32 dst_offset: 目的内存块的偏移量（Bytes）
               DeviceMem& src：源端内存块
               Stream& stream: 任务流
     输出参数  : 无
     返 回 值  : HcclResult
    **********************************************************************/
    HcclResult run(const std::shared_ptr<Transport> &link, const u64 dstOffset, DeviceMem &src, Stream &stream,
        const UserMemType dstMemType = UserMemType::INPUT_MEM) const;

    HcclResult run(const std::shared_ptr<Transport> &link, const std::vector<SenderMemoryInfo> &senderMems,
        Stream &stream) const;

protected:

private:
    const HcclDataType dataType_;
    const HcclReduceOp reductionOp_;
    const u64 reduceAttribute_;
};
}  // namespace hccl

#endif /* SENDER_PUB_H */
