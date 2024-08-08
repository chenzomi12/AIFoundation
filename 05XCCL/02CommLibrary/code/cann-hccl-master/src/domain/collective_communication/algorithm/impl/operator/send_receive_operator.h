/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SEND_RECEIVE_OPERATOR_H
#define SEND_RECEIVE_OPERATOR_H

#include "coll_alg_operator.h"
#include <set>

namespace hccl {
class SendReceiveOperator : public CollAlgOperator {
public:
    SendReceiveOperator(std::unique_ptr<hcclImpl> &pImpl, std::unique_ptr<TopoMatcher> &topoMatcher);
    ~SendReceiveOperator();
    HcclResult SendRun(const std::string &tag, DeviceMem& inputPtr, u64 count, HcclDataType dataType,
        u32 destUserRank, Stream stream);
    HcclResult Send(const std::string &tag, void *inputPtr, u64 count, HcclDataType dataType,
        u32 destRank, Stream stream);
    HcclResult SendOutPlace(const std::string &tag, void *inputPtr, u64 count, HcclDataType dataType,
        u32 destRank, Stream stream);
    HcclResult ReceiveRun(const std::string &tag, DeviceMem &outputPtr, u64 count, HcclDataType dataType,
                          u32 srcUserRank, Stream &stream);
    HcclResult Receive(const std::string &tag, void *outputPtr, u64 count, HcclDataType dataType,
        u32 srcRank, Stream stream);
    HcclResult ReceiveOutPlace(const std::string &tag, void *outputPtr, u64 count, HcclDataType dataType,
        u32 srcRank, Stream stream);
private:
    HcclResult SendCommon(const std::string &tag, DeviceMem &inputPtr, u64 count,
        HcclDataType dataType, u32 destUserRank, Stream stream);
    HcclResult ReceiveCommon(const std::string &tag, DeviceMem &outputPtr, u64 count,
        HcclDataType dataType, u32 srcUserRank, Stream stream);
};
}
#endif /** __SEND_RECEIVE_OPERATOR_H__ */