/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#ifndef RECEIVE_OPERATOR_H
#define RECEIVE_OPERATOR_H
 
#include "common_operator.h"
#include <set>
#include "coll_alg_op_registry.h"
 
namespace hccl {
class ReceiveOperator : public CollAlgOperator {
public:
    ReceiveOperator(std::unique_ptr<hcclImpl> &pImpl, std::unique_ptr<TopoMatcher> &topoMatcher);
    ~ReceiveOperator();
 
    HcclResult SelectAlg(const std::string& tag, const OpParam& param, std::string& algName, std::string& newTag);
};
}
 
 
#endif /** __RECEIVE_OPERATOR_H__ */