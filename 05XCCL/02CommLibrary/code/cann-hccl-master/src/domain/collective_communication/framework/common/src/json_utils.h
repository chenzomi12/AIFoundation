/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_JSON_UTILS_H
#define HCCL_JSON_UTILS_H

#include <map>
#include <string>
#include <nlohmann/json.hpp>
#include <hccl/base.h>

namespace hccl {
class JsonUtils {
public:
    JsonUtils() = default;
    ~JsonUtils() = default;
    static HcclResult GetJsonProperty(const nlohmann::json &obj, const std::string &propName, u32 &propValue);
    static HcclResult GetJsonProperty(const nlohmann::json &obj, const std::string &propName, std::string &propValue);
    static HcclResult GetJsonProperty(const nlohmann::json &obj, const std::string &propName,
        nlohmann::json &propValue);
    static HcclResult ParseInformation(nlohmann::json &parseInformation, const std::string &information);
};
}

#endif // end HCCL_JSON_UTILS_H