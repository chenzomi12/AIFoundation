/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "json_utils.h"
#include "log.h"

namespace hccl {
HcclResult JsonUtils::GetJsonProperty(const nlohmann::json &obj, const std::string &propName, u32 &propValue)
{
    /* 查找json对象中是否有该属性, 不存在的属性不能直接访问 */
    CHK_PRT_RET(obj.find(propName) == obj.end(),
        HCCL_ERROR("[Get][JsonProperty]json object has no property called %s", propName.c_str()), HCCL_E_INTERNAL);

    /* 所有属性值都必须是字符串 */
    if (!obj[propName].is_number()) {
        HCCL_ERROR("[Get][JsonProperty]property value of Name[%s] is not number!", propName.c_str());
        return HCCL_E_INTERNAL;
    }

    propValue = obj[propName];
    return HCCL_SUCCESS;
}

HcclResult JsonUtils::GetJsonProperty(const nlohmann::json &obj, const std::string &propName, std::string &propValue)
{
    /* 查找json对象中是否有该属性, 不存在的属性不能直接访问 */
    CHK_PRT_RET(obj.find(propName) == obj.end(),
        HCCL_ERROR("[Get][JsonProperty]json object has no property called %s", propName.c_str()), HCCL_E_INTERNAL);

    /* 所有属性值都必须是字符串 */
    if (obj[propName].is_string()) {
        propValue = obj[propName];
        return HCCL_SUCCESS;
    } else {
        printf("property value of Name[%s] is not string!", propName.c_str());
        return HCCL_E_INTERNAL;
    }
}

HcclResult JsonUtils::GetJsonProperty(const nlohmann::json &obj, const std::string &propName, nlohmann::json &propValue)
{
    /* 查找json对象中是否有该属性, 不存在的属性不能直接访问 */
    CHK_PRT_RET(obj.find(propName) == obj.end(),
        HCCL_ERROR("[Get][JsonProperty]json object has no property called %s", propName.c_str()), HCCL_E_INTERNAL);

    propValue = obj[propName];
    return HCCL_SUCCESS;
}

HcclResult JsonUtils::ParseInformation(nlohmann::json &parseInformation, const std::string &information)
{
    try {
        parseInformation = nlohmann::json::parse(information);
    } catch (...) {
        HCCL_ERROR("[Parse][Information] errNo[0x%016llx] load allocated resource to json fail. "\
            "please check json input!", HCOM_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}
}