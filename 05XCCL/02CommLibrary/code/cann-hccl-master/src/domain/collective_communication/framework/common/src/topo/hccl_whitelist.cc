/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_whitelist.h"
#include <fstream>
#include <string>
#include <map>
#include <hccl/base.h>

#include "comm.h"
#include "json_utils.h"

namespace hccl {
HcclWhitelist &HcclWhitelist::GetInstance()
{
    static HcclWhitelist wl;
    return wl;
}

HcclWhitelist::HcclWhitelist()
{
}

HcclWhitelist::~HcclWhitelist()
{
    std::unique_lock<std::mutex> lock(whiteListsMutex_);
    whiteLists_.clear();
}

HcclResult HcclWhitelist::GetHostWhiteList(std::vector<HcclIpAddress>& whiteList)
{
    std::unique_lock<std::mutex> lock(whiteListsMutex_);
    whiteList.clear();
    auto iter = whiteLists_.find(HcclWhiteListType::HCCL_WHITELIST_HOST);
    if (iter == whiteLists_.end()) {
        HCCL_INFO("GetHostWhiteList: white list is empty.");
        return HCCL_SUCCESS;
    }
    whiteList = whiteLists_[HcclWhiteListType::HCCL_WHITELIST_HOST];
    HCCL_INFO("GetHostWhiteList: whitelist length is %zu.", whiteList.size());
    return HCCL_SUCCESS;
}

HcclResult HcclWhitelist::LoadConfigFile(const std::string& realName)
{
    CHK_PRT_RET(realName.empty(), HCCL_ERROR("[Load][ConfigFile]whitelist file path is nullptr."), HCCL_E_PARA);
    std::unique_lock<std::mutex> lock(whiteListsMutex_);
    whiteLists_.clear();

    nlohmann::json fileContent;
    std::ifstream infile(realName.c_str(), std::ifstream::in);
    if (!infile) {
        HCCL_ERROR("[Load][ConfigFile]open file %s failed", realName.c_str());
        return HCCL_E_INTERNAL;
    } else {
        fileContent.clear();
        try {
            infile >> fileContent; // 将文件内容读取到json对象内
        } catch (...) {
            HCCL_ERROR("[Load][ConfigFile]errNo[0x%016llx] load file[%s] to json fail. please check json file format.",
                HCOM_ERROR_CODE(HCCL_E_INTERNAL), realName.c_str());
            infile.close();
            return HCCL_E_INTERNAL;
        }
    }
    infile.close();

    nlohmann::json hostWhitelist;
    CHK_RET(JsonUtils::GetJsonProperty(fileContent, "host_ip", hostWhitelist));

    for (auto& ipJson : hostWhitelist) {
        std::string ipStr;
        try {
            ipStr = ipJson.get<std::string>();
        } catch (...) {
            HCCL_ERROR("[Load][ConfigFile]errNo[0x%016llx]get ipStr from ipJson failed, please check host white list",
                HCOM_ERROR_CODE(HCCL_E_INTERNAL));
            return HCCL_E_PARA;
        }
        HcclIpAddress ip(ipStr);
        CHK_PRT_RET(ip.IsInvalid(), HCCL_ERROR("string[%s] is invalid ip", ipStr.c_str()), HCCL_E_PARA);
        whiteLists_[HcclWhiteListType::HCCL_WHITELIST_HOST].push_back(ip);
    }
    return HCCL_SUCCESS;
}
}
