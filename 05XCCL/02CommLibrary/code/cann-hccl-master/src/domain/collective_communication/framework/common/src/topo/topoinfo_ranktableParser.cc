/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topoinfo_ranktableParser_pub.h"

#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <unistd.h>
#include <chrono>
#include <iostream>
#include <arpa/inet.h>

#include "hccl_comm_pub.h"
#include "topoinfo_ranktableStandard.h"
#include "topoinfo_ranktableConcise.h"
#include "topoinfo_ranktableHeterog.h"
#include "config.h"
#include "adapter_error_manager_pub.h"
#include "json_utils.h"

using namespace std;
using namespace hccl;

constexpr char TYPE_IP[] = "ip";
constexpr char TYPE_NAME[] = "Name";
constexpr char TYPE_ETH_NAME[] = "ethName";

constexpr u32 SERVERID_MIN = 0;         // 整形serverId最小值
constexpr u32 SERVERID_MAX = 0xFFFFFFFF;     // 整形serverId最大值

constexpr u32 HCCL_RANKTABLE_TIMEOUT_S = (30 * 60); // 读取ranktable json文件超时时间30 * 60s

const std::map<JsonUniqueInfoType, std::string> JsonInfoTypeNameMap = {
    {JsonUniqueInfoType::UNIQUE_INFO_TYPE_DEVICE_IP, "device_ip"},
    {JsonUniqueInfoType::UNIQUE_INFO_TYPE_SERVER_ID, "server_id"},
    {JsonUniqueInfoType::UNIQUE_INFO_TYPE_ETH_IP, "eth_ip"},
    {JsonUniqueInfoType::UNIQUE_INFO_TYPE_GROUP_NAME, "group_name"},
    {JsonUniqueInfoType::UNIQUE_INFO_TYPE_POD_NAME, "pod_name"},
    {JsonUniqueInfoType::UNIQUE_INFO_TYPE_ETH_NAME, "eth_name"},
    {JsonUniqueInfoType::UNIQUE_INFO_TYPE_SUPER_POD_ID, "super_pod_id"},
    {JsonUniqueInfoType::UNIQUE_INFO_NUM, "info_num"},
};

bool CheckFilePath(const std::string &filePath, const std::string &fileType)
{
    HCCL_DEBUG("file_path:%s,file_type:%s", filePath.c_str(), fileType.c_str());

    /* 如果file_path是file_type类型的文件，则file_path是以file_type结尾的 */
    return filePath.find(fileType) + fileType.length() == filePath.length();
}

HcclResult GetFileName(const std::string &filePath, const std::string &fileType, std::string &fileName)
{
    HCCL_DEBUG("file_path:%s,file_type:%s", filePath.c_str(), fileType.c_str());

    auto fi = filePath.find_last_of('/');
    if (fi != std::string::npos) {
        fileName = filePath.substr(fi + 1);
        fileName = fileName.substr(0, fileName.length() - fileType.length());
    }

    HCCL_DEBUG("get file_name:%s", fileName.c_str());

    return fileName.empty() ? HCCL_E_PARA : HCCL_SUCCESS;
}

HcclResult CheckAverageDev(u32 uDeviceNum, u32 uServerNum)
{
    if (uServerNum == 0) {
        HCCL_ERROR("[Check][AverageDev]errNo[0x%016llx] server num is zero", HCOM_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }
    // check the valid of average device per sever
    u32 averageDevice = uDeviceNum / uServerNum;

    // 如果device_num/server_num 不能被整除，报错
    // averageDevice不等于1 ，2，4，8... 报错
    if ((uDeviceNum % uServerNum) ||
        ((averageDevice != AVERAGE_DEVICE_SIXTEEN) &&
         (averageDevice != AVERAGE_DEVICE_EIGHT) &&
         (averageDevice != AVERAGE_DEVICE_FOUR)  &&
         (averageDevice != AVERAGE_DEVICE_TWO)   &&
         (averageDevice != AVERAGE_DEVICE_ONE))) {
        HCCL_ERROR("[Check][AverageDev]errNo[0x%016llx] device / server is invalid, uDeviceNum[%u], uServerNum[%u]",
            HCOM_ERROR_CODE(HCCL_E_PARA), uDeviceNum, uServerNum);
        return HCCL_E_PARA;
    }

    return HCCL_SUCCESS;
}

TopoInfoRanktableParser::TopoInfoRanktableParser(const std::string &rankTableM, const std::string &identify)
    : rankTableFile_(rankTableM), identify_(identify), statusCompleted_(false),
      uniqueInfoCheckPool_(static_cast<u32>(JsonUniqueInfoType::UNIQUE_INFO_NUM)), devMap_()
{
}

TopoInfoRanktableParser::~TopoInfoRanktableParser()
{
}

HcclResult TopoInfoRanktableParser::ReadFile(const std::string &readFile)
{
    std::string fileType = ".json";
    // 获取文件名字
    HcclResult ret = GetFileName(readFile, fileType, fileName_);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Read][File]get file[%s] name error", readFile.c_str()), HCCL_E_PARA);

    // 已只读方式打开该文件
    std::ifstream infile(readFile.c_str(), std::ifstream::in);
    if (!infile) {
        HCCL_ERROR("[Read][File]errNo[0x%016llx],open file %s failed",
            HCOM_ERROR_CODE(HCCL_E_INTERNAL), readFile.c_str());
        return HCCL_E_INTERNAL;
    } else {
        fileContent_.clear();
        try {
            infile >> fileContent_; // 将文件内容读取到json对象内
        } catch (...) {
            RPT_INPUT_ERR(true, "EI0004", std::vector<std::string>({"error_reason", "ranktable_path"}), \
                std::vector<std::string>({"Invalid ranktable format.", readFile.c_str()}));

            HCCL_ERROR("[Read][File]errNo[0x%016llx] load file[%s] to json fail. please check json file!",
                HCOM_ERROR_CODE(HCCL_E_INTERNAL), readFile.c_str());
            infile.close();
            return HCCL_E_INTERNAL;
        }
    }
    infile.close();
    return HCCL_SUCCESS;
}

HcclResult TopoInfoRanktableParser::LoadFile(const std::string &file)
{
    if (file.empty()) {
        HCCL_ERROR("[Load][File]errNo[0x%016llx] json file length is zero", HCOM_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }
    std::string fileType = ".json";

    if (!CheckFilePath(file, fileType)) {
        HCCL_ERROR("[Load][File]errNo[0x%016llx] path %s is not a valid %s file",
            HCOM_ERROR_CODE(HCCL_E_INTERNAL), file.c_str(), fileType.c_str());
        return HCCL_E_INTERNAL;
    }
    // 打开该文件前，判断该文件路径是否有效 规范
    char realFile[PATH_MAX] = {0};
    if (realpath(file.c_str(), realFile) == nullptr) {
        HCCL_ERROR("[Load][File]errNo[0x%016llx] path %s is not a valid real path",
            HCOM_ERROR_CODE(HCCL_E_PARA), file.c_str());
        return HCCL_E_PARA;
    }
    string strFilePath = std::string(realFile);
    const std::chrono::seconds TIMEOUT(HCCL_RANKTABLE_TIMEOUT_S);
    auto startTime = std::chrono::steady_clock::now();

    // load file and check the status
    HCCL_INFO("waiting for ranktable load complete...");

    // set timeout for status of file to be completed
    // 实验室场景下总是completed，cloud场景初始填入initlizing，kube补充完整后成为completed状态
    do {
        if ((std::chrono::steady_clock::now() - startTime) >= TIMEOUT) {
            HCCL_ERROR("[Load][File]errNo[0x%016llx] Load ranktable file[%s] timeout[%lld]s",
                HCOM_ERROR_CODE(HCCL_E_TIMEOUT), strFilePath.c_str(), TIMEOUT);
            return HCCL_E_TIMEOUT;
        }
        // 读取文件内容
        HcclResult ret = ReadFile(strFilePath);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Load][File]read file[%s] error", strFilePath.c_str()), HCCL_E_PARA);

        CHK_RET(RefreshStatus());

        if (IsReady()) {
            break;
        }

        SalSleep(1); // 每间隔一段时间去检测文件是否ready
    } while (true);

    HCCL_INFO("ranktable is ready");
    return HCCL_SUCCESS;
}

HcclResult TopoInfoRanktableParser::LoadRankTableString(const std::string &string)
{
    CHK_RET(LoadString(string));
    fileName_ = "rankTableString";
    CHK_RET(RefreshStatus());

    HCCL_INFO("ranktable is ready");
    return HCCL_SUCCESS;
}

HcclResult TopoInfoRanktableParser::LoadConfigString(const std::string &string)
{
    CHK_RET(LoadString(string));
    fileName_ = "configString";

    HCCL_INFO("ranktable is ready");
    return HCCL_SUCCESS;
}

HcclResult TopoInfoRanktableParser::LoadString(const std::string &string)
{
    // 入参检查
    if (string.empty()) {
        HCCL_ERROR("[Load][JsonString]errNo[0x%016llx] json string length is zero", HCOM_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }
    HCCL_INFO("waiting for json string load complete...");
    // 将字符串内容读取到json对象中
    CHK_RET(JsonUtils::ParseInformation(fileContent_, string));
    return HCCL_SUCCESS;
}

HcclResult TopoInfoRanktableParser::GetClusterInfo(RankTable_t &clusterInfo)
{
    return HCCL_SUCCESS;
}
HcclResult TopoInfoRanktableParser::GetClusterInfo(hccl::HcclCommParams &params,
    hccl::RankTable_t &rankTable)
{
    return HCCL_SUCCESS;
}

HcclResult TopoInfoRanktableParser::Init()
{
    HcclResult ret = LoadRankTableString(rankTableFile_);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Init][TopoInfoRanktableParser]load string[%s] error", fileName_.c_str()), ret);

    return HCCL_SUCCESS;
}

HcclResult TopoInfoRanktableParser::LoadFileInit(std::string &rankTableM)
{
    HcclResult ret = LoadFile(rankTableFile_);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Init][TopoInfoRanktableParser]load file[%s] error", rankTableFile_.c_str()), ret);

    rankTableM = fileContent_.dump();
    return HCCL_SUCCESS;
}

HcclResult TopoInfoRanktableParser::GetRanktableVersion(std::string &version)
{
    /* 查找json对象中是否有该属性, 不存在的属性不能直接访问 */
    if (fileContent_.find(CLUSTER_PROP_VERSION) == fileContent_.end()) {
        version = "Standard";
        HCCL_DEBUG("RankTableVersion is not found, Parser: %s", version.c_str());
    } else {
        CHK_RET(GetJsonProperty(fileContent_, CLUSTER_PROP_VERSION, version_));
        version = version_;
        HCCL_DEBUG("%s.json -> version: %s", fileName_.c_str(), version.c_str());
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoRanktableParser::RefreshStatus()
{
    std::string status = "";
    CHK_RET(GetJsonProperty(fileContent_, "status", status));
    HCCL_DEBUG("%s.json -> status: %s", fileName_.c_str(), status.c_str());
    statusCompleted_ = (status == "completed");
    return HCCL_SUCCESS;
}

bool TopoInfoRanktableParser::IsReady() const
{
    return this->statusCompleted_;
}

HcclResult TopoInfoRanktableParser::GetJsonProperty(const nlohmann::json &obj, const char *propName,
    std::string &propValue) const
{
    /* 查找json对象中是否有该属性, 不存在的属性不能直接访问 */
    if (obj.find(propName) == obj.end()) {
        HCCL_WARNING("json object has no property called %s", propName);
        return HCCL_E_NOT_FOUND;
    }

    if (obj[propName].is_string()) {
        propValue = obj[propName];
        return HCCL_SUCCESS;
    } else {
        HCCL_ERROR("[Get][JsonProperty]errNo[0x%016llx] json object property value of Name[%s] is not string!",
            HCOM_ERROR_CODE(HCCL_E_PARA), propName);
        return HCCL_E_PARA;
    }
}

HcclResult TopoInfoRanktableParser::GetJsonProperty(const nlohmann::json &obj, const char *propName,
    nlohmann::json &propValue) const
{
    /* 查找json对象中是否有该属性, 不存在的属性不能直接访问 */
    if (obj.find(propName) == obj.end()) {
        HCCL_WARNING("json object has no property called %s", propName);
        return HCCL_E_NOT_FOUND;
    }
    propValue = obj[propName];
    CHK_PRT_RET(propValue.size() == 0, HCCL_ERROR("[Get][JsonProperty]get property[%s] size is zero", propName),
        HCCL_E_PARA);
    return HCCL_SUCCESS;
}

HcclResult TopoInfoRanktableParser::GetJsonArrayMemberProperty(const nlohmann::json &obj, const u32 index,
    const char *propName, std::string &propValue) const
{
    if (!obj.is_array() || index >= obj.size()) {
        HCCL_ERROR("[Get][JsonArrayMemberProperty]errNo[0x%016llx] index[%u] is out of json object range",
            HCOM_ERROR_CODE(HCCL_E_NOT_FOUND), index);
        return HCCL_E_PARA;
    }

    nlohmann::json subObj = obj.at(index);
    if (subObj.find(propName) == subObj.end()) {
        HCCL_WARNING("json object index[%u] has no property called %s", index, propName);
        return HCCL_E_NOT_FOUND;
    }
    if (subObj[propName].is_string()) {
        propValue = subObj[propName];
        return HCCL_SUCCESS;
    } else {
        std::string buffer = obj.dump(2);
        HCCL_ERROR("[Get][JsonArrayMemberProperty]errNo[0x%016llx] json[%s] object property value of Name[%s] is "\
            "not string!", HCOM_ERROR_CODE(HCCL_E_PARA), buffer.c_str(), propName);
        return HCCL_E_PARA;
    }
}

HcclResult TopoInfoRanktableParser::GetJsonArrayMemberProperty(const nlohmann::json &obj, const u32 index,
    const char *propName, u32 &propValue)
{
    if (!obj.is_array() || index >= obj.size()) {
        HCCL_ERROR("[Get][JsonArrayMemberProperty]errNo[0x%016llx] index[%u] is out of json object range",
            HCOM_ERROR_CODE(HCCL_E_NOT_FOUND), index);
        return HCCL_E_PARA;
    }

    nlohmann::json subObj = obj.at(index);
    if (subObj.find(propName) == subObj.end()) {
        HCCL_WARNING("json object index[%u] has no property called %s", index, propName);
        return HCCL_E_NOT_FOUND;
    }
    if (subObj[propName].is_number_unsigned()) {
        propValue = subObj[propName];
        return HCCL_SUCCESS;
    } else {
        HCCL_ERROR("[Get][JsonArrayMemberProperty]errNo[0x%016llx] json object property value of Name[%s] is "\
            "not unsigned int!", HCOM_ERROR_CODE(HCCL_E_PARA), propName);
        return HCCL_E_PARA;
    }
}

HcclResult TopoInfoRanktableParser::GetJsonArrayMemberProperty(const nlohmann::json &obj, const u32 index,
    const char *propName, nlohmann::json &propValue) const
{
    if (!obj.is_array() || index >= obj.size()) {
        HCCL_ERROR("[Get][JsonArrayMemberProperty]errNo[0x%016llx] index[%u] is out of json object range",
            HCOM_ERROR_CODE(HCCL_E_NOT_FOUND), index);
        return HCCL_E_PARA;
    }

    nlohmann::json subObj = obj.at(index);
    if (subObj.find(propName) == subObj.end()) {
        HCCL_WARNING("json object index[%u] has no property called %s", index, propName);
        return HCCL_E_NOT_FOUND;
    }
    propValue = subObj[propName];
    CHK_PRT_RET(propValue.size() == 0, HCCL_ERROR("[Get][JsonArrayMemberProperty]get index[%u] property[%s] size is "\
        "zero", index, propName), HCCL_E_PARA);
    return HCCL_SUCCESS;
}

/* 依据检查类型进行入参内容检查，并将检查选项转为strType带出以便后续信息打印 */
HcclResult TopoInfoRanktableParser::CheckUniquePara(const JsonUniqueInfoType &type, const std::string &value,
    string &strType) const
{
    auto it = JsonInfoTypeNameMap.find(type);
    CHK_PRT_RET(it == JsonInfoTypeNameMap.end(),
        HCCL_ERROR("[Get][JsonArrayMemberProperty][Check][UniquePara]errNo[0x%016llx] jsonUniqueInfoType[%d] "\
        " is not in JsonInfoTypeNameMap", HCOM_ERROR_CODE(HCCL_E_PARA), type), HCCL_E_PARA);
    strType = it->second;
    switch (type) {
        case JsonUniqueInfoType::UNIQUE_INFO_TYPE_DEVICE_IP:
        case JsonUniqueInfoType::UNIQUE_INFO_TYPE_ETH_IP: {
            /* 必须是一个有效的ip地址 */
            HcclIpAddress ip(value);
            if (ip.IsInvalid()) {
                RPT_INPUT_ERR(true, "EI0004", std::vector<std::string>({ "error_reason", "ranktable_path" }),
                    std::vector<std::string>({ "The ip in ranktable is not a valid ip address",
                    "The ranktable path configured in the training can be found in the plogs." }));
                HCCL_ERROR("[Check][UniquePara]errNo[0x%016llx] ip[%s] is not a valid ip address",
                    HCOM_ERROR_CODE(HCCL_E_PARA), value.c_str());
                return HCCL_E_PARA;
            }
        }
            break;

        case JsonUniqueInfoType::UNIQUE_INFO_TYPE_SERVER_ID:
        case JsonUniqueInfoType::UNIQUE_INFO_TYPE_GROUP_NAME:
        case JsonUniqueInfoType::UNIQUE_INFO_TYPE_SUPER_POD_ID:
        case JsonUniqueInfoType::UNIQUE_INFO_TYPE_POD_NAME: {
            if (value.length() > GROUP_NAME_MAX_LEN) {
                HCCL_ERROR("[Check][UniquePara]errNo[0x%016llx] Name[%s] length[%lu] is too long",
                    HCOM_ERROR_CODE(HCCL_E_PARA), value.c_str(), value.length());
                return HCCL_E_PARA;
            }
        }
            break;

        case JsonUniqueInfoType::UNIQUE_INFO_TYPE_ETH_NAME:
            break;

        default: {
            HCCL_ERROR("[Check][UniquePara]errNo[0x%016llx] invalid unique info type[%d]",
                HCOM_ERROR_CODE(HCCL_E_PARA), type);
            return HCCL_E_PARA;
        }
            break;
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoRanktableParser::CheckUniqueAndInsertPool(const JsonUniqueInfoType &type,
    const std::string &value, const JsonCheckOpType &opType)
{
    string strUniqueInfoType;
    /* 先依据type检查入参内容合法 */
    CHK_RET(CheckUniquePara(type, value, strUniqueInfoType));
    /* 关键信息添加保存 */
    auto srvIt = uniqueInfoCheckPool_[static_cast<u32>(type)].find(value);
    if (opType == JsonCheckOpType::CHECK_OP_TYPE_INSERT) {
        /* JsonCheckOpType::CHECK_OP_TYPE_INSERT操作插入保存到资源池中，若资源池中存在则报错 */
        if (srvIt != uniqueInfoCheckPool_[static_cast<u32>(type)].end()) {
            RPT_INPUT_ERR(true, "EI0004", std::vector<std::string>({ "error_reason", "ranktable_path" }),
                std::vector<std::string>({ "The IP is used repeatedly.", "The ranktable path configured "
                "in the training can be found in the plogs." }));
            HCCL_ERROR("[Check][UniqueAndInsertPool]errNo[0x%016llx] [%s]:[%s] is already exist",
                HCOM_ERROR_CODE(HCCL_E_PARA), strUniqueInfoType.c_str(), value.c_str());
            return HCCL_E_PARA;
        }
        uniqueInfoCheckPool_[static_cast<u32>(type)].insert(value);
    } else if (opType == JsonCheckOpType::CHECK_OP_TYPE_FIND) {
        /* JsonCheckOpType::CHECK_OP_TYPE_FIND操作仅在资源池中查找是否存在，不存在报错 */
        if (srvIt == uniqueInfoCheckPool_[static_cast<u32>(type)].end()) {
            HCCL_ERROR("[Check][UniqueAndInsertPool]errNo[0x%016llx] [%s]:[%s] is not exist",
                HCOM_ERROR_CODE(HCCL_E_PARA), strUniqueInfoType.c_str(), value.c_str());
            return HCCL_E_NOT_FOUND;
        }
    } else {
        HCCL_ERROR("[Check][UniqueAndInsertPool]errNo[0x%016llx] invalid json check op Type[%d]",
            HCOM_ERROR_CODE(HCCL_E_PARA), opType);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

void TopoInfoRanktableParser::GenerateServerIdx(const std::string &serverId, u32 &serverIdx)
{
    auto it = find(ServerIdRecord_.begin(), ServerIdRecord_.end(), serverId);
    if (it == ServerIdRecord_.end()) {
        serverIdx = ServerIdRecord_.size();
        ServerIdRecord_.push_back(serverId);
    } else {
        serverIdx = std::distance(ServerIdRecord_.begin(), it);
    }
}

HcclResult TopoInfoRanktableParser::CheckUniqueIntegerAndInsertPool(const std::string &serverId)
{
    u32 serverIdValue = 0;
    if (SalStrToULong(serverId, HCCL_BASE_DECIMAL, serverIdValue) != HCCL_SUCCESS) {
        RPT_INPUT_ERR(true, "EI0004", std::vector<std::string>({ "error_reason", "ranktable_path" }),
            std::vector<std::string>({ "The serverId in ranktable is invalid.", "The ranktable path configured "
            "in the training can be found in the plogs." }));
        HCCL_ERROR("[Check][UniqueIntegerAndInsertPool]errNo[0x%016llx] serverId[%s] is invalid.",
            HCOM_ERROR_CODE(HCCL_E_PARA), serverId.c_str());
        return HCCL_E_PARA;
    }
    if (serverIdValue < SERVERID_MIN || serverIdValue > SERVERID_MAX) {
        HCCL_ERROR("[Check][UniqueIntegerAndInsertPool]errNo[0x%016llx] The value range of " \
            "serverId:[%u] must be 0 to 4294967295.", HCOM_ERROR_CODE(HCCL_E_PARA), serverIdValue);
        return HCCL_E_PARA;
    }
    auto srvIt = uniqueInfoCheckPool_[static_cast<u32>(JsonUniqueInfoType::UNIQUE_INFO_TYPE_SERVER_ID)].find(serverId);
        /* JsonCheckOpType::CHECK_OP_TYPE_INSERT操作插入保存到资源池中，若资源池中存在则报错 */
    if (srvIt != uniqueInfoCheckPool_[static_cast<u32>(JsonUniqueInfoType::UNIQUE_INFO_TYPE_SERVER_ID)].end()) {
        RPT_INPUT_ERR(true, "EI0004", std::vector<std::string>({ "error_reason", "ranktable_path" }),
            std::vector<std::string>({ "The serverId is used repeatedly.", "The ranktable path configured "
            "in the training can be found in the plogs." }));
        HCCL_ERROR("[Check][UniqueIntegerAndInsertPool]errNo[0x%016llx] serverId:[%s] is already exist",
            HCOM_ERROR_CODE(HCCL_E_PARA), serverId.c_str());
        return HCCL_E_PARA;
    }
    uniqueInfoCheckPool_[static_cast<u32>(JsonUniqueInfoType::UNIQUE_INFO_TYPE_SERVER_ID)].insert(serverId);
    return HCCL_SUCCESS;
}

HcclResult TopoInfoRanktableParser::ConvertIpAddress(const std::string &ipStr, HcclIpAddress &ipAddr)
{
    HcclResult ret = ipAddr.SetReadableAddress(ipStr);
    RPT_INPUT_ERR(ret != HCCL_SUCCESS, "EI0004", std::vector<std::string>({ "error_reason", "ranktable_path" }),
        std::vector<std::string>({
            "The ips in ranktable is invalid",
            "The ranktable path configured in the training can be found in the plogs."
        }));
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Convert][IpAddress]ipStr[%s] convert failed", ipStr.c_str()), ret);
    return ret;
}
