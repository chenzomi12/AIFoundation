/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOPOINFO_RANKTABLEPARSER_H
#define TOPOINFO_RANKTABLEPARSER_H

#include "nlohmann/json.hpp"
#include "topoinfo_struct.h"
#include "base.h"
#include "hccl_types.h"
#include "sal_pub.h"
#include "comm.h"

constexpr char CLUSTER_PROP_VERSION[] = "version";
constexpr char HCCL_CLUSTER_VERSION[] = "1.0";      // HCCL集群版本
constexpr char HETEROG_CLUSTER_VERSION[] = "1.1";   // 异构集群版本
constexpr char SUPERPOD_CLUSTER_VERSION[] = "1.2";      // 超节点集群版本

constexpr int AVERAGE_DEVICE_SIXTEEN = 16;
constexpr int AVERAGE_DEVICE_EIGHT = 8;
constexpr int AVERAGE_DEVICE_FOUR = 4;
constexpr int AVERAGE_DEVICE_TWO = 2;
constexpr int AVERAGE_DEVICE_ONE = 1;
constexpr int SERVER_BOARD_ID_MASK_BIT = 0xFFFFFFF0;  // X86与arm环境无需比较后4位
constexpr int EVB_BOARD_ID_MASK_BIT = 0xFFFFF000;  // EVB环境只需比较前2位
constexpr int ITEM_BOADR_ID_LEN = 6;  // ranktable内BoardID字段的长度
constexpr u32 COLLECTIVEID_MAX_LEN = 127; // 最大的collectiveId的长度

// roleTable字段
constexpr char NODE_ID[] = "id";
constexpr char NODE_IP[] = "IP";
constexpr char NODE_PORT[] = "port";
constexpr char SERVER_LIST[] = "Servers";
constexpr char CLIENT_LIST[] = "Clients";

HcclResult GetFileName(const std::string &filePath, const std::string &fileType, std::string &fileName);
bool CheckFilePath(const std::string &filePath, const std::string &fileType);
HcclResult CheckAverageDev(u32 uDeviceNum, u32 uServerNum);

namespace hccl {
enum class JsonUniqueInfoType {
    UNIQUE_INFO_TYPE_DEVICE_IP = 0,
    UNIQUE_INFO_TYPE_SERVER_ID = 1,
    UNIQUE_INFO_TYPE_ETH_IP   = 2,
    UNIQUE_INFO_TYPE_GROUP_NAME = 3,
    UNIQUE_INFO_TYPE_POD_NAME = 4,
    UNIQUE_INFO_TYPE_ETH_NAME = 5,
    UNIQUE_INFO_TYPE_SUPER_POD_ID = 6,
    UNIQUE_INFO_NUM
};

enum class ServerHostNicIndex {
    HOST_NIC_IP_INDEX = 0,
    HOST_NIC_PORT_INDEX = 1,
    HOST_NIC_INDEX_NUM
};

enum class JsonCheckOpType {
    CHECK_OP_TYPE_FIND = 0,
    CHECK_OP_TYPE_INSERT = 1,
    CHECK_OP_TYPE_NUM
};

class TopoInfoRanktableParser {
public:

    explicit TopoInfoRanktableParser(const std::string &rankTableM, const std::string &identify);
    virtual ~TopoInfoRanktableParser();

    virtual HcclResult Init();
    virtual HcclResult GetClusterInfo(RankTable_t &clusterInfo);
    virtual HcclResult GetClusterInfo(hccl::HcclCommParams &params,
        hccl::RankTable_t &rankTable);
    HcclResult GetRanktableVersion(std::string &version);
    HcclResult LoadFileInit(std::string &rankTableM);
protected:

    nlohmann::json fileContent_;  // json文件的内容
    std::string rankTableFile_;
    std::string version_;         // rankTable版本信息
    std::string identify_;
    std::string fileName_;
    bool statusCompleted_;        // reveal the "status" in rankTableFile
    std::vector<std::set<std::string>> uniqueInfoCheckPool_;
    // 整个通信域内dev的信息(kname为服务器的server_id,按照服务器区分)
    std::map<std::string, std::vector<hccl::RankInfo_t> > devMap_;

    std::unique_ptr<TopoInfoRanktableParser> topoRanktable_;

    HcclResult LoadFile(const std::string &file);
    HcclResult LoadRankTableString(const std::string &string);
    HcclResult LoadConfigString(const std::string &string);
    /* 访问json信息的一个名为prop_name的属性，参数返回这个属性的值prop_value，属性值重载了string和json类型 */
    HcclResult GetJsonProperty(const nlohmann::json &obj, const char *propName, std::string &propValue) const;
    HcclResult GetJsonProperty(const nlohmann::json &obj, const char *propName, nlohmann::json &propValue) const;
    // json数组中索引为index的一个名为prop_name的属性
    // 数返回这个属性的值prop_value，属性值重载了string和json类型
    HcclResult GetJsonArrayMemberProperty(const nlohmann::json &obj, const u32 index, const char *propName,
                                                std::string &propValue) const;
    HcclResult GetJsonArrayMemberProperty(const nlohmann::json &obj, const u32 index, const char *propName,
                                                nlohmann::json &propValue) const;
    HcclResult GetJsonArrayMemberProperty(const nlohmann::json &obj, const u32 index, const char *propName,
                                                u32 &propValue);

    HcclResult CheckUniquePara(const JsonUniqueInfoType &type, const std::string &value, std::string &strType) const;
    HcclResult CheckUniqueAndInsertPool(const JsonUniqueInfoType &type, const std::string &value,
        const JsonCheckOpType &opType);
    void GenerateServerIdx(const std::string &serverId, u32 &serverIdx);
    HcclResult CheckUniqueIntegerAndInsertPool(const std::string &serverId);
    HcclResult ConvertIpAddress(const std::string &ipStr, HcclIpAddress &ipAddr);
    // 所有集群信息
    hccl::HcclCommParams params_;
    hccl::RankTable_t rankTable_;

    std::vector<std::string> ServerIdRecord_;

private:
    TopoInfoRanktableParser(const TopoInfoRanktableParser&);
    TopoInfoRanktableParser& operator=(const TopoInfoRanktableParser&);
    
    static bool CompareWithDevicePhyId(const RankInfo_t &left, const RankInfo_t &right);
    bool IsReady() const;
    HcclResult RefreshStatus();
    HcclResult ReadFile(const std::string &readFile);
    HcclResult LoadString(const std::string &string);
};
}  // namespace hccl

#endif
