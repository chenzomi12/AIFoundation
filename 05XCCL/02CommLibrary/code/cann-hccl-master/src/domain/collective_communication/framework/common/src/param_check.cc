/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log.h"

#include "base.h"
#include "rank_consistent.h"
#include "topoinfo_ranktableParser_pub.h"
#include "config.h"
#include "param_check.h"

using namespace std;
using namespace hccl;

HcclResult HcomGetRanktableRealPath(const char *rankTable, std::string &realFilePath)
{
    CHK_PTR_NULL(rankTable);

    u32 rankTablePathLen = strnlen(rankTable, RANK_TABLE_MAX_LEN + 1);
    if (rankTablePathLen == (RANK_TABLE_MAX_LEN + 1) || rankTablePathLen == 0) {
        HCCL_ERROR("[Get][RanktableRealPath]errNo[0x%016llx] rankTable file name is invalid, len is %u",
            HCOM_ERROR_CODE(HCCL_E_PARA), rankTablePathLen);
        return HCCL_E_PARA;
    }
    // 校验文件是否存在
    char realFile[PATH_MAX] = {0};
    if (realpath(rankTable, realFile) == nullptr) {
        HCCL_ERROR("[Get][RanktableRealPath]errNo[0x%016llx] path %s is not a valid real path",
            HCOM_ERROR_CODE(HCCL_E_PARA), rankTable);
        return HCCL_E_PARA;
    }
    realFilePath = std::string(realFile);
    return HCCL_SUCCESS;
}

HcclResult HcomCheckRankTable(const char *rankTableM, u32 &rankTableSize)
{
    CHK_PTR_NULL(rankTableM);

    size_t rankTableLen = strnlen(rankTableM, STRING_MAX_LENGTH + 1);
    if (rankTableLen == (STRING_MAX_LENGTH + 1) || rankTableLen == 0) {
        HCCL_ERROR("[HcomCheckRankTable]errNo[0x%016llx] rankTable string is invalid, len is %u",
            HCOM_ERROR_CODE(HCCL_E_PARA), rankTableLen);
        return HCCL_E_PARA;
    }

    rankTableSize = rankTableLen;
    return HCCL_SUCCESS;
}

HcclResult HcomLoadRanktableFile(const char *rankTablePath, std::string &rankTableM, std::string &realFilePath)
{
    CHK_PTR_NULL(rankTablePath);

    HcclResult ret = HcomGetRanktableRealPath(rankTablePath, realFilePath);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[HcomLoadRanktableFile]get file[%s] real path error", rankTablePath),
        HCCL_E_PARA);
    TopoInfoRanktableParser myTopoRanktable(realFilePath, "0");
    CHK_RET(myTopoRanktable.LoadFileInit(rankTableM));

    return HCCL_SUCCESS;
}

HcclResult HcomCalcCRC(hccl::HcclCommParams &params, const char *rankTable)
{
    (void)params;
    CHK_RET(RankConsistent::GetInstance().RecordRankTableCrc(rankTable));
    return HCCL_SUCCESS;
}

HcclResult HcomCheckIdentify(const char *identify)
{
    CHK_PTR_NULL(identify);

    u32 identifyLen = strnlen(identify, IDENTIFY_MAX_LEN + 1);
    if (identifyLen == (IDENTIFY_MAX_LEN + 1) || identifyLen == 0) {
        HCCL_ERROR("[Check][Identify]errNo[0x%016llx] identify name is invalid, len is %u",
            HCOM_ERROR_CODE(HCCL_E_PARA), identifyLen);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult HcomCheckDeviceId(const u32 device_id)
{
    if (device_id >= HCCL_AISERVER_DEVICE_NUM) {
        HCCL_ERROR("[Check][DeviceId]errNo[0x%016llx] device_id[%u] is invalid,should in (0~7)",
            HCOM_ERROR_CODE(HCCL_E_PARA), device_id);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult HcomCheckTag(const char *tag)
{
    CHK_PTR_NULL(tag);

    u32 tagLen = strnlen(tag, TAG_MAX_LEN + 1);
    if (tagLen == (TAG_MAX_LEN + 1) || tagLen == 0) {
        HCCL_ERROR("[Check][Tag]errNo[0x%016llx] tag is too long", HCOM_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult HcomCheckCount(const u64 count)
{
    if (count > SYS_MAX_COUNT) {
        HCCL_ERROR("[Check][Count]errNo[0x%016llx] count[%llu] is invalid(bigger than MAX count[%llu])",
            HCOM_ERROR_CODE(HCCL_E_PARA), count, SYS_MAX_COUNT);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult HcomCheckAlltoAllVExternalMem(const void *sendBuf, const void *sendCounts,
    const void *recvBuf, const void *recvCounts, u32 rankSize)
{
    u64 *sendCountsPtr = const_cast<u64 *>(static_cast<const u64 *>(sendCounts));
    u64 *recvCountsPtr = const_cast<u64 *>(static_cast<const u64 *>(recvCounts));
    std::string sendCountStr = "sendCounts:";
    std::string recvCountStr = "recvCounts:";
    bool hasSend = false;
    bool hasRecv = false;
    bool invalidSendCount = false;
    bool invalidRecvCount = false;
    for (u32 i = 0; i < rankSize; i++) {
        sendCountStr += ' ' + std::to_string(*(sendCountsPtr + i));
        recvCountStr += ' ' + std::to_string(*(recvCountsPtr + i));
        if (*(sendCountsPtr + i) != 0) {
            if (!invalidSendCount && *(sendCountsPtr + i) > SYS_MAX_COUNT) {
                invalidSendCount = true;
            }
            hasSend = true;
        }
        if (*(recvCountsPtr + i) != 0) {
            if (!invalidRecvCount && *(recvCountsPtr + i) > SYS_MAX_COUNT) {
                invalidRecvCount = true;
            }
            hasRecv = true;
        }
    }

    CHK_PRT_RET(invalidSendCount,
        HCCL_ERROR("HcomCheckAlltoAllVExternalMem sendCounts[%s] is invalid.(bigger than MAX count[%llu])",
        sendCountStr.c_str(), SYS_MAX_COUNT),
        HCCL_E_PARA);
    CHK_PRT_RET(invalidRecvCount,
        HCCL_ERROR("HcomCheckAlltoAllVExternalMem recvCounts[%s] is invalid.(bigger than MAX count[%llu])",
        recvCountStr.c_str(), SYS_MAX_COUNT),
        HCCL_E_PARA);

    HCCL_DEBUG("[HcomCheckAlltoAllVExternalMem] sendCounts: %s", sendCountStr.c_str());
    HCCL_DEBUG("[HcomCheckAlltoAllVExternalMem] recvCounts: %s", recvCountStr.c_str());

    if (hasSend) {
        RPT_INPUT_ERR(sendBuf == nullptr, "EI0003",\
            std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
            std::vector<std::string>({"HcomCheckAlltoAllVExternalMem", "sendBuf", "nullptr", "please check sendBuf"}));
        CHK_PTR_NULL(sendBuf);
    }
    if (hasRecv) {
        RPT_INPUT_ERR(recvBuf == nullptr, "EI0003",\
            std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
            std::vector<std::string>({"HcomCheckAlltoAllVExternalMem", "recvBuf", "nullptr", "please check recvBuf"}));
        CHK_PTR_NULL(recvBuf);
    }
    return HCCL_SUCCESS;
}

HcclResult HcomCheckAlltoAllVCExternalMem(const void *sendBuf, const void *sendCountMatrix,
    const void *recvBuf, u32 rankSize, u32 rank)
{
    u64 *sendCountMatrixPtr = const_cast<u64 *>(static_cast<const u64 *>(sendCountMatrix));
    bool hasSend = false;
    bool hasRecv = false;

    for (u32 i = 0; i < rankSize; i++) {
        for (u32 j = 0; j < rankSize; j++) {
            HCCL_DEBUG("[HcomCheckAlltoAllVCExternalMem] sendCounts[%u][%u]: %llu", i, j, *(sendCountMatrixPtr + i * \
                rankSize + j));
        }
        CHK_RET(HcomCheckCount(*(sendCountMatrixPtr + rank * rankSize + i)));
        if (hasSend == false && *(sendCountMatrixPtr + rank * rankSize + i) != 0) {
            hasSend = true;
        }
        if (hasRecv == false && *(sendCountMatrixPtr + i * rankSize + rank) != 0) {
            hasRecv = true;
        }
    }
    if (hasSend) {
        RPT_INPUT_ERR(sendBuf == nullptr, "EI0003",\
            std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
            std::vector<std::string>({"HcomCheckAlltoAllVCExternalMem", "sendBuf", "nullptr", "please check sendBuf"}));
        CHK_PTR_NULL(sendBuf);
    }
    if (hasRecv) {
        RPT_INPUT_ERR(recvBuf == nullptr, "EI0003",\
            std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
            std::vector<std::string>({"HcomCheckAlltoAllVCExternalMem", "recvBuf", "nullptr", "please check recvBuf"}));
        CHK_PTR_NULL(recvBuf);
    }
    return HCCL_SUCCESS;
}

void HcomGetHashFromSendCountMatrix(u64 &sendCountMatrixHash, const void *sendCountMatrix,
    u64 rankSize, const std::string &tag)
{
    std::string sendCountMatrixStr;
    std::hash<std::string> hashString;
    for (u32 i = 0; i < rankSize; i++) {
        for (u32 j = 0; j < rankSize; j++) {
            std::string curSendCountStr =
                std::to_string(*(static_cast<const u64 *>(sendCountMatrix) + i * rankSize + j));
            sendCountMatrixStr += curSendCountStr + '_';
        }
    }
    sendCountMatrixHash = hashString(sendCountMatrixStr.c_str());
    HCCL_DEBUG("[HcomGetHashFromSendCountMatrix] tag[%s], sendCountMatrixHash[%llu]",
        tag.c_str(), sendCountMatrixHash);
}

HcclResult HcomCheckDataType(const HcclDataType dataType)
{
    if ((dataType >= HCCL_DATA_TYPE_RESERVED) || (dataType < HCCL_DATA_TYPE_INT8)) {
        HCCL_ERROR("[Check][DataType]errNo[0x%016llx] data type[%s] not supported",
            HCOM_ERROR_CODE(HCCL_E_NOT_SUPPORT), GetDataTypeEnumStr(dataType).c_str());
        return HCCL_E_NOT_SUPPORT;
    }
    return HCCL_SUCCESS;
}

HcclResult HcomCheckGroupName(const char *group)
{
    if (group != nullptr) {
        u32 groupLen = strnlen(group, GROUP_NAME_MAX_LEN + 1);
        if (groupLen == (GROUP_NAME_MAX_LEN + 1) || groupLen == 0) {
            HCCL_ERROR("[Check][GroupName]errNo[0x%016llx] group name[%s] length[%lu] is invalid",
                HCOM_ERROR_CODE(HCCL_E_PARA), group, groupLen);
            return HCCL_E_PARA;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcomCheckReductionOp(const HcclReduceOp op)
{
    if ((op >= HCCL_REDUCE_RESERVED) || (op < HCCL_REDUCE_SUM)) {
        RPT_INPUT_ERR(true, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
            std::vector<std::string>({ "HcomCheckReductionOp", "op", GetReduceOpEnumStr(op), "op not supported" }));
        HCCL_ERROR("[Check][ReductionOp]errNo[0x%016llx] Op:[%s] not supported", HCOM_ERROR_CODE(HCCL_E_PARA),
            GetReduceOpEnumStr(op).c_str());
        return HCCL_E_NOT_SUPPORT;
    }
    return HCCL_SUCCESS;
}

HcclResult HcomCheckReduceDataType(const HcclDataType dataType, const HcclReduceOp op, DevType deviceType)
{
    if ((deviceType == DevType::DEV_TYPE_910B) || (deviceType == DevType::DEV_TYPE_910_73)) {
        if ((op == HCCL_REDUCE_PROD) &&
        ((dataType == HCCL_DATA_TYPE_INT16) || (dataType == HCCL_DATA_TYPE_BFP16))) {
            RPT_INPUT_ERR(true, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
                std::vector<std::string>({
                "HcomCheckReduceDataType",
                "dataType",
                GetDataTypeEnumStr(dataType),
                "please check dataType when optype is prod"
                }));
            HCCL_ERROR(
                "[Check][DataType]errNo[0x%016llx] device type[%d] does not support the data type[%s] and data "\
                "type[%s] for Op[%s]", HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), deviceType,
                GetDataTypeEnumStr(HcclDataType::HCCL_DATA_TYPE_BFP16).c_str(),
                GetDataTypeEnumStr(HcclDataType::HCCL_DATA_TYPE_INT16).c_str(),
                GetReduceOpEnumStr(op).c_str());
            return HCCL_E_NOT_SUPPORT;
        }
    } else if (deviceType == DevType::DEV_TYPE_910) {
        if (dataType == HCCL_DATA_TYPE_INT16) {
            RPT_INPUT_ERR(true, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
                std::vector<std::string>({
                "HcomCheckReduceDataType",
                "dataType",
                GetDataTypeEnumStr(dataType),
                "please check the data type when the device type is 910."
                }));
            HCCL_ERROR(
                "[Check][DataType]errNo[0x%016llx] device type[%d] does not support the data type[%s]",\
                HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), deviceType,
                GetDataTypeEnumStr(dataType).c_str());
            return HCCL_E_NOT_SUPPORT;
        }
    } else if (deviceType == DevType::DEV_TYPE_310P3) {
        if (dataType == HcclDataType::HCCL_DATA_TYPE_INT16 && op != HcclReduceOp::HCCL_REDUCE_SUM) {
            RPT_INPUT_ERR(true, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
                std::vector<std::string>({
                "HcomCheckReduceDataType",
                "op",
                GetReduceOpEnumStr(op),
                "please check operation type when the data type is int16."
            }));
            HCCL_ERROR(
                "[Check][DataType]errNo[0x%016llx] device type[%d] does not support the data type[%s] for Op[%s]",\
                HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), deviceType,
                GetDataTypeEnumStr(HcclDataType::HCCL_DATA_TYPE_INT16).c_str(),
                GetReduceOpEnumStr(op).c_str());
            return HCCL_E_NOT_SUPPORT;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcomCheckUserRank(const u32 totalRanks, const u32 userRank)
{
    if (userRank >= totalRanks) {
        HCCL_ERROR("[Check][UserRank]errNo[0x%016llx] userRank:[%u] is out of range[0 ~ %u]",
            HCOM_ERROR_CODE(HCCL_E_PARA), userRank, totalRanks - 1);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult HcomCheckOpParam(const char *tag, const u64 count, const HcclDataType dataType, const char *group,
    const void *stream)
{
    HcclResult ret = HcomCheckGroupName(group);
    RPT_INPUT_ERR(ret != HCCL_SUCCESS, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({tag, "group", group, "please check group"}));
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Check][OpParam]errNo[0x%016llx] group name is invalid",
        HCOM_ERROR_CODE(ret)), ret);

    CHK_RET(HcomCheckOpParam(tag, count, dataType, stream));

    return HCCL_SUCCESS;
}

HcclResult HcomCheckOpParam(const char *tag, const u64 count, const HcclDataType dataType, const void *stream)
{
    CHK_RET(HcomCheckOpParam(tag, count, dataType));

    RPT_INPUT_ERR(stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({tag, "stream", "nullptr", "please check stream"}));
    CHK_PTR_NULL(stream);

    return HCCL_SUCCESS;
}

HcclResult HcomCheckOpParam(const char *tag, const u64 count, const HcclDataType dataType)
{
    HcclResult ret = HcomCheckTag(tag);
    RPT_INPUT_ERR(ret != HCCL_SUCCESS, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcomCheckTag", "tag", tag, "please check tag"}));
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Check][OpParam]errNo[0x%016llx] tag is invalid",
        HCOM_ERROR_CODE(ret)), ret);

    ret = HcomCheckCount(count);
    RPT_INPUT_ERR(ret != HCCL_SUCCESS, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({tag, "count", std::to_string(count), "please check count"}));
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Check][OpParam]errNo[0x%016llx] count is out of range",
        HCOM_ERROR_CODE(ret)), ret);

    ret = HcomCheckDataType(dataType);
    RPT_INPUT_ERR(ret != HCCL_SUCCESS, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({tag, "dataType", GetDataTypeEnumStr(dataType), "please check dataType"}));
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Check][OpParam]errNo[0x%016llx] dataType is invalid",
        HCOM_ERROR_CODE(ret)), ret);

    return HCCL_SUCCESS;
}

HcclResult HcclParseRanktable(const std::string &rankTableM, const std::string &identify, hccl::HcclCommParams &params,
    hccl::RankTable_t &rankTable)
{
    // 记录版本信息
    std::string curVersion = GetExternalInputCannVersion();
    CHK_RET(RankConsistent::GetInstance().RecordVerInfo(curVersion));

    // ranktableCRC计算
    if (rankTableM.c_str() == nullptr) {
        HCCL_INFO("rank table is null, rankTableCrc is 0.");
    } else {
        HcclResult ret = HcomCalcCRC(params, rankTableM.c_str());
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Init][OtherInfo]errNo[0x%016llx] calc ranktable crc error",
            HCCL_ERROR_CODE(HCCL_E_INTERNAL)), HCCL_E_INTERNAL);
    }

    // 解析rankTable_json对象，将解析的信息保存在rankinfo中
    HcclResult ret = CfgGetClusterInfo(rankTableM, identify, params, rankTable);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Init][HcclComm]errNo[0x%016llx] cfg get clusterInfo jsonString ",
        HCCL_ERROR_CODE(ret)), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}
