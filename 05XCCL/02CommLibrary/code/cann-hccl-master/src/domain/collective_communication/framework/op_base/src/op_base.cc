/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_base.h"
#include <algorithm>
#include <future>
#include <map>
#include <string>
#include <hccl/hccl_types.h>

#include "hccl/base.h"
#include "workflow_pub.h"
#include "param_check_pub.h"
#include "rank_consistent.h"
#include "externalinput_pub.h"
#include "../common/src/topo/topoinfo_detect.h"
#include "sal_pub.h"
#include "profiling_manager_pub.h"
#include "adapter_prof_pub.h"
#include "adapter_qos_pub.h"
#include "adapter_rts_common.h"
#include "device_capacity.h"
#include "mem_host_pub.h"
#include "hcom_common.h"
#include "comm_config_pub.h"

#define DOUBLE_SIZE 2
#define RANKSIZE_TWO 2

using namespace std;
using namespace hccl;
const std::string HCCL_ALLTOALL = "ALLTOALL";
const std::string HCCL_ALLTOALLV = "ALLTOALLV";
const std::string HCCL_ALLTOALLVC = "ALLTOALLVC";

map<std::string, shared_ptr<TopoInfoDetect>> g_topoDetectServerPtrMap;

HcclResult CallMsprofReportHostApi(hccl::hcclComm* hcclComm, HcclCMDType cmdType, uint64_t beginTime, u64 count,
    HcclDataType dataType, std::string tag)
{
    if (GetExternalInputHcclIfProf()) {
        AlgType algType = AlgType::ALG_DEFAULT;
        CHK_RET(hcclComm->GetAlgType(algType, cmdType));
        uint64_t groupName = hrtMsprofGetHashId(hcclComm->GetIdentifier().c_str(), hcclComm->GetIdentifier().length());
        CHK_RET_AND_PRINT_IDE(ProfilingManagerPub::CallMsprofReportHostApi(cmdType, beginTime, count, dataType, algType,
            groupName), tag.c_str());
    }
    return HCCL_SUCCESS;
}

thread_local s32 g_hcclDeviceId = INVALID_INT;
HcclOpInfoCtx g_opHcomInfos[MAX_MODULE_DEVICE_NUM + 1];

HcclResult HcclGetDeviceId(void)
{
    if (g_hcclDeviceId == INVALID_INT) {
        CHK_PRT_RET(hrtGetDevice(&g_hcclDeviceId) != HCCL_SUCCESS,
            HCCL_WARNING("[HcclGetDeviceId] get fail deviceLogicId[%d]", g_hcclDeviceId), HCCL_E_INTERNAL);
    }
    CHK_PRT_RET(static_cast<u32>(g_hcclDeviceId) >= MAX_MODULE_DEVICE_NUM,
        HCCL_WARNING("[HcclGetDeviceId]deviceLogicId[%d] is bigger than HCCL_AISERVER_DEVICE_NUM_MAX:[%u]",
        g_hcclDeviceId, MAX_MODULE_DEVICE_NUM), HCCL_E_INTERNAL);
    HCCL_INFO("[HcclGetDeviceId] deviceLogicId[%d] ", g_hcclDeviceId);
    return HCCL_SUCCESS;
}

s32 HcclGetThreadDeviceId()
{
    CHK_PRT_RET(HcclGetDeviceId() != HCCL_SUCCESS, HCCL_WARNING("[HcclGetThreadDeviceId] get fail deviceLogicId[%d]",
        g_hcclDeviceId), INVALID_INT);
    return g_hcclDeviceId;
}

HcclOpInfoCtx &GetHcclOpInfoCtx(void)
{
    if (HcclGetDeviceId() == HCCL_SUCCESS) {
        if (!g_opHcomInfos[g_hcclDeviceId].isUsed) {
            HCCL_INFO("[GetHcclOpInfoCtx] Set device, use g_hcclDeviceId[%d] ", g_hcclDeviceId);
            if (g_opHcomInfos[MAX_MODULE_DEVICE_NUM].isUsed) {
                g_hcclDeviceId = MAX_MODULE_DEVICE_NUM;
                HCCL_INFO("[GetHcclOpInfoCtx] Used cover bottom g_hcclDeviceId[%d]", g_hcclDeviceId);
                return g_opHcomInfos[g_hcclDeviceId];
            }
        }
        g_opHcomInfos[g_hcclDeviceId].isUsed = true;
        return g_opHcomInfos[g_hcclDeviceId];
    }

    for (u32 i = 0; i < MAX_MODULE_DEVICE_NUM; i++) {
        if (g_opHcomInfos[i].isUsed) {
            g_hcclDeviceId = i;
            HCCL_INFO("[GetHcclOpInfoCtx] Not set device, Used g_hcclDeviceId[%u] ", i);
            return g_opHcomInfos[g_hcclDeviceId];
        }
    }
    g_hcclDeviceId = MAX_MODULE_DEVICE_NUM;
    g_opHcomInfos[MAX_MODULE_DEVICE_NUM].isUsed = true;
    HCCL_INFO("[GetHcclOpInfoCtx] Used cover bottom g_hcclDeviceId[%d]", g_hcclDeviceId);
    return g_opHcomInfos[MAX_MODULE_DEVICE_NUM];
}

HcclResult GetDeviceComm(uint32_t ndev, const HcclRootInfo &rootHandle, const s32 rank, const s32 logicDeviceId,
    HcclComm &comm)
{
    CHK_PRT_RET(hrtSetDevice(logicDeviceId) != HCCL_SUCCESS,
        HCCL_ERROR("[GetDeviceComm] set fail logicDeviceId[%d]", logicDeviceId), HCCL_E_INTERNAL);
    HcclResult ret = HcclCommInitRootInfo(ndev, &rootHandle, rank, &comm);
    if (ret != HCCL_SUCCESS || comm == nullptr) {
        comm = nullptr;
        HCCL_ERROR("[GetDeviceComm] rank[%d] Get device comm failed!", rank);
        CHK_PRT_RET(hrtResetDevice(logicDeviceId) != HCCL_SUCCESS,
            HCCL_ERROR("[GetDeviceComm] reset fail logicDeviceId[%d]", logicDeviceId), HCCL_E_INTERNAL);
        return ret;
    }
    hcclComm *pComm = static_cast<hcclComm *>(comm);
    pComm->ResetDeviceEnable();
    return HCCL_SUCCESS;
}

HcclResult HcclGetCommAll(uint32_t ndev, int32_t *devices, HcclComm *comms)
{
    // 入参校验
    CHK_PRT_RET(ndev <= 0, HCCL_ERROR("[HcclGetCommAll] ndev is unvalid ndev[%u]", ndev), HCCL_E_PARA);
    CHK_PTR_NULL(comms);
    CHK_PTR_NULL(devices);

    CHK_PRT_RET(hrtSetDevice(devices[0]) != HCCL_SUCCESS,
        HCCL_ERROR("[HcclGetCommAll] set fail devices[0][%d]", devices[0]), HCCL_E_INTERNAL);

    HcclRootInfo rootHandle;
    CHK_RET(HcclGetRootInfo(&rootHandle));

    // 获取通信域之前, 先把所有通信域设置为空
    for (uint32_t i = 0; i < ndev; i++) {
        comms[i] = nullptr;
    }
    std::vector<std::unique_ptr<std::thread>> threads(ndev);
    for (uint32_t rankId = 0; rankId < ndev; rankId++) {
        threads[rankId].reset(new (std::nothrow) std::thread(&GetDeviceComm, ndev, std::ref(rootHandle), rankId,
            devices[rankId], std::ref(comms[rankId])));
        CHK_PRT_RET(!threads[rankId], HCCL_ERROR("[HcclGetCommAll]threads[%u] reset failed ", rankId), HCCL_E_INTERNAL);
    }
    for (uint32_t i = 0; i < ndev; i++) {
        threads[i]->join();
    }

    // 如果任何一个通信域初始化失败，将所有已经成功创建的通信域销毁
    bool isFailed = false;
    for (uint32_t i = 0; i < ndev; ++i) {
        if (comms[i] == nullptr) {
            HCCL_ERROR("[HcclGetCommAll] rank[%u] get comm failed!", i);
            isFailed = true;
            break;
        }
    }
    if (isFailed) {
        for (uint32_t i = 0; i < ndev; ++i) {
            if (comms[i] != nullptr) {
                (void)HcclCommDestroy(comms[i]);
            }
        }
        return HCCL_E_INTERNAL;
    }

    CHK_PRT_RET(hrtResetDevice(devices[0]) != HCCL_SUCCESS,
        HCCL_ERROR("[HcclGetCommAll] reset fail devices[0][%d]", devices[0]), HCCL_E_INTERNAL);

    return HCCL_SUCCESS;
}

HcclResult HcclCommInitAll(uint32_t ndev, int32_t *devices, HcclComm *comms)
{
    HcclUs startut = TIME_NOW();
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));

    // 入参校验
    CHK_PRT_RET(ndev <= 0, HCCL_ERROR("[HcclCommInitAll] ndev is unvalid ndev:[%u]", ndev), HCCL_E_PARA);
    CHK_PTR_NULL(comms);
    CHK_PTR_NULL(devices);

    // 判断设备List中是否有重复id,报错退出
    set<int32_t> devSet(devices, devices + ndev);
    uint32_t devSetSize = devSet.size();
    CHK_PRT_RET((devSetSize != ndev),
        HCCL_ERROR("[HcclCommInitAll] Duplicate device id exist in the device list. devSetSize:[%u], ndev:[%u]",
        devSetSize, ndev),
        HCCL_E_PARA);

    std::future<HcclResult> threadResult;
    std::unique_ptr<std::thread> getCommThread;
    getCommThread.reset(new (std::nothrow) std::thread(
        [=, &threadResult]() { threadResult = std::async(std::launch::async, HcclGetCommAll, ndev, devices, comms); }));
    CHK_PRT_RET(!getCommThread, HCCL_ERROR("[HcclCommInitAll]thread reset failed "), HCCL_E_INTERNAL);
    getCommThread->join();

    HcclResult ret = threadResult.get();
    if (ret != HCCL_SUCCESS) {
        for (uint32_t i = 0; i < ndev; ++i) {
            if (comms[i] != nullptr) {
                (void)HcclCommDestroy(comms[i]);
                comms[i] = nullptr;
            }
        }
        HCCL_ERROR("HcclCommInitAll failed! threadResult[%d]", ret);
        return ret;
    }
    HCCL_RUN_INFO("HcclCommInitAll success, take time [%lld]us, deviceLogicId[%d]", DURATION_US(TIME_NOW() - startut),
        deviceLogicId);
    return HCCL_SUCCESS;
}

HcclResult HcclCommInitClusterInfo(const char *clusterInfo, uint32_t rank, HcclComm *comm)
{
    HcclUs startut = TIME_NOW();
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));

    // 入参合法性校验
    CHK_PTR_NULL(clusterInfo);
    CHK_PTR_NULL(comm);

    std::string rankTableM;
    std::string realFilePath;
    HcclResult ret = HcomLoadRanktableFile(clusterInfo, rankTableM, realFilePath);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Init][CommClusterInfo]errNo[0x%016llx] clusterInfo[%s] rank[%u] "
        "load rankTable error.", HCCL_ERROR_CODE(HCCL_E_UNAVAIL), clusterInfo, rank), HCCL_E_INTERNAL);

    u32 rankTableSize = 0;
    ret = HcomCheckRankTable(rankTableM.c_str(), rankTableSize);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Init][CommClusterInfo]check rankTable string error, rankTableSize [%u]",
            rankTableSize), HCCL_E_PARA);

    ret = InitExternalInput();
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Init][CommClusterInfo]errNo[0x%016llx] init external input error",
        HCCL_ERROR_CODE(ret)), HCCL_E_PARA);

    HcclOpInfoCtx& opBaseHcom = GetHcclOpInfoCtx();
    /* 防止重复调用初始化 */
    CHK_PRT_RET((opBaseHcom.pComm != nullptr), HCCL_ERROR("[Init][CommClusterInfo]errNo[0x%016llx] rank[%u] "\
        "op_base hccl multiple initialization", HCCL_ERROR_CODE(HCCL_E_UNAVAIL), rank), HCCL_E_UNAVAIL);
    opBaseHcom.pComm.reset(
        new (std::nothrow) hccl::hcclComm(
            GetExternalInputCCLBuffSize(), GetExternalInputCCLBuffSize(), HCCL_WORLD_GROUP));
    CHK_PTR_NULL(opBaseHcom.pComm);

    /* 接口交互信息日志 */
    HCCL_RUN_INFO("Entry-HcclCommInitClusterInfo:clusterInfo[%s], rank[%u], deviceLogicId[%d]",
        realFilePath.c_str(), rank, deviceLogicId);

    /* --------------初始化------------------------- */
    bool errorFlag = false;
    do {
        RankConsistent::GetInstance().SetCheckCannVersionSwitch(true); // 打开CANN软件版本校验开关
        ret = InitOtherInfo(opBaseHcom.params, rankTableM.c_str());
        CHK_PRT_BREAK(ret != HCCL_SUCCESS, HCCL_ERROR("[Init][CommClusterInfo]errNo[0x%016llx] init other Info",
            HCCL_ERROR_CODE(ret)), errorFlag = true);

        HCCL_INFO("rootInfo[%s]", opBaseHcom.params.id.internal);

        ret = InitWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
        CHK_PRT_BREAK(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Init][CommClusterInfo]errNo[0x%016llx] init work flow mode error",
                HCCL_ERROR_CODE(ret)), errorFlag = true);

        ret = CfgGetClusterInfo(rankTableM, to_string(rank), opBaseHcom.params, opBaseHcom.rankTable);
        CHK_PRT_BREAK(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Init][CommClusterInfo]errNo[0x%016llx] cfg get clusterInfo[%s]"\
                "info error:rank[%u]", HCCL_ERROR_CODE(ret), realFilePath.c_str(), rank), errorFlag = true);

        HCCL_INFO("rootInfo[%s]", opBaseHcom.params.id.internal);

        ret = opBaseHcom.pComm->init(opBaseHcom.params, opBaseHcom.rankTable);
        CHK_PRT_BREAK(ret != HCCL_SUCCESS, HCCL_ERROR("[Init][CommClusterInfo]errNo[0x%016llx] hcclComm init error",
            HCCL_ERROR_CODE(ret)), errorFlag = true);

        ret = ShowRanktableConfigInfo(opBaseHcom.cloudFlag, opBaseHcom.params,
            opBaseHcom.rankTable);
        CHK_PRT_BREAK(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Init][CommClusterInfo]errNo[0x%016llx] put ranktable info error",
                HCCL_ERROR_CODE(ret)), errorFlag = true);

        // 初始化完成的comm指针赋给出参
        *comm = opBaseHcom.pComm.get();
        std::unique_lock<std::mutex> lock(opBaseHcom.opGroupMapMutex);
        opBaseHcom.opGroup2CommMap[HCCL_WORLD_GROUP] = opBaseHcom.pComm;
    } while (0);

    if (errorFlag) {
        HCCL_ERROR("[Init][CommClusterInfo]HcclCommInitClusterInfo failed, rankNum[%u], rank[%u], server[%s],"\
            "device[%d], return[0x%016llx]", opBaseHcom.rankTable.rankNum, rank,
            opBaseHcom.params.serverId.c_str(), opBaseHcom.params.logicDevId, HCCL_ERROR_CODE(ret));
        (void)HcclCommDestroy(opBaseHcom.pComm.get());
        return ret;
    }

    /* 关键状态记录 */
    HCCL_RUN_INFO("[HCCL_TRACE]HcclCommInitClusterInfo success,take time [%lld]us, clusterInfo[%s], rankNum[%u],"\
        "rank[%u],server[%s], device[%d]", DURATION_US(TIME_NOW() - startut), realFilePath.c_str(),
        opBaseHcom.rankTable.rankNum, rank, opBaseHcom.params.serverId.c_str(),
        opBaseHcom.params.logicDevId);
    return HCCL_SUCCESS;
}

HcclResult HcclGetRootInfo(HcclRootInfo *rootInfo)
{
    HcclUs startut = TIME_NOW();
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));

    // input check
    CHK_PTR_NULL(rootInfo);
    HCCL_RUN_INFO("Entry-HcclGetRootInfo:rootInfo[%p], deviceLogicId[%d]", rootInfo, deviceLogicId);
    // get commId from env
    CHK_RET(InitExternalInput());

    HcclRootHandle rootHandle;
    std::shared_ptr<TopoInfoDetect> topoDetectServer = std::make_shared<TopoInfoDetect>();
    CHK_SMART_PTR_NULL(topoDetectServer);
    CHK_RET(topoDetectServer->SetupServer(rootHandle));

    if (sizeof(HcclRootHandle) > HCCL_ROOT_INFO_BYTES) {
        HCCL_ERROR("[Get][RootInfo]hccl root info overflow. max length: %u, actual:%zu, identifier[%s]",
            HCCL_ROOT_INFO_BYTES, sizeof(HcclRootHandle), rootHandle.identifier);
        return HCCL_E_INTERNAL;
    } else {
        s32 sRet = memcpy_s(rootInfo->internal, HCCL_ROOT_INFO_BYTES, &rootHandle, sizeof(HcclRootHandle));
        CHK_PRT_RET(sRet != EOK, HCCL_ERROR("[Get][RootInfo]memcpy root info fail. errorno[%d] "\
            "params:destMaxSize[%u], count[%u]", sRet, HCCL_ROOT_INFO_BYTES,
            sizeof(HcclRootHandle)), HCCL_E_MEMORY);
    }

    HcclOpInfoCtx& opBaseInfo = GetHcclOpInfoCtx();
    opBaseInfo.hcclCommTopoInfoDetectServer.insert({rootHandle.identifier, topoDetectServer});
    /* 首节点诊断信息记录 */
    HCCL_RUN_INFO("[HCCL_TRACE]HcclGetRootInfo success, take time [%lld]us, identifier[%s]",
        DURATION_US(TIME_NOW() - startut), rootHandle.identifier);
    return HCCL_SUCCESS;
}

HcclResult GetSelfClusterInfo(const HcclBasicRankInfo &rankInfo, HcclCommParams &params)
{
    params.deviceType = rankInfo.deviceType;
    params.rank = rankInfo.rank;
    params.userRank = rankInfo.rank;
    params.logicDevId = rankInfo.deviceLogicID;
    params.totalRanks = rankInfo.rankSize;
    params.serverId = rankInfo.hostIP.GetReadableAddress();

    return HCCL_SUCCESS;
}

HcclResult HcclGetCommName(HcclComm commHandle, char *commName)
{
    CHK_PTR_NULL(commHandle);
    CHK_PTR_NULL(commName);
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(commHandle);
    s32 ret = strncpy_s(commName, ROOTINFO_INDENTIFIER_MAX_LENGTH, hcclComm->GetIdentifier().c_str(),
        hcclComm->GetIdentifier().size());
    CHK_PRT_RET(ret != EOK, HCCL_ERROR("HcclGetCommName str copy fail. return[%d]", ret), HCCL_E_INTERNAL);
    HCCL_INFO("HcclGetCommName input handle=%p commName=%s", commHandle, commName);
    return HCCL_SUCCESS;
}

HcclResult HcclGetCommHandle(const char *commName, std::shared_ptr<hccl::hcclComm> &comm)
{
    CHK_PTR_NULL(commName);
    std::string group(commName);

    HcclOpInfoCtx& opBaseHcom = GetHcclOpInfoCtx();
    std::unique_lock<std::mutex> lock(opBaseHcom.opGroupMapMutex);
    auto iter = opBaseHcom.opGroup2CommMap.find(group);
    if (iter == opBaseHcom.opGroup2CommMap.end()) {
        HCCL_WARNING("please check the group name is correct, group=%s", commName);
        return HCCL_E_PARA;
    } else {
        comm = iter->second;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclGetCommConnections(const HcclRootHandle &rootHandle, HcclCommConnections &commConnections)
{
    HcclOpInfoCtx& opBaseInfo = GetHcclOpInfoCtx();
    auto iterServer = opBaseInfo.hcclCommTopoInfoDetectServer.find(rootHandle.identifier);
    if (iterServer == opBaseInfo.hcclCommTopoInfoDetectServer.end()) {
        commConnections.isRoot = false;
    } else {
        commConnections.isRoot = true;
        CHK_RET(iterServer->second->GetServerConnections(commConnections.serverConnections));
    }

    auto iterAgent = opBaseInfo.hcclCommTopoInfoDetectAgent.find(rootHandle.identifier);
    if (iterAgent == opBaseInfo.hcclCommTopoInfoDetectAgent.end()) {
        HCCL_ERROR("hccl get agent connections failed, rootHandle.identifier=%s", rootHandle.identifier);
        return HCCL_E_PARA;
    } else {
        CHK_RET(iterAgent->second->GetAgentConnection(commConnections.agentConnection));
    }
    return HCCL_SUCCESS;
}

void HcclCloseCommConnections(const std::string &identifier)
{
    HcclOpInfoCtx& opBaseInfo = GetHcclOpInfoCtx();
    opBaseInfo.hcclCommTopoInfoDetectServer.erase(identifier);
    opBaseInfo.hcclCommTopoInfoDetectAgent.erase(identifier);
    return;
}

HcclResult InitCommRootInfo(const u32 nRanks, const u32 rank, const HcclRootHandle &rootHandle,
    const CommConfig &commConfig, HcclComm *comm)
{
    HcclResult ret = HCCL_SUCCESS;
    bool errorFlag = false;
    std::shared_ptr<hccl::hcclComm> pComm;
    HcclOpInfoCtx& opBaseHcom = GetHcclOpInfoCtx();
    hccl::HcclCommParams params;
    do {
        RankConsistent::GetInstance().SetCheckCannVersionSwitch(true); // 打开CANN软件版本校验开关
        pComm.reset(new hccl::hcclComm(commConfig.GetConfigBufferSize(), commConfig.GetConfigBufferSize(),
            rootHandle.identifier));
        CHK_SMART_PTR_NULL(pComm);

        std::shared_ptr<TopoInfoDetect> topoDetectAgent = std::make_shared<TopoInfoDetect>();
        CHK_SMART_PTR_NULL(topoDetectAgent);
        ret = topoDetectAgent->SetupAgent(nRanks, rank, rootHandle);
        CHK_PRT_BREAK(ret != HCCL_SUCCESS, HCCL_ERROR("[Init][CommRootInfo]errNo[0x%016llx] setup topo detect error",
            HCCL_ERROR_CODE(ret)), errorFlag = true);

        RankTable_t rankTable;
        ret = topoDetectAgent->GetCluterInfo(rankTable);
        CHK_PRT_BREAK(ret != HCCL_SUCCESS, HCCL_ERROR("[Init][CommRootInfo]errNo[0x%016llx] GetCluterInfo error",
            HCCL_ERROR_CODE(ret)), errorFlag = true);

        /* 初始化hccl comm */
        HcclBasicRankInfo localRankInfo;
        ret = topoDetectAgent->GetLocalRankInfo(localRankInfo);
        CHK_PRT_BREAK(ret != HCCL_SUCCESS, HCCL_ERROR("[Init][CommRootInfo]errNo[0x%016llx] GetLocalRankInfo error.",
            HCCL_ERROR_CODE(ret)), errorFlag = true);

        ret = GetSelfClusterInfo(localRankInfo, params);
        CHK_PRT_BREAK(ret != HCCL_SUCCESS,
            HCCL_ERROR("[InitCommRootInfo]errNo[0x%016llx] GetRankInfo error.", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        ret = topoDetectAgent->WaitComplete(rootHandle);
        CHK_PRT_BREAK(ret != HCCL_SUCCESS,
            HCCL_ERROR("[InitCommRootInfo]errNo[0x%016llx] wait complete topo detect error", HCCL_ERROR_CODE(ret)),
            errorFlag = true);
        CHK_RET(DisplayRanktableInfo(rankTable));

        bool retryEnable = GetExternalInputIntraServerRetryEnable() || GetExternalInputInterServerRetryEnable() ||
            GetExternalInputInterSuperPodRetryEnable();
        if (retryEnable) {
            opBaseHcom.hcclCommTopoInfoDetectAgent.insert({ rootHandle.identifier, topoDetectAgent });
            ret = HcclGetCommConnections(rootHandle, params.commConnections);
            CHK_PRT_BREAK(ret != HCCL_SUCCESS, HCCL_ERROR("[Init][RootInfo]HcclGetCommConnections failed."),
                errorFlag = true);
        } else {
            ret = topoDetectAgent->Teardown();
            CHK_PRT_BREAK(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Init][RootInfo]errNo[0x%016llx] Teardown topo detect error", HCCL_ERROR_CODE(ret)),
                errorFlag = true);
        }

        ret = InitWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
        CHK_PRT_BREAK(ret != HCCL_SUCCESS,
            HCCL_ERROR("[InitCommRootInfo]errNo[0x%016llx] init work flow mode error", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        ret = InitOtherInfo(params, nullptr);
        CHK_PRT_BREAK(ret != HCCL_SUCCESS,
            HCCL_ERROR("[InitCommRootInfo]errNo[0x%016llx] init other Info", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        HCCL_INFO("rootInfo[%s], params.logiceDevice[%d]", params.id.internal, params.logicDevId);
        ret = pComm->init(params, rankTable);
        CHK_PRT_BREAK(ret != HCCL_SUCCESS,
            HCCL_ERROR("[InitCommRootInfo]errNo[0x%016llx] hcclComm init error", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        /* 设置确定性计算配置 */
        ret = pComm->SetDeterministicConfig(commConfig.GetConfigDeterministic());
        CHK_PRT_BREAK(ret != HCCL_SUCCESS,
            HCCL_ERROR("[InitCommRootInfo]errNo[0x%016llx] set deterministic error", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        // 初始化完成的comm指针赋给出参
        *comm = pComm.get();
        std::unique_lock<std::mutex> lock(opBaseHcom.opGroupMapMutex);
        opBaseHcom.opGroup2CommMap[pComm->GetIdentifier()] = pComm;
        lock.unlock();

        ret = HcomSetGroupTopoInfo(pComm->GetIdentifier().c_str(), nRanks);
        CHK_PRT_BREAK(ret != HCCL_SUCCESS,
            HCCL_ERROR("[InitCommRootInfo]errNo[0x%016llx] setGroupTopoInfo error", HCCL_ERROR_CODE(ret)),
            errorFlag = true);
    } while (0);

    if (errorFlag) {
        HCCL_ERROR("[InitCommRootInfo]Init failed, return[0x%016llx], rankNum[%u], rank[%u], "\
            "rootInfo identifier[%s], server[%s], logicDevId[%d]", HCCL_ERROR_CODE(ret), nRanks, rank,
            rootHandle.identifier, GetLocalServerId(params.serverId).c_str(), params.logicDevId);
        (void)HcclCommDestroy(pComm.get());
        return ret;
    }

    HCCL_INFO("[InitCommRootInfo]Init success, rankNum[%u], rank[%u], rootInfo identifier[%s], server[%s], "
              "logicDevId[%d]",
        nRanks, rank, rootHandle.identifier, params.serverId.c_str(), params.logicDevId);

    return HCCL_SUCCESS;
}

HcclResult HcclCommInitRootInfoInner(uint32_t nRanks, const HcclRootInfo *rootInfo, uint32_t rank,
                                     HcclComm *comm, string &identifier)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclUs startut = TIME_NOW();

    CHK_SMART_PTR_NULL(rootInfo);
    HcclRootHandle rootHandle;
    s32 sRet = memcpy_s(&rootHandle, sizeof(HcclRootHandle), rootInfo->internal, sizeof(HcclRootHandle));
    CHK_PRT_RET(sRet != EOK, HCCL_ERROR("[Init][RootInfoInner]memcpy root info fail. errorno[%d] "\
        "params:destMaxSize[%u], count[%u]", sRet, sizeof(HcclRootHandle),
        sizeof(HcclRootHandle)), HCCL_E_MEMORY);
    rootHandle.identifier[ROOTINFO_INDENTIFIER_MAX_LENGTH - 1] = '\0';
    identifier = rootHandle.identifier;

    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));

    CHK_PRT_RET((nRanks == 0), HCCL_ERROR("[Init][CommRootInfoInner]errNo[0x%016llx] nRanks[%u] should "\
        "be greater than 0.", HCCL_ERROR_CODE(HCCL_E_PARA), nRanks), HCCL_E_PARA);

    CHK_PRT_RET((rank >= nRanks), HCCL_ERROR("[Init][CommRootInfoInner]errNo[0x%016llx] rank[%u] should "\
        "be less than nRanks[%u].", HCCL_ERROR_CODE(HCCL_E_PARA), rank, nRanks), HCCL_E_PARA);
    CHK_SMART_PTR_NULL(comm);

    ret = InitExternalInput();
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Init][CommRootInfoInner]errNo[0x%016llx] init "\
        "external input error", HCCL_ERROR_CODE(ret)), HCCL_E_PARA);
    /* 接口交互信息日志 */
    HCCL_RUN_INFO("Entry-HcclCommInitRootInfoInner:ranks[%u], rank[%u], rootinfo: host ip[%s] port[%u] "\
        "nicDeploy[%d] identifier[%s], deviceLogicId[%d]", nRanks, rank, rootHandle.ip, rootHandle.port,
        rootHandle.nicDeploy, rootHandle.identifier, deviceLogicId);

    CommConfig commConfig;

    /* --------------初始化------------------------- */
    ret = InitCommRootInfo(nRanks, rank, rootHandle, commConfig, comm);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Init][CommRootInfoConfig]errNo[0x%016llx]HcclCommInitRootInfo failed.",
        HCCL_ERROR_CODE(ret)),
        ret);

    /* 关键状态记录 */
    HCCL_RUN_INFO("[HCCL_TRACE]HcclCommInitRootInfoInner success, take time [%lld]us, rankNum[%u], rank[%u]",
        DURATION_US(TIME_NOW() - startut), nRanks, rank);
    return HCCL_SUCCESS;
}

HcclResult HcclCommInitRootInfo(uint32_t nRanks, const HcclRootInfo *rootInfo, uint32_t rank, HcclComm *comm)
{
    HcclResult ret = HCCL_SUCCESS;
    string identifier;
    ret = HcclCommInitRootInfoInner(nRanks, rootInfo, rank, comm, identifier);
    if (g_topoDetectServerPtrMap.find(identifier) != g_topoDetectServerPtrMap.end()) {
        g_topoDetectServerPtrMap[identifier] = nullptr;
    }
    return ret;
}

HcclResult HcclCommInitRootInfoConfigInner(uint32_t nRanks, const HcclRootInfo *rootInfo, uint32_t rank,
    const HcclCommConfig *config, HcclComm *comm, string &identifier)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclUs startut = TIME_NOW();

    CHK_SMART_PTR_NULL(rootInfo);

    HcclRootHandle rootHandle;
    s32 sRet = memcpy_s(&rootHandle, sizeof(HcclRootHandle), rootInfo->internal, sizeof(HcclRootHandle));
    CHK_PRT_RET(sRet != EOK, HCCL_ERROR("[Init][RootInfo]memcpy root info fail. errorno[%d] "\
        "params:destMaxSize[%u], count[%u]", sRet, sizeof(HcclRootHandle),
        sizeof(HcclRootHandle)), HCCL_E_MEMORY);
    rootHandle.identifier[ROOTINFO_INDENTIFIER_MAX_LENGTH - 1] = '\0';
    identifier = rootHandle.identifier;

    // 检查配置参数是否为空
    RPT_INPUT_ERR(config == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclCommInitRootInfoConfigInner", "config", "nullptr", "please check comm"}));
    CHK_SMART_PTR_NULL(config);

    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));

    CHK_PRT_RET((nRanks == 0),
        HCCL_ERROR("[Init][CommRootInfoConfigInner]errNo[0x%016llx] nRanks[%u] should be greater than 0.",
        HCCL_ERROR_CODE(HCCL_E_PARA), nRanks),
        HCCL_E_PARA);

    CHK_PRT_RET((rank >= nRanks),
        HCCL_ERROR("[Init][CommRootInfoConfigInner]errNo[0x%016llx] rank[%u] should be less than nRanks[%u].",
        HCCL_ERROR_CODE(HCCL_E_PARA), rank, nRanks),
        HCCL_E_PARA);
    CHK_SMART_PTR_NULL(comm);

    ret = InitExternalInput();
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Init][CommRootInfoConfigInner]errNo[0x%016llx] init "\
        "external input error", HCCL_ERROR_CODE(ret)), HCCL_E_PARA);

    /* 读取用户配置 */
    CommConfig commConfig;
    ret = commConfig.Load(config, rootHandle.identifier);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Init][CommRootInfoConfigInner]errNo[0x%016llx] load comm config failed.",
        HCCL_ERROR_CODE(ret)), HCCL_E_PARA);

    /* 接口交互信息日志 */
    HCCL_RUN_INFO("Entry-HcclCommInitRootInfoConfigInner:ranks[%u], rank[%u], rootinfo: host ip[%s] "\
        "port[%u] nicDeploy[%d] identifier[%s], deviceLogicId[%d]", nRanks, rank, rootHandle.ip,
        rootHandle.port, rootHandle.nicDeploy, rootHandle.identifier, deviceLogicId);

    /* --------------初始化------------------------- */
    ret = InitCommRootInfo(nRanks, rank, rootHandle, commConfig, comm);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Init][CommRootInfoConfigInner]errNo[0x%016llx]HcclCommInitRootInfoConfigInner failed.",
        HCCL_ERROR_CODE(ret)),
        ret);

    HCCL_RUN_INFO("[HCCL_TRACE]HcclCommInitRootInfoConfigInner success, take time [%lld]us, "\
        "rankNum[%u], rank[%u]", DURATION_US(TIME_NOW() - startut), nRanks, rank);

    return HCCL_SUCCESS;
}

HcclResult HcclCommInitRootInfoConfig(uint32_t nRanks, const HcclRootInfo *rootInfo, uint32_t rank,
    const HcclCommConfig *config, HcclComm *comm)
{
    HcclResult ret = HCCL_SUCCESS;
    string identifier;
    ret = HcclCommInitRootInfoConfigInner(nRanks, rootInfo, rank, config, comm, identifier);
    if (g_topoDetectServerPtrMap.find(identifier) != g_topoDetectServerPtrMap.end()) {
        g_topoDetectServerPtrMap[identifier] = nullptr;
    }
    return ret;
}

HcclResult HcclSetConfig(HcclConfig config, HcclConfigValue configValue)
{
    if (config == HCCL_DETERMINISTIC) {
        std::string hcclDeterministicEnv = SalGetEnv("HCCL_DETERMINISTIC");
        if (hcclDeterministicEnv == "EmptyString") {
            if (configValue.value == 1) {
                CHK_RET(SetDeterministic(true));
                HCCL_INFO("[HcclSetConfig] Set HCCL_DETERMINISTIC is true");
            } else if (configValue.value == 0) {
                CHK_RET(SetDeterministic(false));
                HCCL_INFO("[HcclSetConfig] Set HCCL_DETERMINISTIC is false");
            } else {
                HCCL_ERROR("[HcclSetConfig] HCCL_DETERMINISTIC is only support 0 or 1");
                return HCCL_E_PARA;
            }
        } else {
            HCCL_WARNING("[HcclSetConfig] HCCL_DETERMINISTIC has been setted by Env, so will not be reseted again");
            return HCCL_SUCCESS;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclGetConfig(HcclConfig config, HcclConfigValue *configValue)
{
    CHK_PTR_NULL(configValue);
    if (config == HCCL_DETERMINISTIC) {
        configValue->value = GetExternalInputHcclDeterministic() ? 1 : 0;
        HCCL_INFO("[HcclGetConfig] HCCL_DETERMINISTIC is [%d]", configValue->value);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclSetIfProfile()
{
    bool ifOpbase = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    bool state = ProfilingManagerPub::GetAllState();
    SetIfProfile(ifOpbase, state);
    return HCCL_SUCCESS;
}

HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType,
                         HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    HcclUs startut = TIME_NOW();

    HcclSetIfProfile();

    uint64_t beginTime = hrtMsprofSysCycleTime();

    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));

    CHK_PRT_RET(count == 0, HCCL_WARNING("input count is 0, return all reduce success"), HCCL_SUCCESS);
    // 入参合法性校验
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAllReduce", "comm", "nullptr", "please check comm"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(sendBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAllReduce", "sendBuf", "nullptr", "please check sendBuf"}));
    CHK_PTR_NULL(sendBuf);
    RPT_INPUT_ERR(recvBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAllReduce", "recvBuf", "nullptr", "please check recvBuf"}));
    CHK_PTR_NULL(recvBuf);

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    const std::lock_guard<std::mutex> lock(hcclComm->operatorlock_);
    // 同通信域同算子复用tag
    const string tag = "AllReduce_" + hcclComm->GetIdentifier();
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));


    CHK_RET_AND_PRINT_IDE(HcomCheckOpParam(tag.c_str(), count, dataType, stream), tag.c_str());

    CHK_RET_AND_PRINT_IDE(HcomCheckReductionOp(op), tag.c_str());

    CHK_RET_AND_PRINT_IDE(HcomCheckReduceDataType(dataType, op, devType), tag.c_str());

    u32 localRank = INVALID_VALUE_RANKID;
    CHK_RET_AND_PRINT_IDE(hcclComm->GetUserRank(localRank), tag.c_str());

    s32 streamId = 0;
    CHK_RET_AND_PRINT_IDE(hrtGetStreamId(stream, streamId), tag.c_str());

    /* 接口交互信息日志 */
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
        "tag[%s], sendBuf[%p], recvBuf[%p], count[%llu], dataType[%s], op[%s], localRank[%u], streamId[%d],"
        "comm[%p], deviceLogicId[%d]",
        tag.c_str(), sendBuf, recvBuf, count, GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str(),
        localRank, streamId, comm, deviceLogicId);

    CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));

    std::string logInfo = "Entry-HcclAllReduce: " + std::string(stackLogBuffer);
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveOpbaseKeyTraceInfo(logInfo), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(sendBuf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(recvBuf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetDefaultQosConfig(hcclComm), tag.c_str());
    CHK_RET_AND_PRINT_IDE(SetOverFlowAddr(hcclComm), tag.c_str());
    CHK_RET_AND_PRINT_IDE(hcclComm->AllReduceOutPlace(tag, sendBuf, recvBuf, count, dataType, op, stream), tag.c_str());
    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_ALLREDUCE, beginTime, count, dataType, tag));
    ResetIfProfile();
    HcclUs endut = TIME_NOW();
    /* 关键状态记录 */
    std::string endInfo = "HcclAllReduce:success,take time: " +
        std::to_string(DURATION_US(endut - startut).count()) + " us," + std::string(stackLogBuffer);
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveOpbaseKeyTraceInfo(endInfo), tag.c_str());
    return HCCL_SUCCESS;
}


HcclResult HcclBarrier(HcclComm comm, aclrtStream stream)
{
    HcclUs startut = TIME_NOW();
    HcclSetIfProfile();
    uint64_t beginTime = hrtMsprofSysCycleTime();

    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));

    // 入参合法性校验
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    // Allreduce入参定义
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    // 同通信域同算子复用tag
    const string tag = "AllReduce_" + hcclComm->GetIdentifier();

    s32 streamId = 0;
    CHK_RET_AND_PRINT_IDE(hrtGetStreamId(stream, streamId), tag.c_str());

    HCCL_PROFILER_ADD_TAG(tag, hcclComm->GetIdentifier(), HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    HCCL_PROFILER_ADD_STREAM(stream, tag, 0, AlgType::ALG_RESERVED);
    HCCL_PROFILER_ADD_OPDATA(tag, HCCL_BARRIER_DEFAULT_COUNT, hcclComm->barrierSendBuf, hcclComm->barrierRecvBuf, \
        dataType, INVALID_VALUE_RANKID, hcclComm->GetIdentifier());
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET_AND_PRINT_IDE(hcclComm->GetRankSize(rankSize), tag.c_str());
    u32 rankId = INVALID_VALUE_RANKID;
    CHK_RET_AND_PRINT_IDE(hcclComm->GetGroupRank(rankId), tag.c_str());
    HCCL_PROFILER_ADD_GROUPRANK(hcclComm->GetIdentifier(), rankSize, rankId);

    /* 接口交互信息日志 */
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
        "tag[%s], sendBuf[%p], recvBuf[%p], count[%d], dataType[%s], op[%s], streamId[%d],"
        "deviceLogicId[%d]",
        tag.c_str(), hcclComm->barrierSendBuf, hcclComm->barrierRecvBuf, HCCL_BARRIER_DEFAULT_COUNT,
        GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str(), streamId, deviceLogicId);

    CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
    std::string logInfo = "Entry-HcclBarrier:" + std::string(stackLogBuffer);
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveOpbaseKeyTraceInfo(logInfo), tag.c_str());

    CHK_RET_AND_PRINT_IDE(hcclComm->CreateBarrierMemory(), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(hcclComm->barrierSendBuf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(hcclComm->barrierRecvBuf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetDefaultQosConfig(hcclComm), tag.c_str());

    CHK_RET_AND_PRINT_IDE(hcclComm->AllReduceOutPlace(tag, hcclComm->barrierSendBuf, hcclComm->barrierRecvBuf,
        HCCL_BARRIER_DEFAULT_COUNT, dataType, op, stream, SyncMode::UNLIMITED_TIMEWAITSYNCMODE), tag.c_str());

    HCCL_PROFILER_DEL_STREAM(stream);
    HCCL_PROFILER_DEL_TAG(tag);
    HCCL_PROFILER_DEL_OPDATA(tag);
    HCCL_PROFILER_DEL_GROUPRANK(tag);
    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_ALLREDUCE, beginTime, HCCL_BARRIER_DEFAULT_COUNT,
        dataType, tag));
    ResetIfProfile();
    HcclUs endut = TIME_NOW();
    /* 关键状态记录 */
    std::string endInfo = "HcclBarrier:success,take time: " +
        std::to_string(DURATION_US(endut - startut).count()) + " us," + std::string(stackLogBuffer);
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveOpbaseKeyTraceInfo(endInfo), tag.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclBroadcast(void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm,
                         aclrtStream stream)
{
    HcclUs startut = TIME_NOW();
    HcclSetIfProfile();
    uint64_t beginTime = hrtMsprofSysCycleTime();
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));

    CHK_PRT_RET(count == 0, HCCL_WARNING("input count is 0, return broadcast success"), HCCL_SUCCESS);

    // 入参合法性校验
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclBroadcast", "comm", "nullptr", "please check comm"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(buf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclBroadcast", "buf", "nullptr", "please check buf"}));
    CHK_PTR_NULL(buf);
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    const std::lock_guard<std::mutex> lock(hcclComm->operatorlock_);
    // 同通信域同算子复用tag
    const string tag = "Broadcast_" + hcclComm->GetIdentifier();

    CHK_RET(HcomCheckOpParam(tag.c_str(), count, dataType, stream));

    HcomCollOpInfo opInfo = {"", buf, buf, count, dataType, root, HCCL_REDUCE_RESERVED};

    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET_AND_PRINT_IDE(hcclComm->GetRankSize(rankSize), tag.c_str());

    CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, root), tag.c_str());

    s32 streamId = 0;
    CHK_RET_AND_PRINT_IDE(hrtGetStreamId(stream, streamId), tag.c_str());

    /* 接口交互信息日志 */
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
        "tag[%s], buf[%p], count[%llu], dataType[%s], root[%u], streamId[%d], deviceLogicId[%d]",
        tag.c_str(), buf, count, GetDataTypeEnumStr(dataType).c_str(), root, streamId, deviceLogicId);

    CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
    std::string logInfo = "Entry-HcclBroadcast:" + std::string(stackLogBuffer);
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveOpbaseKeyTraceInfo(logInfo), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE), tag.c_str());

    CHK_RET_AND_PRINT_IDE(hcclComm->CreateOpBasedResources(HcclCMDType::HCCL_CMD_BROADCAST, tag, opInfo), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(buf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetDefaultQosConfig(hcclComm), tag.c_str());

    CHK_RET_AND_PRINT_IDE(hcclComm->BroadcastOutPlace(tag, buf, count, dataType, root, stream), tag.c_str());

    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_BROADCAST, beginTime, count, dataType, tag));
    ResetIfProfile();
    HcclUs endut = TIME_NOW();
    /* 关键状态记录 */
    std::string endInfo = "HcclBroadcast:success,take time: " +
        std::to_string(DURATION_US(endut - startut).count()) + " us," + std::string(stackLogBuffer);
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveOpbaseKeyTraceInfo(endInfo), tag.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
                             HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    HcclUs startut = TIME_NOW();
    HcclSetIfProfile();
    uint64_t beginTime = hrtMsprofSysCycleTime();
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));

    CHK_PRT_RET(recvCount == 0, HCCL_WARNING("input recvCount is 0, return reduce scatter success"), HCCL_SUCCESS);
    // 入参合法性校验
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclReduceScatter", "comm", "nullptr", "please check comm"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(sendBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclReduceScatter", "sendBuf", "nullptr", "please check sendBuf"}));
    CHK_PTR_NULL(sendBuf);
    RPT_INPUT_ERR(recvBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclReduceScatter", "recvBuf", "nullptr", "please check recvBuf"}));
    CHK_PTR_NULL(recvBuf);
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    const std::lock_guard<std::mutex> lock(hcclComm->operatorlock_);
    // 同通信域同算子复用tag
    const string tag = "ReduceScatter_" + hcclComm->GetIdentifier();
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));

    CHK_RET_AND_PRINT_IDE(HcomCheckOpParam(tag.c_str(), recvCount, dataType, stream), tag.c_str());

    CHK_RET_AND_PRINT_IDE(HcomCheckReductionOp(op), tag.c_str());

    CHK_RET_AND_PRINT_IDE(HcomCheckReduceDataType(dataType, op, devType), tag.c_str());

    s32 streamId = 0;
    CHK_RET_AND_PRINT_IDE(hrtGetStreamId(stream, streamId), tag.c_str());

    /* 接口交互信息日志 */
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
        "tag[%s], sendBuf[%p], recvBuf[%p], recvCount[%llu], dataType[%s], op[%s],"
        "streamId[%d], deviceLogicId[%d]",
        tag.c_str(), sendBuf, recvBuf, recvCount, GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str(),
        streamId, deviceLogicId);

    CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
    std::string logInfo = "Entry-HcclReduceScatter:" + std::string(stackLogBuffer);
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveOpbaseKeyTraceInfo(logInfo), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(sendBuf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(recvBuf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetDefaultQosConfig(hcclComm), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetOverFlowAddr(hcclComm), tag.c_str());

    CHK_RET_AND_PRINT_IDE(hcclComm->ReduceScatterOutPlace(tag, sendBuf, recvBuf, recvCount, dataType, op, stream),
                          tag.c_str());
    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_REDUCE_SCATTER, beginTime, recvCount,
        dataType, tag));
    ResetIfProfile();
    HcclUs endut = TIME_NOW();
    /* 关键状态记录 */
    std::string endInfo = "HcclReduceScatter:success,take time: " +
        std::to_string(DURATION_US(endut - startut).count()) + " us," + std::string(stackLogBuffer);
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveOpbaseKeyTraceInfo(endInfo), tag.c_str());
    return HCCL_SUCCESS;
}

HcclResult CheckScatterInputPara(uint64_t recvCount, HcclComm comm, void *recvBuf)
{
    CHK_PRT_RET(recvCount == 0, HCCL_WARNING("input count is 0, return scatter success"), HCCL_SUCCESS);
    // 入参合法性校验
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclScatter", "comm", "nullptr", "please check comm"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(recvBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclScatter", "recvBuf", "nullptr", "please check recvBuf"}));
    CHK_PTR_NULL(recvBuf);

    return HCCL_SUCCESS;
}

HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream)
{
    HcclUs startut = TIME_NOW();
    HcclSetIfProfile();
    uint64_t beginTime = hrtMsprofSysCycleTime();
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));

    CHK_RET(CheckScatterInputPara(recvCount, comm, recvBuf));

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    const std::lock_guard<std::mutex> lock(hcclComm->operatorlock_);
    u32 commRank = INVALID_VALUE_RANKID;
    CHK_RET(hcclComm->GetUserRank(commRank));
    if (commRank == root) { // 本rank为root节点，send_buff不为空
        RPT_INPUT_ERR(sendBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
            std::vector<std::string>({"HcclScatter", "sendBuf", "nullptr", "please check sendBuf"}));
        CHK_PTR_NULL(sendBuf);
    }

    // 同通信域同算子复用tag
    const string tag = "Scatter_" + hcclComm->GetIdentifier();

    CHK_RET_AND_PRINT_IDE(HcomCheckOpParam(tag.c_str(), recvCount, dataType, stream), tag.c_str());

    HcomCollOpInfo opInfo = {"", sendBuf, recvBuf, recvCount, dataType, root};

    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET_AND_PRINT_IDE(hcclComm->GetRankSize(rankSize), tag.c_str());

    CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, root), tag.c_str());

    s32 streamId = 0;
    CHK_RET_AND_PRINT_IDE(hrtGetStreamId(stream, streamId), tag.c_str());

    /* 接口交互信息日志 */
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
        "tag[%s], sendBuf[%p], recvBuf[%p], recvCount[%llu], dataType[%s], root[%u], streamId[%d], deviceLogicId[%d]",
        tag.c_str(), sendBuf, recvBuf, recvCount, GetDataTypeEnumStr(dataType).c_str(), root, streamId, deviceLogicId);

    CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
    std::string logInfo = "Entry-HcclScatter:" + std::string(stackLogBuffer);
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveOpbaseKeyTraceInfo(logInfo), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE), tag.c_str());

    CHK_RET_AND_PRINT_IDE(hcclComm->CreateOpBasedResources(HcclCMDType::HCCL_CMD_SCATTER, tag, opInfo), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(recvBuf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetDefaultQosConfig(hcclComm), tag.c_str());

    CHK_RET_AND_PRINT_IDE(hcclComm->ScatterOutPlace(tag, sendBuf, recvBuf, recvCount, dataType, root, stream),
                          tag.c_str());

    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_SCATTER, beginTime, recvCount, dataType,
        tag));
    ResetIfProfile();
    HcclUs endut = TIME_NOW();
    /* 关键状态记录 */
    std::string endInfo = "HcclScatter:success,take time: " +
        std::to_string(DURATION_US(endut - startut).count()) + " us, tag: " + tag;
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveOpbaseKeyTraceInfo(endInfo), tag.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclAllGather(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType,
                         HcclComm comm, aclrtStream stream)
{
    HcclUs startut = TIME_NOW();
    HcclSetIfProfile();
    uint64_t beginTime = hrtMsprofSysCycleTime();
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));

    CHK_PRT_RET(sendCount == 0, HCCL_WARNING("input sendCount is 0, return HcclAllGather success"), HCCL_SUCCESS);
    // 入参合法性校验
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAllGather", "comm", "nullptr", "please check comm"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(sendBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAllGather", "sendBuf", "nullptr", "please check sendBuf"}));
    CHK_PTR_NULL(sendBuf);
    RPT_INPUT_ERR(recvBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAllGather", "recvBuf", "nullptr", "please check recvBuf"}));
    CHK_PTR_NULL(recvBuf);
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    const std::lock_guard<std::mutex> lock(hcclComm->operatorlock_);
    // 同通信域同算子复用tag
    const std::string tag = "AllGather_" + hcclComm->GetIdentifier();
    CHK_RET_AND_PRINT_IDE(HcomCheckOpParam(tag.c_str(), sendCount, dataType, stream), tag.c_str());

    s32 streamId = 0;
    CHK_RET_AND_PRINT_IDE(hrtGetStreamId(stream, streamId), tag.c_str());

    /* 接口交互信息日志 */
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
        "tag[%s], sendBuf[%p], recvBuf[%p], sendCount[%llu], dataType[%s], streamId[%d],"
        "deviceLogicId[%d]",
        tag.c_str(), sendBuf, recvBuf, sendCount, GetDataTypeEnumStr(dataType).c_str(), streamId, deviceLogicId);

    CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
    std::string logInfo = "Entry-HcclAllGather:" + std::string(stackLogBuffer);
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveOpbaseKeyTraceInfo(logInfo), tag.c_str());
    CHK_RET_AND_PRINT_IDE(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(sendBuf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(recvBuf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetDefaultQosConfig(hcclComm), tag.c_str());

    CHK_RET_AND_PRINT_IDE(hcclComm->AllGatherOutPlace(tag, sendBuf, recvBuf, sendCount, dataType, stream), tag.c_str());
    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_ALLGATHER, beginTime, sendCount, dataType,
        tag));
    ResetIfProfile();
    HcclUs endut = TIME_NOW();
    /* 关键状态记录 */
    std::string endInfo = "HcclAllGather:success,take time: " +
        std::to_string(DURATION_US(endut - startut).count()) + " us," + std::string(stackLogBuffer);
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveOpbaseKeyTraceInfo(endInfo), tag.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclSend(void* sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank,
                    HcclComm comm, aclrtStream stream)
{
    HcclUs startut = TIME_NOW();
    HcclSetIfProfile();
    uint64_t beginTime = hrtMsprofSysCycleTime();
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));

    CHK_PRT_RET(count == 0, HCCL_WARNING("input count is 0, return HcclSend success"), HCCL_SUCCESS);
    // 入参合法性校验
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(stream);

    CHK_RET(HcomCheckCount(count));
    CHK_RET(HcomCheckDataType(dataType));

    s32 streamId = 0;
    CHK_RET(hrtGetStreamId(stream, streamId));

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);

    // 同算子复用tag，为实现通信域复用，根据srRank和dstRank构造Tag
    u32 localRank = INVALID_VALUE_RANKID;
    CHK_RET(hcclComm->GetGroupRank(localRank));

    const string tag = "worldCommSendRecv_" + std::to_string(localRank) + "_" + std::to_string(destRank) + "_" +
        hcclComm->GetIdentifier();

    HcomCollOpInfo opInfo = {"", sendBuf, sendBuf, count, dataType, 0, HCCL_REDUCE_RESERVED};

    HCCL_PROFILER_ADD_TAG(tag, hcclComm->GetIdentifier(), HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    HCCL_PROFILER_ADD_STREAM(stream, tag, 0, AlgType::ALG_RESERVED);

    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET_AND_PRINT_IDE(hcclComm->GetRankSize(rankSize), tag.c_str());
    HCCL_PROFILER_ADD_OPDATA(tag, count, sendBuf, sendBuf, dataType, INVALID_VALUE_RANKID, hcclComm->GetIdentifier());
    HCCL_PROFILER_ADD_GROUPRANK(hcclComm->GetIdentifier(), rankSize, localRank);

    /* 接口交互信息日志 */
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
        "tag[%s], sendBuf[%p], count[%llu], dataType[%s], streamId[%d], deviceLogicId[%d]",
        tag.c_str(), sendBuf, count, GetDataTypeEnumStr(dataType).c_str(), streamId, deviceLogicId);

    CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
    std::string logInfo = "Entry-HcclSend:" + std::string(stackLogBuffer);
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveOpbaseKeyTraceInfo(logInfo), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE), tag.c_str());

    CHK_RET_AND_PRINT_IDE(hcclComm->CreateOpBasedResources(HcclCMDType::HCCL_CMD_SEND, tag, opInfo), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(sendBuf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetDefaultQosConfig(hcclComm), tag.c_str());

    CHK_RET_AND_PRINT_IDE(hcclComm->SendOutPlace(tag, sendBuf, count, dataType, destRank, stream), tag.c_str());

    HCCL_PROFILER_DEL_STREAM(stream);
    HCCL_PROFILER_DEL_TAG(tag);
    HCCL_PROFILER_DEL_OPDATA(tag);
    HCCL_PROFILER_DEL_GROUPRANK(tag);
    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_SEND, beginTime, count, dataType, tag));
    ResetIfProfile();
    HcclUs endut = TIME_NOW();
    /* 关键状态记录 */
    std::string endInfo = "HcclSend:success,take time: " +
        std::to_string(DURATION_US(endut - startut).count()) + " us," + std::string(stackLogBuffer);
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveOpbaseKeyTraceInfo(endInfo), tag.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclRecv(void* recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank,
                    HcclComm comm, aclrtStream stream)
{
    HcclUs startut = TIME_NOW();
    HcclSetIfProfile();
    uint64_t beginTime = hrtMsprofSysCycleTime();
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));

    CHK_PRT_RET(count == 0, HCCL_WARNING("input count is 0, return HcclRecv success"), HCCL_SUCCESS);
    // 入参合法性校验
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(stream);

    CHK_RET(HcomCheckCount(count));
    CHK_RET(HcomCheckDataType(dataType));

    s32 streamId = 0;
    CHK_RET(hrtGetStreamId(stream, streamId));

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);

    // 同算子复用tag，为实现通信域复用，根据srRank和dstRank构造Tag
    u32 localRank = INVALID_VALUE_RANKID;
    CHK_RET(hcclComm->GetGroupRank(localRank));

    const string tag = "worldCommSendRecv_" + std::to_string(srcRank) + "_" + std::to_string(localRank) + "_" +
        hcclComm->GetIdentifier();

    HcomCollOpInfo opInfo = {"", recvBuf, recvBuf, count, dataType, 0, HCCL_REDUCE_RESERVED};

    HCCL_PROFILER_ADD_TAG(tag, hcclComm->GetIdentifier(), HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    HCCL_PROFILER_ADD_STREAM(stream, tag, 0, AlgType::ALG_RESERVED);

    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET_AND_PRINT_IDE(hcclComm->GetRankSize(rankSize), tag.c_str());
    HCCL_PROFILER_ADD_OPDATA(tag, count, recvBuf, recvBuf, dataType, INVALID_VALUE_RANKID, hcclComm->GetIdentifier());
    HCCL_PROFILER_ADD_GROUPRANK(hcclComm->GetIdentifier(), rankSize, localRank);

    /* 接口交互信息日志 */
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
        "tag[%s], recvBuf[%p], count[%llu], dataType[%s], streamId[%d], deviceLogicId[%d]",
        tag.c_str(), recvBuf, count, GetDataTypeEnumStr(dataType).c_str(), streamId, deviceLogicId);

    CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
    std::string logInfo = "Entry-HcclRecv:" + std::string(stackLogBuffer);
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveOpbaseKeyTraceInfo(logInfo), tag.c_str());
    CHK_RET_AND_PRINT_IDE(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE), tag.c_str());

    CHK_RET_AND_PRINT_IDE(hcclComm->CreateOpBasedResources(HcclCMDType::HCCL_CMD_RECEIVE, tag, opInfo), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(recvBuf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetDefaultQosConfig(hcclComm), tag.c_str());

    CHK_RET_AND_PRINT_IDE(hcclComm->ReceiveOutPlace(tag, recvBuf, count, dataType, srcRank, stream), tag.c_str());

    HCCL_PROFILER_DEL_STREAM(stream);
    HCCL_PROFILER_DEL_TAG(tag);
    HCCL_PROFILER_DEL_OPDATA(tag);
    HCCL_PROFILER_DEL_GROUPRANK(tag);
    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_RECEIVE, beginTime, count, dataType, tag));
    ResetIfProfile();
    HcclUs endut = TIME_NOW();
    /* 关键状态记录 */
    std::string endInfo = "HcclRecv:success,take time: " +
        std::to_string(DURATION_US(endut - startut).count()) + " us," + std::string(stackLogBuffer);
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveOpbaseKeyTraceInfo(endInfo), tag.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclCommDestroy(HcclComm comm)
{
    HcclUs startut = TIME_NOW();
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
    CHK_PRT_RET(comm == nullptr, HCCL_WARNING("[Destroy][HcclComm]An empty comm given, skip destroy."), HCCL_SUCCESS);

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    CHK_RET(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));
    if (hcclComm->IsNeedResetDevice()) {
        HCCL_RUN_INFO("op_base com destroy, com is not global com");
        s32 logicDeviceId = 0;
        hcclComm->GetDeviceId(logicDeviceId);
        HCCL_RUN_INFO("[HcclCommDestroy] reset logicDeviceId[%d]", logicDeviceId);
        CHK_PRT_RET(hrtResetDevice(logicDeviceId) != HCCL_SUCCESS,
            HCCL_ERROR("[HcclCommDestroy] reset fail logicDeviceId[%d]", logicDeviceId), HCCL_E_INTERNAL);
        g_hcclDeviceId = logicDeviceId;
    }

    HcclOpInfoCtx& opBaseHcom = GetHcclOpInfoCtx();
    string group;
    if (comm == opBaseHcom.pComm.get()) {
        group = opBaseHcom.pComm->GetIdentifier();
        opBaseHcom.pComm = nullptr;
        HcclCloseCommConnections(group);
    } else {
        HCCL_RUN_INFO("com is not global com");
        group = hcclComm->GetIdentifier();
    }

    HcomUnSetGroupTopoInfo(group.c_str());

    std::unique_lock<std::mutex> lock(opBaseHcom.opGroupMapMutex);
    auto iter = opBaseHcom.opGroup2CommMap.find(group);
    if (iter != opBaseHcom.opGroup2CommMap.end()) {
        opBaseHcom.opGroup2CommMap.erase(group);
        HcclCloseCommConnections(group);
    } else {
        HCCL_ERROR("[HcclCommDestroy] comm is not exist, comm=%p, group=%s, deviceLogicId=%d", comm, group.c_str(),
            deviceLogicId);
        return HCCL_E_PARA;
    }

    HcclUs endut = TIME_NOW();
    /* 关键状态记录 */
    HCCL_RUN_INFO("op_base comm destroy complete,take time [%lld]us, rankNum[%u], rank[%u], deviceLogicId[%d]",
        DURATION_US(endut - startut), opBaseHcom.rankTable.rankNum, opBaseHcom.params.rank,
        deviceLogicId);
    return HCCL_SUCCESS;
}

HcclResult HcclGenerateCommId(hccl::HcclCommParams &params)
{
    s32 sRet = memset_s(params.id.internal, HCCL_ROOT_INFO_BYTES, 0, sizeof(params.id.internal));
    CHK_PRT_RET(sRet != EOK, HCCL_ERROR("[GenerateCommId]memory set error. return[%d].", sRet), HCCL_E_PARA);

    HcclRootInfo uniqueId;
    std::string group;
    CHK_RET(hcclComm::GetUniqueId(&uniqueId));

    if (!params.isHeterogComm) {
        group = "hccl_world_group";
    } else {
        group = "hccl_heterog_group";
    }

    sRet = snprintf_s(params.id.internal, HCCL_ROOT_INFO_BYTES, HCCL_ROOT_INFO_BYTES - 1, "%s%s%s",
        uniqueId.internal, "-", group.c_str());
    CHK_PRT_RET(sRet == -1, HCCL_ERROR("[GenerateCommId]errNo[0x%016llx] sal snprintf_s error",
        HCCL_ERROR_CODE(HCCL_E_INTERNAL)), HCCL_E_INTERNAL);
    HCCL_INFO("params.id.internal [%s]", params.id.internal);
    return HCCL_SUCCESS;
}

HcclResult InitOtherInfo(hccl::HcclCommParams &params, const char *rankTable)
{
    // 记录版本信息
    std::string curVersion = GetExternalInputCannVersion();
    CHK_RET(RankConsistent::GetInstance().RecordVerInfo(curVersion));

    // ranktableCRC计算
    if (rankTable == nullptr) {
        HCCL_INFO("rank table is null, rankTableCrc is 0.");
    } else {
        HcclResult ret = HcomCalcCRC(params, rankTable);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Init][OtherInfo]errNo[0x%016llx] calc ranktable crc error",
            HCCL_ERROR_CODE(HCCL_E_INTERNAL)), HCCL_E_INTERNAL);
    }

    // 生成通信域标识符
    HcclResult ret = HcclGenerateCommId(params);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Init][OtherInfo]errNo[0x%016llx] generate CommId error, params: dest[%p]",
            HCCL_ERROR_CODE(HCCL_E_INTERNAL), params.id.internal), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterLoop(const std::string &tag, void *inputPtr, void *outputPtr, const u64 &count,
                             HcclDataType dataType, HcclReduceOp op, hccl::hcclComm *hcclComm, rtStream_t stream)
{
    HcclResult ret;
    void *commInputPtr = nullptr;
    void *commOutputPtr = nullptr;
    u64 commInputSize, commOutputSize;

    CHK_RET(hcclComm->GetInCCLbuffer(commInputPtr, commInputSize));

    CHK_RET(hcclComm->GetOutCCLbuffer(commOutputPtr, commOutputSize));

    u32 unitSize;
    CHK_RET(SalGetDataTypeSize(dataType, unitSize));

    char *curInputPtr = static_cast<char *>(inputPtr);
    char *curOutputPtr = static_cast<char *>(outputPtr);
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(hcclComm->GetRankSize(rankSize));

    u32 qosCfg = INVALID_QOSCFG; // qos不使能的情况下为全F
    CHK_RET(hcclComm->GetQosCfg(qosCfg));

    u64 maxCountPerLoop = commInputSize / (rankSize * unitSize); // 中转内存单次最多能够接受的output count
    u64 curCount = 0;

    for (u64 countLeft = count, inputOffset = 0, outputOffset = 0; countLeft > 0; countLeft -= curCount) {
        curInputPtr += inputOffset;
        curOutputPtr += outputOffset;
        HCCL_INFO("-OP_BASE-ReduceScatterLoop:inputOffset[%llu], outputOffset[%llu]", inputOffset, outputOffset);
        // 判断剩余数据量对应的input size是否大于中转input size
        curCount = ((countLeft * unitSize * rankSize) > commInputSize) ? maxCountPerLoop : countLeft;
        u64 curSize = curCount * unitSize; // 单位：字节

        for (u32 i = 0; i < rankSize; i++) {
            // 拷贝input上每个slice的数据到中转内存，源端每个slice的size固定为output的size
            ret = hrtMemAsyncCopyByQos(static_cast<char *>(commInputPtr) + curSize * i, curSize,
                curInputPtr + count * unitSize * i, curSize, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE,
                stream, qosCfg);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Loop][ReduceScatter]In OP_BASE inputbuffer transit,[%u]slice memcopy "\
                    "failed", i), HCCL_E_MEMORY);
        }
        CHK_RET(RankConsistent::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_REDUCE_SCATTER,
            const_cast<char *>(tag.c_str()), curCount, dataType, op, commInputSize, commOutputSize));

        ret = hcclComm->ReduceScatter(const_cast<char *>(tag.c_str()), commInputPtr, commOutputPtr,
            curCount, dataType, op, stream);

        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Loop][ReduceScatter]errNo[0x%016llx] op_base hcclComm reduce_scatter error, "\
            "tag[%s], input_ptr[%p], output_ptr[%p], count[%llu], data_type[%s], op[%s]",
            HCCL_ERROR_CODE(ret), tag.c_str(), commInputPtr, commOutputPtr, curCount,
            GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str()), ret);
        CHK_RET(RankConsistent::GetInstance().DelOpPara(tag));

        CHK_RET(hrtMemAsyncCopyByQos(curOutputPtr, curSize, commOutputPtr, curSize,
            HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE, stream, qosCfg));

        CHK_PRT_RET((curCount == 0), HCCL_ERROR("[Loop][ReduceScatter]In OP_BASE curCount is zero"), HCCL_E_PARA);
        inputOffset = curSize;
        outputOffset = curSize;
    }

    return HCCL_SUCCESS;
}

// 获取算子所需workspace memory大小[byte]
HcclResult HcclGetOpBasedMemSize(const HcclCMDType &opType, u64 &size,
    const HcomCollOpInfo &opInfo)
{
    u64 opMemSize = 0;

    if (opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
        // ReduceScatter 算子所需memory大小为 GetExternalInputCCLBuffSize()
        DevType devType;
        CHK_RET(hrtGetDeviceType(devType));
        if (IsSupportSDMAReduce(opInfo.inputAddr, opInfo.outputAddr, opInfo.dataType, opInfo.reduceOp) &&
            IsSupportRDMAReduce(opInfo.dataType, opInfo.reduceOp) && devType == DevType::DEV_TYPE_910B) {
                opMemSize = 0;
            } else {
                opMemSize = GetExternalInputCCLBuffSize();
            }
    } else {
        opMemSize = 0;
    }
    size = HCCL_WORKSPACE_MEM_32_KB + opMemSize;
    HCCL_INFO("workspace memory size: op[%d], memory size[%llu]", opType, size);
    return HCCL_SUCCESS;
}

HcclResult HcclGetRankSize(HcclComm comm, uint32_t *rankSize)
{
    // 入参合法性校验
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(rankSize);

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    u32 tmpRankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(hcclComm->GetRankSize(tmpRankSize));
    *rankSize = tmpRankSize;
    /* 关键状态记录 */
    HCCL_INFO("HcclGetRankSize success, rankSizePtr[%p], rankSize[%u]", rankSize, tmpRankSize);
    return HCCL_SUCCESS;
}

HcclResult HcclGetRankId(HcclComm comm, uint32_t *rank)
{
    // 入参合法性校验
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(rank);

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    u32 tmpRankId = INVALID_VALUE_RANKID;
    CHK_RET(hcclComm->GetUserRank(tmpRankId));
    *rank = tmpRankId;
    /* 关键状态记录 */
    HCCL_INFO("HcclGetRankId success, rankIdPtr[%p], rankId[%u]", rank, tmpRankId);
    return HCCL_SUCCESS;
}

HcclResult HcclAlltoAll(const void *sendBuf, uint64_t sendCount, HcclDataType sendType, const void *recvBuf,
    uint64_t recvCount, HcclDataType recvType, HcclComm comm, aclrtStream stream)
{
    HcclSetIfProfile();
    HcclUs startut = TIME_NOW();
    uint64_t beginTime = hrtMsprofSysCycleTime();
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
    CHK_PRT_RET(sendCount == 0 && recvCount == 0,
        HCCL_WARNING("sendCount and recvCount are both 0, return alltoall success"), HCCL_SUCCESS);
    RPT_INPUT_ERR(sendBuf == nullptr, "EI0003",\
        std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAlltoAll", "sendBuf", "nullptr", "please check sendBuf"}));
    CHK_PTR_NULL(sendBuf);
    RPT_INPUT_ERR(recvBuf == nullptr, "EI0003",\
        std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAlltoAll", "recvBuf", "nullptr", "please check recvBuf"}));
    CHK_PTR_NULL(recvBuf);
    CHK_PRT_RET(sendCount != recvCount,
        HCCL_ERROR("sendCount[%llu] and recvCount[%llu] are not equal, please check params",
            sendCount, recvCount), HCCL_E_PARA);
    CHK_PRT_RET(sendType != recvType,
        HCCL_ERROR("sendType[%s] and recvType[%s] are not equal, please check params",
            GetDataTypeEnumStr(sendType).c_str(), GetDataTypeEnumStr(recvType).c_str()), HCCL_E_PARA);

    RPT_INPUT_ERR(stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAlltoAll", "stream", "nullptr", "please check stream"}));
    CHK_PTR_NULL(stream);
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAlltoAll", "comm", "nullptr", "please check comm"}));
    CHK_PTR_NULL(comm);

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);

    const std::string tag = HCCL_ALLTOALL + "_" + hcclComm->GetIdentifier();
    CHK_RET(HcomCheckOpParam(tag.c_str(), 0, sendType, stream));
    CHK_RET(HcomCheckDataType(recvType));

    HCCL_PROFILER_ADD_TAG(tag, hcclComm->GetIdentifier(), HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    HCCL_PROFILER_ADD_STREAM(stream, tag, 0, AlgType::ALG_RESERVED);

    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET_AND_PRINT_IDE(hcclComm->GetRankSize(rankSize), tag.c_str());
    u32 localRank = INVALID_VALUE_RANKID;
    CHK_RET(hcclComm->GetGroupRank(localRank));
    HCCL_PROFILER_ADD_OPDATA(tag, sendCount, sendBuf, recvBuf, sendType, INVALID_VALUE_RANKID, \
        hcclComm->GetIdentifier());
    HCCL_PROFILER_ADD_GROUPRANK(hcclComm->GetIdentifier(), rankSize, localRank);

    s32 streamId = 0;
    CHK_RET_AND_PRINT_IDE(hrtGetStreamId(stream, streamId), tag.c_str());

    // 接口交互信息日志
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
        "tag[%s], sendCount[%llu], recvCount[%llu], sendType[%s], recvType[%s], streamId[%d],"
        "deviceLogicId[%d]",
        tag.c_str(), sendCount, recvCount, GetDataTypeEnumStr(sendType).c_str(), GetDataTypeEnumStr(recvType).c_str(),
        streamId, deviceLogicId);

    CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
    std::string logInfo = "Entry-HcclAlltoAll:" + std::string(stackLogBuffer);
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveOpbaseKeyTraceInfo(logInfo), tag.c_str());
    CHK_RET_AND_PRINT_IDE(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(sendBuf), tag.c_str());
    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(recvBuf), tag.c_str());

    /* 记录cclBufferSize用于一致性校验 */
    CHK_RET_AND_PRINT_IDE(RankConsistent::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_ALLTOALL,
        tag.c_str(), sendCount, sendType, hcclComm->GetConfigInCCLbufferSize(), 0, nullptr), tag.c_str());

    CHK_RET_AND_PRINT_IDE(hcclComm->AlltoAll(sendBuf, sendCount, sendType, recvBuf, recvCount, recvType, stream, tag),
                          tag.c_str());

    HCCL_PROFILER_DEL_STREAM(stream);
    HCCL_PROFILER_DEL_TAG(tag);
    HCCL_PROFILER_DEL_OPDATA(tag);
    HCCL_PROFILER_DEL_GROUPRANK(tag);
    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_ALLTOALL, beginTime, sendCount, sendType,
        tag));

    HcclUs endut = TIME_NOW();
    std::string endInfo = "HcclAlltoAll:success,take time: " +
        std::to_string(DURATION_US(endut - startut).count()) + " us," + std::string(stackLogBuffer);
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveOpbaseKeyTraceInfo(endInfo), tag.c_str());
    return HCCL_SUCCESS;
}

// sendBuf & recvBuf为device mem, 其它为host mem
HcclResult HcclAlltoAllV(const void *sendBuf, const void *sendCounts, const void *sdispls, HcclDataType sendType,
                         const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType,
                         HcclComm comm, aclrtStream stream)
{
    HcclSetIfProfile();
    HcclUs startut = TIME_NOW();
    uint64_t beginTime = hrtMsprofSysCycleTime();
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));

    RPT_INPUT_ERR(sendCounts == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAlltoAllV", "sendCounts", "nullptr", "please check sendCounts"}));
    CHK_PTR_NULL(sendCounts);
    RPT_INPUT_ERR(sdispls == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAlltoAllV", "sdispls", "nullptr", "please check sdispls"}));
    CHK_PTR_NULL(sdispls);
    RPT_INPUT_ERR(recvCounts == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAlltoAllV", "recvCounts", "nullptr", "please check recvCounts"}));
    CHK_PTR_NULL(recvCounts);
    RPT_INPUT_ERR(rdispls == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAlltoAllV", "rdispls", "nullptr", "please check rdispls"}));
    CHK_PTR_NULL(rdispls);
    RPT_INPUT_ERR(stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAlltoAllV", "stream", "nullptr", "please check stream"}));
    CHK_PTR_NULL(stream);
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAlltoAllV", "comm", "nullptr", "please check comm"}));
    CHK_PTR_NULL(comm);

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);

    u32 rankSize = 0;
    CHK_RET(hcclComm->GetRankSize(rankSize));
    CHK_RET(HcomCheckAlltoAllVExternalMem(sendBuf, sendCounts, recvBuf, recvCounts, rankSize));

    const std::string tag = HCCL_ALLTOALLV + "_" + hcclComm->GetIdentifier();
    CHK_RET_AND_PRINT_IDE(HcomCheckOpParam(tag.c_str(), 0, sendType, stream), tag.c_str());
    CHK_RET_AND_PRINT_IDE(HcomCheckDataType(recvType), tag.c_str());

    HCCL_PROFILER_ADD_TAG(tag, hcclComm->GetIdentifier(), HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    HCCL_PROFILER_ADD_TAG(HCCL_ALLTOALL_PARA_ALLGATHER, hcclComm->GetIdentifier(),
        HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    HCCL_PROFILER_ADD_STREAM(stream, tag, 0, AlgType::ALG_RESERVED);

    u32 localRank = INVALID_VALUE_RANKID;
    CHK_RET(hcclComm->GetGroupRank(localRank));
    HCCL_PROFILER_ADD_OPDATA(tag, 0, sendBuf, recvBuf, sendType, INVALID_VALUE_RANKID, hcclComm->GetIdentifier());
    HCCL_PROFILER_ADD_GROUPRANK(hcclComm->GetIdentifier(), rankSize, localRank);

    s32 streamId = 0;
    CHK_RET_AND_PRINT_IDE(hrtGetStreamId(stream, streamId), tag.c_str());

    /* 接口交互信息日志 */
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
        "tag[%s], sendBuf[%p], recvBuf[%p], sendCounts[%p], recvCounts[%p], sendType[%s],"
        "recvType[%s], streamId[%d], deviceLogicId[%d]",
        tag.c_str(), sendBuf, recvBuf, sendCounts, recvCounts, GetDataTypeEnumStr(sendType).c_str(),
        GetDataTypeEnumStr(recvType).c_str(), streamId, deviceLogicId);

    CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
    std::string logInfo = "Entry-HcclAlltoAllV:" + std::string(stackLogBuffer);
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveOpbaseKeyTraceInfo(logInfo), tag.c_str());
    CHK_RET_AND_PRINT_IDE(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE), tag.c_str());

    if (sendBuf != nullptr) {
        CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(sendBuf), tag.c_str());
    }
    if (recvBuf != nullptr) {
        CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(recvBuf), tag.c_str());
    }

    /* 记录cclBufferSize用于一致性校验 */
    CHK_RET_AND_PRINT_IDE(RankConsistent::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_ALLTOALLV,
        tag.c_str(), 0, HCCL_DATA_TYPE_RESERVED, hcclComm->GetConfigInCCLbufferSize(), 0, nullptr), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetDefaultQosConfig(hcclComm), tag.c_str());

    if (!GetExternalInputHcclEnableFfts()) {
        CHK_RET_AND_PRINT_IDE(hcclComm->AlltoAllV(sendBuf, sendCounts, sdispls, sendType, recvBuf,
            recvCounts, rdispls, recvType, stream, tag), tag.c_str());
    } else {
        CHK_RET_AND_PRINT_IDE(hcclComm->AlltoAllVOutPlace(sendBuf, sendCounts, sdispls, sendType, recvBuf,
            recvCounts, rdispls, recvType, stream, tag), tag.c_str());
    }

    u64 sendCount = 0;
    for (u32 i = 0; i < rankSize; i++) {
        u64 curSendCount = *(static_cast<const u64 *>(sendCounts) + i) + *(static_cast<const u64 *>(sdispls) + i);
        sendCount = std::max(sendCount, curSendCount);
    }
    HCCL_PROFILER_DEL_STREAM(stream);
    HCCL_PROFILER_DEL_TAG(HCCL_ALLTOALL_PARA_ALLGATHER);
    HCCL_PROFILER_DEL_TAG(tag);
    HCCL_PROFILER_DEL_OPDATA(tag);
    HCCL_PROFILER_DEL_GROUPRANK(tag);

    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_ALLTOALLV, beginTime, sendCount, sendType,
        tag));

    HcclUs endut = TIME_NOW();
    /* 关键状态记录 */
    std::string endInfo = "HcclAlltoAllV:success,take time: " +
        std::to_string(DURATION_US(endut - startut).count()) + " us," + std::string(stackLogBuffer);
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveOpbaseKeyTraceInfo(endInfo), tag.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclAlltoAllVC(const void *sendBuf, const void *sendCountMatrix,
    HcclDataType sendType, const void *recvBuf, HcclDataType recvType,
    HcclComm comm, rtStream_t stream)
{
    HcclSetIfProfile();
    HcclUs startut = TIME_NOW();
    uint64_t beginTime = hrtMsprofSysCycleTime();
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));

    RPT_INPUT_ERR(sendCountMatrix == nullptr, "EI0003",\
        std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAlltoAllVC", "sendCountMatrix", "nullptr", "please check sendCountMatrix"}));
    CHK_PTR_NULL(sendCountMatrix);
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAlltoAllVC", "comm", "nullptr", "please check comm"}));
    CHK_PTR_NULL(comm);

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);

    u32 rankSize = 0;
    CHK_RET(hcclComm->GetRankSize(rankSize));
    u32 rank = 0;
    hcclComm->GetUserRank(rank);
    u32 userRank = 0;
    hcclComm->GetGroupRank(userRank);

    CHK_RET(HcomCheckAlltoAllVCExternalMem(sendBuf, sendCountMatrix, recvBuf, rankSize, rank));
    const std::string tag = HCCL_ALLTOALLVC + "_" + hcclComm->GetIdentifier();
    CHK_RET_AND_PRINT_IDE(HcomCheckOpParam(tag.c_str(), 0, sendType, stream), tag.c_str());
    CHK_RET_AND_PRINT_IDE(HcomCheckDataType(recvType), tag.c_str());

    HCCL_PROFILER_ADD_TAG(tag, hcclComm->GetIdentifier(), HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    HCCL_PROFILER_ADD_STREAM(stream, tag, 0, AlgType::ALG_RESERVED);

    HCCL_PROFILER_ADD_OPDATA(tag, 0, sendBuf, recvBuf, sendType, INVALID_VALUE_RANKID, hcclComm->GetIdentifier());
    HCCL_PROFILER_ADD_GROUPRANK(hcclComm->GetIdentifier(), rankSize, userRank);

    s32 streamId = 0;
    CHK_RET_AND_PRINT_IDE(hrtGetStreamId(stream, streamId), tag.c_str());

    u64 sendCountMatrixHash;
    HcomGetHashFromSendCountMatrix(sendCountMatrixHash, sendCountMatrix, rankSize, tag);
    /* 接口交互信息日志 */
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
        "tag[%s], sendBuf[%p], sendCountMatrixHash[%llu], sendType[%s], recvBuf[%p],"
        "recvType[%s], streamId[%d], deviceLogicId[%d]",
        tag.c_str(), sendBuf, sendCountMatrixHash, GetDataTypeEnumStr(sendType).c_str(), recvBuf,
        GetDataTypeEnumStr(recvType).c_str(), streamId, deviceLogicId);

    CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
    std::string logInfo = "Entry-HcclAlltoAllVC:" + std::string(stackLogBuffer);
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveOpbaseKeyTraceInfo(logInfo), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE), tag.c_str());

    if (sendBuf != nullptr) {
        CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(sendBuf), tag.c_str());
    }
    if (recvBuf != nullptr) {
        CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(recvBuf), tag.c_str());
    }

    /* 记录cclBufferSize用于一致性校验 */
    CHK_RET_AND_PRINT_IDE(RankConsistent::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_ALLTOALLVC,
        tag.c_str(), 0, HCCL_DATA_TYPE_RESERVED, hcclComm->GetConfigInCCLbufferSize(), 0, nullptr), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetDefaultQosConfig(hcclComm), tag.c_str());

    if (!GetExternalInputHcclEnableFfts()) {
        CHK_RET_AND_PRINT_IDE(hcclComm->AlltoAllVC(sendBuf, sendCountMatrix, sendType, recvBuf, recvType, stream, tag),
            tag.c_str());
    } else {
        CHK_RET_AND_PRINT_IDE(hcclComm->AlltoAllVCOutPlace(sendBuf, sendCountMatrix, sendType, recvBuf,
            recvType, stream, tag), tag.c_str());
    }

    HCCL_PROFILER_DEL_STREAM(stream);
    HCCL_PROFILER_DEL_TAG(tag);
    HCCL_PROFILER_DEL_OPDATA(tag);
    HCCL_PROFILER_DEL_GROUPRANK(tag);
    u64 sendCount = 0;
    for (u32 i = 0; i < rankSize; i++) {
        sendCount += *(static_cast<const u64 *>(sendCountMatrix) + userRank * rankSize + i);
    }
    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_ALLTOALLVC, beginTime, sendCount, sendType,
        tag));
    HcclUs endut = TIME_NOW();
    /* 关键状态记录 */
    std::string endInfo = "HcclAlltoAllVC:success,take time: " +
        std::to_string(DURATION_US(endut - startut).count()) + " us," + std::string(stackLogBuffer);
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveOpbaseKeyTraceInfo(endInfo), tag.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
                      uint32_t root, HcclComm comm, aclrtStream stream)
{
    HcclSetIfProfile();
    HcclUs startut = TIME_NOW();
    uint64_t beginTime = hrtMsprofSysCycleTime();
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));

    CHK_PRT_RET(count == 0, HCCL_WARNING("input count is 0, return reduce success"), HCCL_SUCCESS);
    // 入参合法性校验
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclReduce", "comm", "nullptr", "please check comm"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(sendBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclReduce", "sendBuf", "nullptr", "please check sendBuf"}));
    CHK_PTR_NULL(sendBuf);
    RPT_INPUT_ERR(recvBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclReduce", "recvBuf", "nullptr", "please check recvBuf"}));
    CHK_PTR_NULL(recvBuf);
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    // 同通信域同算子复用tag
    const string tag = "Reduce_" + hcclComm->GetIdentifier();
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    CHK_RET_AND_PRINT_IDE(HcomCheckOpParam(tag.c_str(), count, dataType, stream), tag.c_str());

    CHK_RET_AND_PRINT_IDE(HcomCheckReductionOp(op), tag.c_str());

    CHK_RET_AND_PRINT_IDE(HcomCheckReduceDataType(dataType, op, devType), tag.c_str());

    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET_AND_PRINT_IDE(hcclComm->GetRankSize(rankSize), tag.c_str());
    CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, root), tag.c_str());

    s32 streamId = 0;
    CHK_RET_AND_PRINT_IDE(hrtGetStreamId(stream, streamId), tag.c_str());

    /* 接口交互信息日志 */
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
        "tag[%s], sendBuf[%p], recvBuf[%p], count[%llu], dataType[%s], op[%s], root[%u],"
        "streamId[%d], deviceLogicId[%d]",
        tag.c_str(), sendBuf, recvBuf, count, GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str(),
        root, streamId, deviceLogicId);

    CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
    std::string logInfo = "Entry-HcclReduce:" + std::string(stackLogBuffer);
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveOpbaseKeyTraceInfo(logInfo), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(sendBuf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(recvBuf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetDefaultQosConfig(hcclComm), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetOverFlowAddr(hcclComm), tag.c_str());

    CHK_RET_AND_PRINT_IDE(hcclComm->ReduceOutPlace(tag, sendBuf, recvBuf, count, dataType, op, root, stream),
                              tag.c_str());
  
    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_REDUCE, beginTime, count, dataType, tag));
    HcclUs endut = TIME_NOW();
    /* 关键状态记录 */
    std::string endInfo = "HcclReduce:success,take time: " +
        std::to_string(DURATION_US(endut - startut).count()) + " us," + std::string(stackLogBuffer);
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveOpbaseKeyTraceInfo(endInfo), tag.c_str());
    return HCCL_SUCCESS;
}

HcclResult ReduceLoop(const std::string &tag, void *inputPtr, void *outputPtr, const u64 count,
    HcclDataType dataType, HcclReduceOp op, const u32 root, hccl::hcclComm *hcclComm, rtStream_t stream)
{
    HcclSetIfProfile();
    void *commInputPtr = nullptr;
    void *commOutputPtr = nullptr;
    u64 commInputSize, commOutputSize;

    HcclResult ret;
    CHK_RET(hcclComm->GetInCCLbuffer(commInputPtr, commInputSize));

    CHK_RET(hcclComm->GetOutCCLbuffer(commOutputPtr, commOutputSize));

    u32 qosCfg = INVALID_QOSCFG; // qos不使能的情况下为全F
    CHK_RET(hcclComm->GetQosCfg(qosCfg));

    u32 unitSize;
    CHK_RET(SalGetDataTypeSize(dataType, unitSize));

    char *curInputPtr = static_cast<char *>(inputPtr);
    char *curOutputPtr = static_cast<char *>(outputPtr);
    u64 inputOffset = 0;
    u64 outputOffset = 0;
    u64 countLeft = count;

    while (countLeft > 0) {
        curInputPtr += inputOffset;
        curOutputPtr += outputOffset;
        HCCL_DEBUG("-OP_BASE-ReduceLoop:inputOffset[%llu], outputOffset[%llu]", inputOffset, outputOffset);
        u64 curCount = ((countLeft * unitSize) > commInputSize) ? (commInputSize / unitSize) : countLeft; // 单次执行操作的数据量
        u64 curSize = curCount * unitSize; // 单位 byte

        HCCL_DEBUG("-OP_BASE-ReduceLoop:curInputPtr[%p], curOutputPtr[%p], curCount[%llu], curSize[%llu]",
            curInputPtr, curOutputPtr, curCount, curSize);

        u32 commRank = INVALID_VALUE_RANKID;
        CHK_RET(hcclComm->GetUserRank(commRank));

        CHK_RET(hrtMemAsyncCopyByQos(commInputPtr, curSize, curInputPtr, curSize,
            HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE, stream, qosCfg));

        /* 记录指令信息用于一致性校验 */
        CHK_RET(RankConsistent::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_REDUCE, tag,
            curCount, dataType, op, root, commInputSize, commOutputSize));

        /* 入参的正确性由HCCL确保 */
        ret = hcclComm->Reduce(const_cast<char *>(tag.c_str()), commInputPtr, commOutputPtr,
                               curCount, dataType, op, root, stream);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Loop][Reduce]errNo[0x%016llx] op_base hcclComm reduce error, tag[%s], "\
            "input_ptr[%p], output_ptr[%p], count[%llu], data_type[%s], op[%s], root[%u]",
            HCCL_ERROR_CODE(ret), tag.c_str(), commInputPtr, commOutputPtr, curCount,
            GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str(), root), ret);
        ret = RankConsistent::GetInstance().DelOpPara(tag);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Loop][Reduce]errNo[0x%016llx] delete CMD with parameters error. tag[%s]",
                HCCL_ERROR_CODE(ret), tag.c_str()), ret);

        if (commRank == root) { // 只root rank需要把数据从中转内存拷贝出去
            CHK_RET(hrtMemAsyncCopyByQos(curOutputPtr, curSize, commOutputPtr, curSize,
                HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE, stream, qosCfg));
        }

        countLeft -= curCount;
        inputOffset = curSize;
        outputOffset = curSize;
    }

    return HCCL_SUCCESS;
}

HcclResult HcclGatherAlltoAllV(HcomGatherAllToAllVParams params, HcclComm comm, aclrtStream stream)
{
    HcclUs startut = TIME_NOW();
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(params.addrInfoCountPerRank);
    CHK_PTR_NULL(params.recvcounts);
    CHK_PTR_NULL(params.gatheredbuf);
    CHK_PTR_NULL(params.rdispls);

    const u32 NUM_TWO = 2;
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    u32 rankSize = 0;
    CHK_RET(hcclComm->GetRankSize(rankSize));

    CHK_RET(SetDefaultQosConfig(hcclComm));

    // 同通信域同算子复用tag
    const string tag = "Reduce_" + hcclComm->GetIdentifier();

    std::vector<u64> addrInfoCountPerRank(rankSize, 0);
    CHK_RET_AND_PRINT_IDE(hrtMemSyncCopy(addrInfoCountPerRank.data(), rankSize * sizeof(u64),
        params.addrInfoCountPerRank, rankSize * sizeof(u64),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST), tag.c_str());
    u64 blockNum = 0;
    for (u32 index = 0; index < rankSize; index++) {
        blockNum += addrInfoCountPerRank[index];
    }
    if (blockNum != 0) {
        CHK_PTR_NULL(params.addrInfo);
    }
    std::vector<u64> addrInfo(blockNum * NUM_TWO, 0);
    CHK_RET_AND_PRINT_IDE(hrtMemSyncCopy(addrInfo.data(), addrInfo.size() * sizeof(u64), params.addrInfo,
        addrInfo.size() * sizeof(u64), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST), tag.c_str());

    char stackLogBuffer[LOG_TMPBUF_SIZE];
    s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
        "tag[%s]", tag.c_str());

    CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
    std::string logInfo = "Entry-HcclGatherAlltoAllV:" + std::string(stackLogBuffer);
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveOpbaseKeyTraceInfo(logInfo), tag.c_str());

    // 执行gather
    u64 sendCounts = static_cast<u64>(rankSize);
    u64 sdispls = static_cast<u64>(rankSize);

    // step1 gather
    GatherPara gatherPara;
    gatherPara.addrInfo = addrInfo;
    gatherPara.rankSize = rankSize;
    gatherPara.addrInfoCountPerRank = addrInfoCountPerRank;
    gatherPara.addrLength = params.addrLength;
    CHK_RET_AND_PRINT_IDE(RunGather(&sendCounts, &sdispls, params.gatheredbuf, gatherPara), tag.c_str());

    // step2 alltoallv
    CHK_RET_AND_PRINT_IDE(HcclAlltoAllV(params.gatheredbuf, &sendCounts, &sdispls, params.recvtype, params.recvbuf,
        params.recvcounts, params.rdispls, params.recvtype, comm, stream), tag.c_str());

    HcclUs endut = TIME_NOW();
    std::string endInfo = "HcclGatherAlltoAllV:success,take time: " +
        std::to_string(DURATION_US(endut - startut).count()) + " us," + std::string(stackLogBuffer);
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveOpbaseKeyTraceInfo(endInfo), tag.c_str());
    return HCCL_SUCCESS;
}

/*
 * **********************************************************************
 * 单算子GatherAllToAllV step1 执行gather，出参作为step2的入参
 * **********************************************************************
 */
HcclResult RunGather(u64 *sendCounts, u64 *sdispls, void *sendDevBuf, GatherPara &gatherPara)
{
    u64 memSize = 0;
    const u32 GATHER_THREAD_NUM = 16;
    const u32 NUM_TWO = 2;
    u64 perThreadCount = gatherPara.addrInfo.size() / NUM_TWO / GATHER_THREAD_NUM;
    std::vector<u64> perThreadCounts(GATHER_THREAD_NUM, perThreadCount);
    perThreadCounts[GATHER_THREAD_NUM - 1] =
        gatherPara.addrInfo.size() / NUM_TWO - perThreadCount * (GATHER_THREAD_NUM - 1);
    std::vector<u64> offset(GATHER_THREAD_NUM, 0);
    if (gatherPara.addrLength == -1) { // 数据包长度不一样的情况
        u32 offsetIndex = 0;
        for (u32 index = 1; index < gatherPara.addrInfo.size(); index += NUM_TWO) { // 由于是二元组，单数为数据包的长度，每个循环+2
            /* 如果数据包数量小于线程数量则offset全置为0 */
            if (perThreadCount != 0 && index / NUM_TWO % perThreadCount == 0 && offsetIndex < GATHER_THREAD_NUM) {
                /* 条件1：当累加的数量达到perThreadCount时往offset中填入累加值，即可计算出前面thread产生的offset值 */
                /* 条件2：由于第0个thread的offset为0，后面的线程的offset为前面线程处理数据量的累加，因此对最后一个值弃之不用 */
                offset[offsetIndex] = memSize;
                offsetIndex++;
            }
            memSize += gatherPara.addrInfo[index];
        }
    } else {
        memSize = gatherPara.addrInfo.size() / NUM_TWO * gatherPara.addrInfo[1];
        for (u32 index = 0; index < GATHER_THREAD_NUM; index++) {
            offset[index] = index * perThreadCount * gatherPara.addrInfo[1];
        }
    }

    // 多线程拷贝
    HostMem tmpHostMem = HostMem::alloc(memSize);
    std::vector<std::unique_ptr<std::thread>> threads(GATHER_THREAD_NUM);
    for (u32 num = 0; num < GATHER_THREAD_NUM; num++) {
        OpBaseMemPara memPara;
        memPara.beginIndex = num * perThreadCount * NUM_TWO;
        memPara.count = perThreadCounts[num];
        memPara.tmpMemSize = memSize;
        threads[num].reset(new (std::nothrow) std::thread(&GatherMemCopyThread, tmpHostMem.ptr(),
            offset[num], std::ref(gatherPara.addrInfo), memPara));
        CHK_PRT_RET(!threads[num], HCCL_ERROR("[Exec][EnqueueGatherAlltoAllV]threads[%u] reset "\
            "failed ", num), HCCL_E_INTERNAL);
    }

    // 构造入参
    auto ret = memset_s(sendCounts, gatherPara.rankSize * sizeof(u64), 0, gatherPara.rankSize * sizeof(u64));
    CHK_PRT_RET(ret != EOK, HCCL_ERROR("[Exec][EnqueueGatherAlltoAllV] mem set failed, count[%lld]",
        gatherPara.rankSize * sizeof(u64)), HCCL_E_SYSCALL);
    u64 prevNum = 0;
    u64 nextNum = 0;
    for (u32 index = 0; index < gatherPara.addrInfoCountPerRank.size(); index++) {
        nextNum += gatherPara.addrInfoCountPerRank[index];
        for (u64 i = NUM_TWO * prevNum; i < NUM_TWO * nextNum; i += NUM_TWO) {
            *(sendCounts + index) += gatherPara.addrInfo[i + 1];
        }
        prevNum = nextNum;
    }

    ret = memset_s(sdispls, gatherPara.rankSize * sizeof(u64), 0, gatherPara.rankSize * sizeof(u64));
    CHK_PRT_RET(ret != EOK, HCCL_ERROR("[Exec][EnqueueGatherAlltoAllV] mem set failed, count[%lld]",
        gatherPara.rankSize * sizeof(u64)), HCCL_E_SYSCALL);
    u64 displ = 0;
    for (u32 i = 0; i < gatherPara.rankSize; i++) {
        *(sdispls + i) = displ;
        displ += *(sendCounts + i);
    }

    // 等待线程执行完毕
    for (u32 num = 0; num < GATHER_THREAD_NUM; num++) {
        threads[num]->join();
    }

    CHK_RET(hrtMemSyncCopy(sendDevBuf, memSize, tmpHostMem.ptr(), memSize,
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    return HCCL_SUCCESS;
}

/*
 * **********************************************************************
 * 单算子GatherAllToAllV gather多线程拷贝
 * **********************************************************************
 */
void GatherMemCopyThread(void *baseAddr, u64 offset, std::vector<u64> &addrInfo, OpBaseMemPara memCpyPara)
{
    void *addr = nullptr;
    const u32 NUM_TWO = 2;
    u64 length = 0;
    auto destMax = [&]()-> u64 {
        return memCpyPara.tmpMemSize < offset ? 0 : memCpyPara.tmpMemSize - offset;
    };

    for (u32 index = 0; index < memCpyPara.count; index++) {
        addr = reinterpret_cast<void *>(addrInfo[memCpyPara.beginIndex + NUM_TWO * index]);
        length = addrInfo[memCpyPara.beginIndex + index * NUM_TWO + 1];
        if (memcpy_s(static_cast<s8 *>(baseAddr) + offset, destMax(), addr, length) != EOK) {
            HCCL_ERROR("[MemCopy][GatherAlltoAllV] mem copy failed, destMax[%llu], count[%llu]",
                memCpyPara.tmpMemSize - offset, length);
            return;
        }
        offset += length;
    }
}

HcclResult SetDefaultQosConfig(hccl::hcclComm *hcclComm)
{
    u32 qosCfg = INVALID_QOSCFG; // qos不使能的情况下为全F
    CHK_RET(hcclComm->GetQosCfg(qosCfg));
    // 防止Lowering下Qos值被覆盖
    if (qosCfg == INVALID_QOSCFG) {
        CHK_RET(hrtGetQosConfig(HCCL_STREAM_DEFAULT_GROUP_ID, qosCfg));
        HCCL_DEBUG("Call SetDefaultQosConfig, qosCfg[%x]", qosCfg);
        CHK_RET(hcclComm->SetQosCfg(qosCfg));
    }
    return HCCL_SUCCESS;
}

/*
 * **********************************************************************
 * 获取HCCL错误
 * **********************************************************************
 */
HcclResult HcclGetCommAsyncError(HcclComm comm, HcclResult *asyncError)
{
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclCommGetAsyncError", "comm", "nullptr", "please check comm"}));
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(asyncError);

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    CHK_RET(hcclComm->CommCheckErrorCqe(*asyncError));

    return HCCL_SUCCESS;
}

/*
 * **********************************************************************
 * HCCL提供错误码到字符串的转换
 * **********************************************************************
 */
const char *HcclGetErrorString(HcclResult code)
{
    if (code < HcclResult::HCCL_SUCCESS || code >= HcclResult::HCCL_E_RESERVED) {
        return "unknow error";
    }
    static const std::map<HcclResult, std::string> errorMap = {{HCCL_SUCCESS, "no error"},
        {HCCL_E_PARA, "parameter error"}, {HCCL_E_PTR, "empty pointer"},
        {HCCL_E_MEMORY, "memory error"}, {HCCL_E_INTERNAL, "internal error"},
        {HCCL_E_NOT_SUPPORT, "not support feature"}, {HCCL_E_NOT_FOUND, "not found specific resource"},
        {HCCL_E_UNAVAIL, "resource unavailable"}, {HCCL_E_SYSCALL, "call system interface error"},
        {HCCL_E_TIMEOUT, "timeout"}, {HCCL_E_OPEN_FILE_FAILURE, "open file fail"},
        {HCCL_E_TCP_CONNECT, "tcp connect fail"}, {HCCL_E_ROCE_CONNECT, "roce connect fail"},
        {HCCL_E_TCP_TRANSFER, "tcp transfer fail"}, {HCCL_E_ROCE_TRANSFER, "roce transfer fail"},
        {HCCL_E_RUNTIME, "call runtime api fail"}, {HCCL_E_DRV, "call driver api fail"},
        {HCCL_E_PROFILING, "call profiling api fail"}, {HCCL_E_CCE, "call cce api fail"},
        {HCCL_E_NETWORK, "call network api fail"}, {HCCL_E_AGAIN, "try again"},
        {HCCL_E_REMOTE, "error cqe"}};

    return errorMap.at(code).data();
}

/*
 * 配置溢出检测地址
 */
HcclResult SetOverFlowAddr(hccl::hcclComm *hcclComm)
{
    std::vector<void *> globalWorkSpaceAddr;
    CHK_RET(hcclComm->SetGlobalWorkSpace(globalWorkSpaceAddr));
    return HCCL_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
HcclResult HcclCreateComResource(const char *commName, u32 streamMode, void** commContext)
{
    RPT_INPUT_ERR(commName == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclCreateComResource", "commName", "nullptr", "please check commName"}));

    RPT_INPUT_ERR(commContext == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclCreateComResource", "commContext", "nullptr", "please check commContext"}));
    // 切换线程后获取不到hcom上下文，需重新刷新一次线程操作的deviceid
    CHK_RET(hrtGetDeviceRefresh(&g_hcclDeviceId));
    std::shared_ptr<hccl::hcclComm> hcclComm;
    CHK_RET(HcomGetCommByGroup(commName, hcclComm));
    HcclComm comm = hcclComm.get();
    CHK_RET(HcclCreateComResourceByComm(comm, streamMode, true, commContext));
    return HCCL_SUCCESS;
}

HcclResult HcclAllocComResource(HcclComm comm, u32 streamMode, void** commContext)
{
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclCreateComResource", "comm", "nullptr", "please check comm"}));

    RPT_INPUT_ERR(commContext == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclCreateComResource", "commContext", "nullptr", "please check commContext"}));
    // 切换线程后获取不到hcom上下文，需重新刷新一次线程操作的deviceid

    CHK_RET(HcclCreateComResourceByComm(comm, streamMode, true, commContext));
    return HCCL_SUCCESS;
}
#ifdef __cplusplus
}
#endif // __cplusplus

HcclResult HcclCreateComResourceByComm(HcclComm comm, u32 streamMode, bool isOpbaseMode,
    void** commContext)
{
    HcclUs startut = TIME_NOW();
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclCreateComResource", "comm", "nullptr", "please check comm"}));

    RPT_INPUT_ERR(commContext == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclCreateComResource", "commContext", "nullptr", "please check commContext"}));

    // 同通信域同算子复用tag
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    const string tag = "CreatecomResource_" + hcclComm->GetIdentifier();

    char stackLogBuffer[LOG_TMPBUF_SIZE];
    s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U, "tag[%s], commContext[%p]", tag.c_str(),
        commContext);
    CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));

    if (LIKELY(hcclComm->GetCommResource(tag, commContext))) {
        /* 接口交互信息日志 */
        HcclUs endGetResourcetut = TIME_NOW();
        std::string getComReslogInfo = "HcclCreateComResource get ComResource success:take time" +
            std::to_string(DURATION_US(endGetResourcetut - startut).count()) + "us, "
            + std::string(stackLogBuffer);
        CHK_RET(hcclComm->SaveOpbaseKeyTraceInfo(getComReslogInfo));
        return HCCL_SUCCESS;
    }
    const std::lock_guard<std::mutex> lock(hcclComm->operatorlock_);

    u32 localRank = INVALID_VALUE_RANKID;
    CHK_RET(hcclComm->GetUserRank(localRank));

    /* 接口交互信息日志 */
    std::string logInfo = "Entry-HcclCreateComResource:localRank[" + std::to_string(localRank)
        + "]" + std::string(stackLogBuffer);
    CHK_RET(hcclComm->SaveOpbaseKeyTraceInfo(logInfo));
    // SetWorkflowMode性能开销hrtGetDevice，0.11us
    HcclUs middleut0 = TIME_NOW();
    CHK_RET(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));
    HcclUs middleut1 = TIME_NOW();
    rtStream_t stream;
    CHK_RET(hcclComm->Mc2AiCpuStreamAllocAndGet(streamMode, stream));
    CHK_RET(hcclComm->CreateCommResource(tag, stream, isOpbaseMode, commContext));

    HcclUs endut = TIME_NOW();
    /* 关键状态记录 */
    std::string endInfo = "HcclCreateComResource success, HcclCreateComResource take time ["
        + std::to_string(DURATION_US(endut - startut).count()) + "]us, CreateComResource take time ["
        + std::to_string(DURATION_US(endut - middleut1).count()) + "]us, SetWorkflowMode take time ["
        + std::to_string(DURATION_US(middleut1 - middleut0).count()) + "]us, localRank["
        + std::to_string(localRank) + "] " + std::string(stackLogBuffer);
    CHK_RET(hcclComm->SaveOpbaseKeyTraceInfo(endInfo));
    return HCCL_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
HcclResult HcclGetAicpuOpStreamNotify(const char *commName, rtStream_t* opstream, void** aicpuNotify)
{
    RPT_INPUT_ERR(commName == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclGetAicpuOpStream", "commName", "nullptr", "please check commName"}));

    RPT_INPUT_ERR(opstream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclGetAicpuOpStream", "opstream", "nullptr", "please check opstream"}));

    RPT_INPUT_ERR(aicpuNotify == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclGetAicpuOpStream", "aicpuNotify", "nullptr", "please check aicpuNotify"}));
    // 切换线程后获取不到hcom上下文，需重新刷新一次线程操作的deviceid
    CHK_RET(hrtGetDeviceRefresh(&g_hcclDeviceId));
    std::shared_ptr<hccl::hcclComm> hcclComm;
    CHK_RET(HcomGetCommByGroup(commName, hcclComm));

    CHK_RET(hcclComm->GetAicpuOpStreamNotify(opstream, aicpuNotify));
    return HCCL_SUCCESS;
}

HcclResult HcclGetAicpuOpStreamAndNotify(HcclComm comm, rtStream_t* opstream, void** aicpuNotify)
{
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclGetAicpuOpStream", "comm", "nullptr", "please check comm"}));

    RPT_INPUT_ERR(opstream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclGetAicpuOpStream", "opstream", "nullptr", "please check opstream"}));

    RPT_INPUT_ERR(aicpuNotify == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclGetAicpuOpStream", "aicpuNotify", "nullptr", "please check aicpuNotify"}));
    // 切换线程后获取不到hcom上下文，需重新刷新一次线程操作的deviceid
    CHK_RET(hrtGetDeviceRefresh(&g_hcclDeviceId));
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);

    CHK_RET(hcclComm->GetAicpuOpStreamNotify(opstream, aicpuNotify));
    return HCCL_SUCCESS;
}
#ifdef __cplusplus
}
#endif // __cplusplus

HcclResult GetPairWiseList(HcclSendRecvItem *sendRecvInfo, u32 itemNum, u32 rankSize, u32 rankId,
    std::vector<HcclSendRecvItem *> &orderedList)
{
    HCCL_INFO("[BatchSendRecv][GetPairWiseList] Start sort the batchSendRecv tasklist.");
    CHK_PTR_NULL(sendRecvInfo);
    /* 对任务个数进行校验：任务列表中不能有重复的send/recv任务,重复指向（从）同一rank发送（接收）两个任务，
    因此会将任务数限制在2*rankSize. */
    CHK_PRT_RET((itemNum == 0 || itemNum > DOUBLE_SIZE * rankSize),
        HCCL_ERROR("[BatchSendRecv][GetPairWiseList] List itemNum is zero or larger than double rankSize, "\
            "itemNum is %u, rankSize is %u, rankID is %u.", itemNum, rankSize, rankId), HCCL_E_PARA);
    std::vector<HcclSendRecvItem *> sendVec(rankSize, nullptr);
    std::vector<HcclSendRecvItem *> recvVec(rankSize, nullptr);

    for (u32 i = 0; i < itemNum; i++) {
        HCCL_INFO("[BatchSendRecv][GetPairWiseList] index is %u, itemNum is %u, rankID is %u, remoteRank is %u, "\
            "sendRecvType is %u, rankSize is %u.", i, itemNum, rankId, sendRecvInfo->remoteRank,
            static_cast<u32>(sendRecvInfo->sendRecvType), rankSize);
        CHK_PTR_NULL(sendRecvInfo->buf);
        CHK_RET(HcomCheckDataType(sendRecvInfo->dataType));
        CHK_RET(HcomCheckCount(sendRecvInfo->count));
        CHK_RET(HcomCheckUserRank(rankSize, sendRecvInfo->remoteRank));

        if (sendRecvInfo->sendRecvType == HcclSendRecvType::HCCL_SEND) {
            // 若send/recv任务存在重复情况，直接退出
            CHK_PRT_RET((sendVec[sendRecvInfo->remoteRank] != nullptr),
                HCCL_ERROR(
                    "[BatchSendRecv][GetPairWiseList] Send Tasks are duplicated, rankID is %u, remoteRank is %u.",
                    rankId, sendRecvInfo->remoteRank), HCCL_E_PARA);
            sendVec[sendRecvInfo->remoteRank] = sendRecvInfo;
        } else if (sendRecvInfo->sendRecvType == HcclSendRecvType::HCCL_RECV) {
            CHK_PRT_RET((recvVec[sendRecvInfo->remoteRank] != nullptr),
                HCCL_ERROR(
                    "[BatchSendRecv][GetPairWiseList] Recv Tasks are duplicated, rankID is %u, remoteRank is %u.",
                    rankId, sendRecvInfo->remoteRank), HCCL_E_PARA);
            recvVec[sendRecvInfo->remoteRank] = sendRecvInfo;
        } else {
            HCCL_ERROR("[BatchSendRecv][GetPairWiseList] sendRecvType wrong sendrecvType is %d, rankID is %u,"\
                "remoteRank is %u.", sendRecvInfo->sendRecvType, rankId,
                       sendRecvInfo->remoteRank);
            return HCCL_E_PARA;
        }
        sendRecvInfo++;
    }
    /* 此处的排序逻辑:
        1.sendVec和recvVec(当前按remoteRank号有序排列)间隔穿插插入数组orderedList
        2.sendVec元素中放入orderedList的规则是:先放remoteRank号小于root rank的第一个任务，依次减小(循环索引)直至放完
        3.recvVec元素中放入orderedList的规则是:先放remoteRank号大于root rank的第一个任务，依次增大(循环索引)直至放完
    */
    // sendVec的索引
    u32 sendIndex = rankId;
    // recvVec的索引
    u32 recvIndex = rankId;
    // orderedList的索引
    u32 index = 0;
    bool sendFirstFlag = true;
    bool recvFirstFlag = true;
    while (index < itemNum) {
        bool foundSendTask = false;
        while (sendFirstFlag || sendIndex != rankId) {
            sendFirstFlag = false;
            if (sendVec[sendIndex] != nullptr) {
                foundSendTask = true;
                break;
            }
            sendIndex = (sendIndex + rankSize - 1) % rankSize;
        }
        if (foundSendTask) {
            orderedList[index++] = sendVec[sendIndex];
            sendIndex = (sendIndex + rankSize - 1) % rankSize;
        }
        bool foundRecvTask = false;
        while (recvFirstFlag || recvIndex != rankId) {
            recvFirstFlag = false;
            if (recvVec[recvIndex] != nullptr) {
                foundRecvTask = true;
                break;
            }
            recvIndex = (recvIndex + 1) % rankSize;
        }
        if (foundRecvTask) {
            orderedList[index++] = recvVec[recvIndex];
            recvIndex = (recvIndex + 1) % rankSize;
        }
    }
    HCCL_INFO("[BatchSendRecv][GetPairWiseList] End sort the batchSendRecv tasklist.");
    return HCCL_SUCCESS;
}

HcclResult HcclBatchSendRecv(HcclSendRecvItem* sendRecvInfo, uint32_t itemNum, HcclComm comm, aclrtStream stream)
{
    HcclUs startut = TIME_NOW();
    HcclSetIfProfile();
    uint64_t beginTime = hrtMsprofSysCycleTime();
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));

    // 入参校验
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    s32 streamId = 0;
    CHK_RET(hrtGetStreamId(stream, streamId));

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);

    u32 rankSize = 0;
    CHK_RET(hcclComm->GetRankSize(rankSize));

    u32 localRank = INVALID_VALUE_RANKID;
    CHK_RET(hcclComm->GetGroupRank(localRank));
    std::vector<HcclSendRecvItem*> orderedList(itemNum, nullptr);
    CHK_RET(GetPairWiseList(sendRecvInfo, itemNum, rankSize, localRank, orderedList));

    // 若任务不同，也复用tag
    const string tag = "worldBatchSendRecv_" + hcclComm->GetIdentifier();

    /* 记录接口交互信息日志 */
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
        "tag[%s], itemNum[%u], streamId[%d], deviceLogicId[%d]", tag.c_str(), itemNum, streamId, deviceLogicId);

    CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
    std::string logInfo = "Entry-HcclBatchSendRecv:" + std::string(stackLogBuffer);
    CHK_RET(hcclComm->SaveOpbaseKeyTraceInfo(logInfo));

    for (const auto& taskItem : orderedList) {
        char stackLogBuffer[LOG_TMPBUF_SIZE];
        s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "SendRecvItem : SendRecvType[%d], remoteRank[%d]", taskItem->sendRecvType, taskItem->remoteRank);
        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
        std::string logInfo = "[HcclBatchSendRecv]" + std::string(stackLogBuffer);
        CHK_RET(hcclComm->SaveOpbaseKeyTraceInfo(logInfo));
    }

    CHK_RET(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));

    HCCL_PROFILER_ADD_TAG(tag, hcclComm->GetIdentifier(), HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    HCCL_PROFILER_ADD_STREAM(stream, tag, 0, AlgType::ALG_RESERVED);

    HCCL_PROFILER_ADD_OPDATA(tag, 0, nullptr, nullptr, HcclDataType::HCCL_DATA_TYPE_RESERVED, INVALID_VALUE_RANKID, \
        hcclComm->GetIdentifier());
    HCCL_PROFILER_ADD_GROUPRANK(hcclComm->GetIdentifier(), rankSize, localRank);
    u32 loopStartIndex = 0;

    // 如果存在自发自收的任务，GetPairWiseList将自发自收的任务放在orderedList最前面两个。若是send/recv不成对，直接退出。
    if ((itemNum >= RANKSIZE_TWO && orderedList[0]->remoteRank == localRank &&
        orderedList[1]->remoteRank != localRank) ||(itemNum == 1 && orderedList[0]->remoteRank == localRank)) {
        HCCL_ERROR("[HcclBatchSendRecv] Send task and recv task to rank itself do not match,"\
            "please check the task list.");
        return HCCL_E_PARA;
    }
    // 适配自发自收场景，GetPairWiseList可以确保任务不重复，并且send任务在先。
    if (itemNum >= RANKSIZE_TWO && orderedList[0]->remoteRank == localRank && orderedList[1]->remoteRank == localRank) {
        u32 qosCfg = INVALID_QOSCFG; // qos不使能的情况下为全F
        CHK_RET_AND_PRINT_IDE(hcclComm->GetQosCfg(qosCfg), tag.c_str());

        if (orderedList[0]->count == orderedList[1]->count && orderedList[0]->dataType == orderedList[1]->dataType) {
            u64 dataSize = orderedList[0]->count * SIZE_TABLE[orderedList[0]->dataType];
            ret = hrtMemAsyncCopyByQos(orderedList[1]->buf, dataSize, orderedList[0]->buf, dataSize,
                HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE, stream, qosCfg);
            // 若是自发自收则跳过前2个任务
            loopStartIndex += 2;
        } else {
            HCCL_ERROR("[HcclBatchSendRecv] Send task and recv task to self : data size do not equal, please check the"\
                "task list.");
            return HCCL_E_PARA;
        }
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[BatchSendRecv] In OP_BASE send to self transit, memcopy failed."), HCCL_E_MEMORY);
    }

    CHK_RET_AND_PRINT_IDE(SetDefaultQosConfig(hcclComm), tag.c_str());
    CHK_RET_AND_PRINT_IDE(hcclComm->ProcessSendRecvTasks(tag, orderedList, itemNum, loopStartIndex, stream),
                          tag.c_str());

    HCCL_PROFILER_DEL_STREAM(stream);
    HCCL_PROFILER_DEL_TAG(tag);
    HCCL_PROFILER_DEL_OPDATA(tag);
    HCCL_PROFILER_DEL_GROUPRANK(tag);
    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, beginTime, sendRecvInfo->count,
        sendRecvInfo->dataType, tag));
    ResetIfProfile();
    HcclUs endut = TIME_NOW();
    /* 关键状态记录 */
    std::string endInfo = "HcclBatchSendRecv:success,take time: " +
        std::to_string(DURATION_US(endut - startut).count()) + " us, tag: " + tag;
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveOpbaseKeyTraceInfo(endInfo), tag.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclDeviceRefresh(void)
{
    HcclResult ret = hrtGetDeviceRefresh(&g_hcclDeviceId);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Get][DeviceRefresh]errNo[0x%016llx] g_hcclDeviceId[%d]"
        "get device refresh error.", ret, g_hcclDeviceId), ret);
    return HCCL_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
HcclResult HcclGetTopoDesc(HcclComm comm, HcclTopoDescs *topoDescs, uint32_t topoSize)
{
    // 入参合法性校验
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclGetTopoDesc", "comm", "nullptr", "please check comm"}));

    RPT_INPUT_ERR(topoDescs == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclGetTopoDesc", "topoDescs", "nullptr", "please check topoDescs"}));

    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    CHK_RET(hcclComm->GetTopoDesc(topoDescs, topoSize));

    return HCCL_SUCCESS;
}
#ifdef __cplusplus
}
#endif // __cplusplus
