/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topoinfo_detect.h"
#include <string>
#include "externalinput_pub.h"
#include "adapter_rts_common.h"
#include "hccl_whitelist.h"
#include "hccl_socket.h"
using namespace std;
namespace hccl {
const u32 TOPO_EXCHANGE_SERVER_STATUS_IDLE = 0;
const u32 TOPO_EXCHANGE_SERVER_STATUS_RUNING = 1;
const u32 TOPO_EXCHANGE_SERVER_STATUS_ERROR = 2;
map<u32, volatile u32> TopoInfoDetect::topoExchangeServerStatus_;
HcclIpAddress TopoInfoDetect::bootstrapHostIP_;

TopoInfoDetect::TopoInfoDetect() : deviceLogicID_(INVALID_INT), localRankInfo_(), clusterTopoInfo_()
{
}

TopoInfoDetect::~TopoInfoDetect()
{
    if (exchangeServerThreadPtr_ && exchangeServerThreadPtr_->joinable()) {
        exchangeServerThreadPtr_->join();
    }
    exchangeServerThreadPtr_ = nullptr;
    pTopoExchangeServer_ = nullptr;
    (void)Teardown();
    return;
}

HcclResult TopoInfoDetect::GetServerConnections(std::map<std::string, std::shared_ptr<HcclSocket>> &connectSockets)
{
    if (pTopoExchangeServer_) {
        return pTopoExchangeServer_->GetConnections(connectSockets);
    } else {
        return HCCL_SUCCESS;
    }
}

HcclResult TopoInfoDetect::GetAgentConnection(std::shared_ptr<HcclSocket> &connectSocket)
{
    CHK_SMART_PTR_NULL(pTopoExchangeAgent_);
    return pTopoExchangeAgent_->GetConnection(connectSocket);
}

void TopoInfoDetect::SetupTopoExchangeServer(s32 devicePhysicID, s32 deviceLogicID, HcclIpAddress hostIP, u32 hostPort,
    vector<HcclIpAddress> whitelist, HcclNetDevCtx netDevCtx, std::shared_ptr<HcclSocket> listenSocket,
    bool isMasterInfo)
{
    HcclResult ret = hrtSetDevice(deviceLogicID);
    if (ret != HCCL_SUCCESS) {
        topoExchangeServerStatus_[hostPort] = TOPO_EXCHANGE_SERVER_STATUS_ERROR;
        HCCL_ERROR("[Setup][TopoExchangeServer]set device[%d] failed, ret[%u]", deviceLogicID, ret);
        return;
    }

    pTopoExchangeServer_.reset(new (nothrow) TopoInfoExchangeServer(hostIP, hostPort, whitelist, netDevCtx,
        listenSocket, rootInfo_.identifier));
    if (!pTopoExchangeServer_) {
        topoExchangeServerStatus_[hostPort] = TOPO_EXCHANGE_SERVER_STATUS_ERROR;
        HCCL_ERROR("[Setup][TopoExchangeServer]build topoExchangeServer failed. ");
    } else {
        ret = isMasterInfo ? pTopoExchangeServer_->SetupByMasterInfo() : pTopoExchangeServer_->Setup();
        if (ret != HCCL_SUCCESS) {
            topoExchangeServerStatus_[hostPort] = TOPO_EXCHANGE_SERVER_STATUS_ERROR;
            HCCL_ERROR("[Setup][TopoExchangeServer]setup topoExchangeServer failed, ret[%u]", ret);
        }
    }

    ret = hrtResetDevice(deviceLogicID);
    if (ret != HCCL_SUCCESS) {
        topoExchangeServerStatus_[hostPort] = TOPO_EXCHANGE_SERVER_STATUS_ERROR;
        HCCL_ERROR("[Setup][TopoExchangeServer]reset device[%d] failed, ret[%u]", deviceLogicID, ret);
        return;
    }
    topoExchangeServerStatus_[hostPort] = TOPO_EXCHANGE_SERVER_STATUS_IDLE;
}
HcclResult TopoInfoDetect::SetupServerByMasterInfo(const HcclIpAddress& masterIP, u32 masterPort, const HcclRootHandle &rootInfo)
{
    CHK_RET(hrtGetDevice(&deviceLogicID_));
    CHK_RET(hrtGetDevicePhyIdByIndex(deviceLogicID_, devicePhysicID_));
    vector<HcclIpAddress> whitelist;
    if (GetExternalInputHcclEnableWhitelist() == HCCL_WHITELIST_ON) {
        CHK_RET(ReadHostSocketWhitelist(whitelist));
    }
    rootInfo_ = rootInfo;
    CHK_RET(HcclNetInit(NICDeployment::NIC_DEPLOYMENT_HOST, devicePhysicID_, deviceLogicID_, true));
    CHK_RET(StartRootNetwork(whitelist, masterIP, masterPort));
    topoExchangeServerStatus_[GetExternalInputMasterInfo().port] = TOPO_EXCHANGE_SERVER_STATUS_RUNING;

    thread threadHandle(&TopoInfoDetect::SetupTopoExchangeServer, this, devicePhysicID_, deviceLogicID_,
        masterIP, GetExternalInputMasterInfo().port, whitelist, serverPortCtx_, listenSocket_, true);
    threadHandle.detach();

    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::SetupServer(HcclRootHandle &rootInfo)
{
    CHK_RET(hrtGetDevice(&deviceLogicID_));

    vector<HcclIpAddress> whitelist;
    if (GetExternalInputHcclEnableWhitelist() == HCCL_WHITELIST_ON) {
        CHK_RET(ReadHostSocketWhitelist(whitelist));
    }
    HcclIpAddress hostIP = GetBootstrapHostIP();
    CHK_RET(hrtGetDevicePhyIdByIndex(deviceLogicID_, devicePhysicID_));
    HCCL_INFO("[Setup][hcclIfBasePort]deviceLogicID_[%u], devicePhysicID_[%u]", deviceLogicID_, devicePhysicID_);

    // true代表感知白名单disable配置
    CHK_RET(HcclNetInit(NICDeployment::NIC_DEPLOYMENT_HOST, devicePhysicID_, deviceLogicID_, true));

    CHK_RET(GetRootHostIP(whitelist, hostIP, devicePhysicID_));
    SetBootstrapHostIP(hostIP);

    s32 deviceNum = 0;
    CHK_RET(hrtGetDeviceCount(&deviceNum));
    CHK_PRT_RET((deviceLogicID_ >= deviceNum),
        HCCL_ERROR("[Setup][Server]deviceLogicID[%d] is invalid,deviceNum[%d].", deviceLogicID_, deviceNum),
        HCCL_E_PARA);
    
    u32 hostPort = HCCL_INVALIED_IF_BASE_PORT;
    if (GetExternalInputHcclIfBasePort() == HCCL_INVALIED_IF_BASE_PORT) {
        hostPort = HOST_CONTROL_BASE_PORT + devicePhysicID_;
    } else {
        hostPort = GetExternalInputHcclIfBasePort() + devicePhysicID_;
    }
    CHK_RET(GenerateRootInfo(hostIP, hostPort, devicePhysicID_, rootInfo_));
    CHK_RET(StartRootNetwork(whitelist, hostIP, hostPort));
    topoExchangeServerStatus_[hostPort] = TOPO_EXCHANGE_SERVER_STATUS_RUNING;
    exchangeServerThreadPtr_.reset(new (nothrow) thread(&TopoInfoDetect::SetupTopoExchangeServer, this, devicePhysicID_,
        deviceLogicID_, hostIP, hostPort, whitelist, serverPortCtx_, listenSocket_, false));
    CHK_SMART_PTR_NULL(exchangeServerThreadPtr_);

    rootInfo = rootInfo_;
    HCCL_INFO("setup topo exchange server complete, identifier[%s]", rootInfo.identifier);
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::GenerateRootInfo(const HcclIpAddress &hostIP, u32 hostPort, u32 devicePhysicID, HcclRootHandle &rootInfo)
{
    u64 timestamp = 0;
    CHK_RET(SalGetCurrentTimestamp(timestamp));

    string identifier = hostIP.GetReadableAddress();
    identifier.append("_");
    identifier.append(to_string(hostPort));
    identifier.append("_");
    identifier.append(to_string(devicePhysicID));
    identifier.append("_");
    identifier.append(to_string(timestamp));
    CHK_PRT_RET((identifier.length() >= ROOTINFO_INDENTIFIER_MAX_LENGTH),
        HCCL_ERROR("[Setup][Server]rootinfo identifier len[%u] is invalid.", identifier.length()), HCCL_E_INTERNAL);
    s32 sret = memcpy_s(&rootInfo.identifier[0], sizeof(rootInfo.identifier), identifier.c_str(),
        (identifier.length() + 1));
    CHK_PRT_RET(sret != EOK, HCCL_ERROR("[Setup][Server]errNo[0x%016llx] memcpy failed. ret[%d], params:"\
        "destMaxSize[%zu],count[%zu]", HCOM_ERROR_CODE(HCCL_E_MEMORY), sret, sizeof(rootInfo.identifier),
        (identifier.length() + 1)), HCCL_E_MEMORY);
    s32 sRet = strncpy_s(rootInfo.ip, sizeof(rootInfo.ip), hostIP.GetReadableIP(), strlen(hostIP.GetReadableIP()));
    CHK_PRT_RET(sRet != EOK, HCCL_ERROR("[Setup][Server]str copy fail. return[%d]", sRet), HCCL_E_INTERNAL);
    rootInfo.port = hostPort;
    rootInfo.nicDeploy = (GetExternalInputHcclDeviceNicDisable()) ?
        NICDeployment::NIC_DEPLOYMENT_HOST : NICDeployment::NIC_DEPLOYMENT_DEVICE;

    HCCL_INFO("rootInfo: ip[%s] port[%u] identifier[%s]", rootInfo.ip, rootInfo.port, rootInfo.identifier);
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::TeardownServer()
{
    if(pTopoExchangeServer_) {
        CHK_RET(pTopoExchangeServer_->Teardown());
    }

    if (serverPortCtx_) {
        HcclNetCloseDev(serverPortCtx_);
        serverPortCtx_ = nullptr;
        CHK_RET(HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_HOST, devicePhysicID_, deviceLogicID_));
    }
    HCCL_INFO("TopoInfoDetect TeardownServer ok, identifier[%s].", rootInfo_.identifier);
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::WaitTopoExchangeServerCompelte(u32 idx) const
{
    const auto start = chrono::steady_clock::now();
    const auto timeout = chrono::seconds(GetExternalInputHcclLinkTimeOut());
    auto iter = topoExchangeServerStatus_.find(idx);
    if (iter == topoExchangeServerStatus_.end()) {
        return HCCL_SUCCESS;
    }
    while (true) {
        if (topoExchangeServerStatus_[idx] == TOPO_EXCHANGE_SERVER_STATUS_ERROR) {
            HCCL_ERROR("[Wait][TopoExchangeServerCompelte]topo detect failed. topoExchangeServer port[%u] failed.",
                idx);
            return HCCL_E_INTERNAL;
        } else if (topoExchangeServerStatus_[idx] == TOPO_EXCHANGE_SERVER_STATUS_IDLE) {
            HCCL_INFO("topoExchangeServer[%u] compeleted.", idx);
            return HCCL_SUCCESS;
        } else {
            const auto elapsed =
                chrono::duration_cast<chrono::seconds>(chrono::steady_clock::now() - start);
            if (elapsed > timeout) {
                HCCL_ERROR("[Wait][TopoExchangeServerCompelte]wait topoExchangeServer[%u] complete timeout[%lld]",
                    idx, elapsed);
                return HCCL_E_INTERNAL;
            }
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
            continue;
        }
    };
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::SetupAgent(u32 rankSize, u32 myrank, const HcclRootHandle &rootInfo)
{
    CHK_PRT_RET((rootInfo.nicDeploy == NICDeployment::NIC_DEPLOYMENT_DEVICE &&
        GetExternalInputHcclDeviceNicDisable()) ||
        (rootInfo.nicDeploy == NICDeployment::NIC_DEPLOYMENT_HOST &&
        !GetExternalInputHcclDeviceNicDisable()),
        HCCL_ERROR("[Setup][Agent]hcclDeviceNicDisable is [%u] when "\
            "nicDeploy form root is [%u]", rootInfo.nicDeploy,
            GetExternalInputHcclDeviceNicDisable()), HCCL_E_PARA);
    CHK_RET(hrtGetDevice(&deviceLogicID_));

    HcclIpAddress rootIP(rootInfo.ip);
    CHK_PRT_RET(rootIP.IsInvalid(), HCCL_ERROR("string[%s] is invalid ip", rootInfo.ip), HCCL_E_PARA);
    CHK_RET(hrtGetDevicePhyIdByIndex(deviceLogicID_, devicePhysicID_));

    CHK_RET(HcclNetInit(NICDeployment::NIC_DEPLOYMENT_HOST, devicePhysicID_, deviceLogicID_, true));

    HcclIpAddress hostIP = GetBootstrapHostIP();
    CHK_RET(GetLocalHostIP(hostIP, devicePhysicID_));

    SetBootstrapHostIP(hostIP);

    bool bInitDevNic = rankSize != 1 ? true : false;
    HcclResult ret = StartNetwork(hostIP, bInitDevNic);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Setup][Agent]topo detect agent start network failed! rank[%u]", myrank), ret);

    ret = GenerateLocalRankInfo(rankSize, myrank, localRankInfo_);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Setup][Agent]topo detect generate local rank info failed! rank[%u]", myrank), ret);

    /* 首节点日志，建链失败属常见问题，在建链前记录相关信息 */
    HCCL_RUN_INFO("[HCCL_TRACE]SetupAgent rankNum[%u], rank[%u], rootInfo identifier[%s], server[%s], "
        "logicDevId[%d], phydevId[%d], deviceIp[%s]", rankSize, myrank, rootInfo.identifier,
        localRankInfo_.hostIP.GetReadableAddress(), localRankInfo_.deviceLogicID,
        localRankInfo_.devicePhysicID, localRankInfo_.deviceIP[0].GetReadableIP()) ;

    pTopoExchangeAgent_.reset(new (nothrow) TopoInfoExchangeAgent(rootIP, rootInfo.port,
        rootInfo.identifier, agentPortCtx_, localRankInfo_));
    CHK_SMART_PTR_NULL(pTopoExchangeAgent_);
    CHK_RET(pTopoExchangeAgent_->Setup());
    CHK_RET(pTopoExchangeAgent_->GetClusterTopoInfo(clusterTopoInfo_));
    rootInfo_ = rootInfo;
    HCCL_INFO("topo detect completed. myrank[%u], totalranks[%u], myhost[%s], totalservers[%u].",
        myrank, rankSize, localRankInfo_.hostIP.GetReadableAddress(), clusterTopoInfo_.serverNum);
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::TeardownAgent()
{
    if (!pTopoExchangeAgent_) {
        return HCCL_SUCCESS;
    }
    CHK_RET(pTopoExchangeAgent_->Teardown());

    bool bInitDevNic = clusterTopoInfo_.rankNum != 1 ? true : false;
    HcclIpAddress hostIP = GetBootstrapHostIP();

    auto ret = StopNetwork(hostIP, bInitDevNic);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Setup][Agent]topo detect agent stop network failed!"), ret);
    HCCL_INFO("TopoInfoDetect TeardownAgent ok, identifier[%s].", rootInfo_.identifier);
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::SetupAgentByMasterInfo(HcclIpAddress &localHostIp, const HcclRootHandle &rootInfo)
{
    CHK_RET(hrtGetDevice(&deviceLogicID_));
    SetBootstrapHostIP(localHostIp);
    CHK_RET(hrtGetDevicePhyIdByIndex(deviceLogicID_, devicePhysicID_));
    rootInfo_ = rootInfo;
    bool bInitDevNic = GetExternalInputMasterInfo().rankSize != 1 ? true : false;
    HcclResult ret = StartNetwork(localHostIp, bInitDevNic);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Setup][Agent]topo detect agent start network failed!"), ret);

    bool errorFlag = false;
    unique_ptr<hccl::TopoInfoExchangeAgent> pTopoExchangeAgent;
    do {
        HcclIpAddress rootIP(rootInfo.ip);
        CHK_PRT_BREAK(rootIP.IsInvalid(), HCCL_ERROR("[Setup][Agent]string[%s] is invalid ip", rootInfo.ip),
            errorFlag = true);
        ret = GenerateLocalRankInfo(GetExternalInputMasterInfo().rankSize, INVALID_VALUE_RANKID, localRankInfo_);
        CHK_PRT_BREAK(ret != HCCL_SUCCESS, HCCL_ERROR("[Setup][Agent]topo detect generate local rank info failed"),
            errorFlag = true);

        pTopoExchangeAgent.reset(new (nothrow) TopoInfoExchangeAgent(rootIP, rootInfo.port,
            rootInfo.identifier, agentPortCtx_, localRankInfo_));
        if (pTopoExchangeAgent == nullptr) {
            HCCL_ERROR("[Setup][Agent]pTopoExchangeAgent is nullptr");
            errorFlag = true;
            ret = HCCL_E_PTR;
            break;
        }

        ret = pTopoExchangeAgent->SetupByMasterInfo();
        CHK_PRT_BREAK(ret != HCCL_SUCCESS, HCCL_ERROR("[Setup][Agent]setup by masterInfo failed"),
            errorFlag = true);

        ret = pTopoExchangeAgent->Teardown();
        CHK_PRT_BREAK(ret != HCCL_SUCCESS, HCCL_ERROR("[Setup][Agent]Teardown failed"),
            errorFlag = true);
    } while (0);

    // 如果StartNetwork后执行有报错，则先StopNetwork，再返回
    HcclResult result = StopNetwork(localHostIp, bInitDevNic);
    CHK_PRT_RET(result != HCCL_SUCCESS, HCCL_ERROR("[Setup][Agent]topo detect agent stop network failed!"), result);

    if (errorFlag) {
        HCCL_ERROR("[Setup][Agent]topo detect agent failed, return[%d]", ret);
        return ret;
    }

    CHK_RET(pTopoExchangeAgent->GetClusterTopoInfo(clusterTopoInfo_));
    CHK_RET(pTopoExchangeAgent->GetIdentifier(identifierNum_));

    HCCL_INFO("topo detect completed. deviceLogicID[%u] totalranks[%u], myhost[%s], totalservers[%u].",
        deviceLogicID_, GetExternalInputMasterInfo().rankSize, localRankInfo_.hostIP.GetReadableAddress(),
        clusterTopoInfo_.serverNum);
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::WaitComplete(const HcclRootHandle &rootInfo)
{
    return WaitTopoExchangeServerCompelte(rootInfo.port);
}

HcclResult TopoInfoDetect::Teardown()
{
    CHK_RET(TeardownAgent());
    CHK_RET(TeardownServer());
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::ReadHostSocketWhitelist(vector<HcclIpAddress> &whitelist) const
{
    RPT_ENV_ERR((GetExternalInputHcclWhiteListFile().length() == 0), "EI0001",
        vector<string>({ "env", "tips" }),
        vector<string>({ "HCCL_WHITELIST_FILE", "HCCL_WHITELIST_DISABLE is [0]"
        "but HCCL_WHITELIST_FILE is not set" }));

    CHK_PRT_RET((GetExternalInputHcclWhiteListFile().length() == 0),
        HCCL_ERROR("[Read][HostSocketWhitelist]environmental variable HCCL_WHITELIST_DISABLE is [0], "\
        "but HCCL_WHITELIST_FILE is not set"), HCCL_E_PARA);

    // 文件路径在处理外部输入时已经做过合法性判断, 无需再次校验
    HcclResult ret =
        HcclWhitelist::GetInstance().LoadConfigFile(GetExternalInputHcclWhiteListFile());
    
    std::string WhiteFileError =
        "hccl whitelist load config file[" + GetExternalInputHcclWhiteListFile() + "] failed.";

    RPT_ENV_ERR(ret != HCCL_SUCCESS, "EI0001",
        std::vector<std::string>({ "env", "tips" }),
        std::vector<std::string>({ "HCCL_WHITELIST_FILE", WhiteFileError}));
          
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Read][HostSocketWhitelist]hccl whitelist load config file[%s] failed. ret[%u].",
            GetExternalInputHcclWhiteListFile().c_str(), ret), ret);
    CHK_RET(HcclWhitelist::GetInstance().GetHostWhiteList(whitelist));

    CHK_PRT_RET(whitelist.empty(), HCCL_ERROR("[Read][HostSocketWhitelist]whitelist file[%s] have no valid host ip.",
        GetExternalInputHcclWhiteListFile().c_str()), HCCL_E_UNAVAIL);
    HCCL_INFO("get host socket whitelist success. there are %zu host ip in the whitelist.", whitelist.size());
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::GetAllHostIfInfos(vector<pair<string, HcclIpAddress>> &ifInfos, u32 devPhyId) const
{
    CHK_RET(hrtGetHostIf(ifInfos, devPhyId));

    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::GetAllValidHostIfInfos(const vector<HcclIpAddress> &whitelist,
    vector<pair<string, HcclIpAddress>> &ifInfos, u32 devPhyId)
{
    vector<pair<string, HcclIpAddress>> orginIfInfos;
    CHK_RET(GetAllHostIfInfos(orginIfInfos, devPhyId));

    for (auto &ifInfo : orginIfInfos) {
        auto iter = find(whitelist.begin(), whitelist.end(), ifInfo.second);
        if (iter != whitelist.end()) {
            ifInfos.push_back({ ifInfo.first, ifInfo.second });
        }
    }

    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::GetRootHostIP(const vector<HcclIpAddress> &whitelist, HcclIpAddress &ip, u32 devPhyId)
{
    if (!ip.IsInvalid()) {
        return HCCL_SUCCESS;
    }
    vector<pair<string, HcclIpAddress>> ifInfos;

    if (GetExternalInputHcclEnableWhitelist() == HCCL_WHITELIST_ON) {
        CHK_RET(GetAllValidHostIfInfos(whitelist, ifInfos, devPhyId));
        CHK_PRT_RET(ifInfos.empty(), HCCL_ERROR("[Get][RootHostIP]there is no valid host if in whitelist."),
            HCCL_E_NOT_FOUND);
    } else {
        CHK_RET(GetAllHostIfInfos(ifInfos, devPhyId));
        CHK_PRT_RET(ifInfos.empty(), HCCL_ERROR("[Get][RootHostIP]there is no host if."), HCCL_E_NOT_FOUND);
    }

    CHK_RET(FindLocalHostIP(ifInfos, ip));
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::StartRootNetwork(const vector<HcclIpAddress> &whitelist, const HcclIpAddress& hostIP,
    u32 usePort)
{
    CHK_RET(HcclNetOpenDev(&serverPortCtx_, NicType::HOST_NIC_TYPE, devicePhysicID_, deviceLogicID_, hostIP));
    CHK_PTR_NULL(serverPortCtx_);

    listenSocket_.reset(new (nothrow) HcclSocket(serverPortCtx_, usePort));
    CHK_SMART_PTR_NULL(listenSocket_);
    CHK_RET(listenSocket_->Init());

    CHK_RET(listenSocket_->Listen());

    HCCL_INFO("topo info exchange server start with host ip[%s] and port[%u]", hostIP.GetReadableAddress(), usePort);

    if (GetExternalInputHcclEnableWhitelist() == HCCL_WHITELIST_ON) {
        CHK_RET(AddSocketWhiteList(usePort, whitelist));
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::AddSocketWhiteList(u32 port,
    const vector<HcclIpAddress> &whitelist) const
{
    vector<SocketWlistInfo> wlistInfosVec;
    for (auto ip : whitelist) {
        SocketWlistInfo wlistInfo;
        wlistInfo.connLimit = HOST_SOCKET_CONN_LIMIT;
        wlistInfo.remoteIp.addr = ip.GetBinaryAddress().addr;
        wlistInfo.remoteIp.addr6 = ip.GetBinaryAddress().addr6;
        string tag = TOPO_DETECT_TAG + "_" + rootInfo_.identifier + "_" + to_string(port);
        s32 sRet = memcpy_s(&wlistInfo.tag[0], sizeof(wlistInfo.tag), tag.c_str(), tag.size() + 1);
        if (sRet != EOK) {
            HCCL_ERROR("[Add][SocketWhiteList]memory copy failed. errorno[%d]", sRet);
            return HCCL_E_MEMORY;
        }
        wlistInfosVec.push_back(wlistInfo);
    }

    CHK_RET(listenSocket_->AddWhiteList(wlistInfosVec));

    HCCL_INFO("add socket white list success. total: %zu", whitelist.size());
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::StartNetwork(HcclIpAddress &hostIP, bool bInitDevNic)
{
    CHK_RET(HcclNetOpenDev(&agentPortCtx_, NicType::HOST_NIC_TYPE, devicePhysicID_, deviceLogicID_, hostIP));
    CHK_PTR_NULL(agentPortCtx_);

    if (!GetExternalInputHcclDeviceNicDisable() && bInitDevNic) {
        CHK_RET(HcclNetInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, devicePhysicID_, deviceLogicID_, false));
        CHK_RET(
            HcclNetOpenDev(&devNicCtx_, NicType::DEVICE_NIC_TYPE, devicePhysicID_, deviceLogicID_, HcclIpAddress(0)));
        CHK_PTR_NULL(devNicCtx_);
    }

    HCCL_INFO("NetworkManager start host net success! ip[%s]", hostIP.GetReadableAddress());

    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::StopNetwork(HcclIpAddress &hostIP, bool bInitDevNic)
{
    if (agentPortCtx_) {
        HcclNetCloseDev(agentPortCtx_);
        agentPortCtx_ = nullptr;
        CHK_RET(HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_HOST, devicePhysicID_, deviceLogicID_));
    }

    if (!GetExternalInputHcclDeviceNicDisable() && bInitDevNic) {
        if (devNicCtx_) {
            HcclNetCloseDev(devNicCtx_);
            devNicCtx_ = nullptr;
            CHK_RET(HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, devicePhysicID_, deviceLogicID_));
        }
    }

    HCCL_INFO("NetworkManager stop host net success! ip[%s] ", hostIP.GetReadableAddress());
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::GenerateLocalRankInfo(u32 rankSize, u32 rankID, HcclBasicRankInfo &localRankInfo)
{
    localRankInfo.hostIP = GetBootstrapHostIP();
    localRankInfo.rank = rankID;
    localRankInfo.rankSize = rankSize;
    localRankInfo.nicDeploy = (GetExternalInputHcclDeviceNicDisable()) ?
        NICDeployment::NIC_DEPLOYMENT_HOST : NICDeployment::NIC_DEPLOYMENT_DEVICE;

    CHK_RET(hrtGetDeviceType(localRankInfo.deviceType));
    CHK_RET(hrtGetDevice(reinterpret_cast<s32 *>(&localRankInfo.deviceLogicID)));
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(localRankInfo.deviceLogicID), localRankInfo.devicePhysicID));

    if (localRankInfo.deviceType == DevType::DEV_TYPE_910_73) {
        s64 superPodId = 0;
        s64 superDeviceId = 0;
        CHK_RET(hrtGetDeviceInfo(localRankInfo.deviceLogicID, HcclRtDeviceModuleType::HCCL_RT_MODULE_TYPE_SYSTEM,
            HcclRtDeviceInfoType::HCCL_INFO_TYPE_SUPER_POD_ID, superPodId));
        CHK_RET(hrtGetDeviceInfo(localRankInfo.deviceLogicID, HcclRtDeviceModuleType::HCCL_RT_MODULE_TYPE_SYSTEM,
            HcclRtDeviceInfoType::HCCL_INFO_TYPE_SDID, superDeviceId));
        localRankInfo.superPodId = std::to_string(superPodId);
        localRankInfo.superDeviceId = static_cast<u32>(superDeviceId);
        HCCL_INFO("[Generate][LocalRankInfo]deviceLogicID[%d], superPodId[%s], superDeviceId[%u]",
            localRankInfo.deviceLogicID, localRankInfo.superPodId.c_str(), localRankInfo.superDeviceId);
    }
    
    localRankInfo.deviceIP.clear();
    if (localRankInfo.nicDeploy == NICDeployment::NIC_DEPLOYMENT_DEVICE && rankSize != 1) {
        std::vector<HcclIpAddress> deviceIPs;
        std::vector<HcclIpAddress> deviceIPv4;
        std::vector<HcclIpAddress> deviceIPv6;
        CHK_RET(hrtRaGetDeviceIP(localRankInfo.devicePhysicID, deviceIPs));
        for (auto &iter : deviceIPs) {
            if (iter.IsIPv6()) {
                deviceIPv6.push_back(iter);
            } else {
                deviceIPv4.push_back(iter);
            }
        }
        // 同时存在ipv4/ipv6时，除非指定socket family，否则ipv4优先
        // 只存在ipv4/ipv6单栈时，不受用户指定的socket family约束
        if ((((GetExternalInputHcclSocketFamily() == -1) ||
            (GetExternalInputHcclSocketFamily() == AF_INET)) &&
            (!deviceIPv4.empty())) || deviceIPv6.empty()) {
            localRankInfo.deviceIP = deviceIPv4;
            HCCL_RUN_INFO("select AF_INET family as device socket family.");
        } else if (!deviceIPv6.empty()) {
            std::sort(deviceIPv6.begin(), deviceIPv6.end());
            localRankInfo.deviceIP.push_back(deviceIPv6[0]);
            HCCL_RUN_INFO("select AF_INET6 family as device socket family.");
        }
    }
    if (localRankInfo.deviceIP.size() == 0) {
        // 和 rank table 保持一致，如果没有device网卡时，默认填充 0。
        HcclIpAddress invalidAddr;
        localRankInfo.deviceIP.push_back(invalidAddr);
        HCCL_RUN_INFO("no device ip: use 0 as device ip.");
    }
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::GetCluterInfo(RankTable_t &clusterInfo)
{
    CHK_PRT_RET((clusterTopoInfo_.rankList.size() == 0),
        HCCL_ERROR("[Get][CluterInfo]GetCluterInfo failed, topo detect has not started."), HCCL_E_INTERNAL);
    clusterInfo = clusterTopoInfo_;
    return HCCL_SUCCESS;
}
HcclResult TopoInfoDetect::GetRankId(u32 &rankId)
{
    rankId = identifierNum_;
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::GetLocalRankInfo(HcclBasicRankInfo &rankInfo)
{
    CHK_PRT_RET((localRankInfo_.rankSize == 0), HCCL_ERROR("[Get][LocalRankInfo]GetLocalRankInfo failed, topo "\
        "detect has not started."), HCCL_E_INTERNAL);
    rankInfo = localRankInfo_;
    return HCCL_SUCCESS;
}

void TopoInfoDetect::SetBootstrapHostIP(HcclIpAddress& ip) const
{
    bootstrapHostIP_ = ip;
}

HcclIpAddress TopoInfoDetect::GetBootstrapHostIP() const
{
    return bootstrapHostIP_;
}
HcclResult TopoInfoDetect::TransformRankTableStr(const RankTable_t &clusterInfo, string &ranktableStr)
{
    nlohmann::json basicJson;
    HcclResult ret = Struct2JsonRankTable(clusterInfo, basicJson);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_WARNING("cluster info to json failed ,ret[%d]", ret), HCCL_E_INTERNAL);
    ranktableStr = basicJson.dump(2); // dump参数为2
    return HCCL_SUCCESS;
}
HcclResult TopoInfoDetect::TransformDeviceList(const RankTable_t &clusterInfo,
    vector<RankInfo_t> &tmpRankList, nlohmann::json &perServerJson, u32 serverIndex)
{
    for (auto it = tmpRankList.begin(); it != tmpRankList.end();) {
        if (it->serverId == clusterInfo.serverList[serverIndex].serverId) {
            nlohmann::json perDeviceJson;
            perDeviceJson[PROP_DEV_ID] = to_string(it->deviceInfo.devicePhyId);
            perDeviceJson[PROP_RANK_ID] = to_string(it->rankId);
            perDeviceJson[PROP_SUPER_DEVICE_ID] = to_string(it->superDeviceId);
            if (clusterInfo.nicDeploy == NICDeployment::NIC_DEPLOYMENT_DEVICE && it->deviceInfo.deviceIp.size() != 0 &&
                !it->deviceInfo.deviceIp[0].IsInvalid()) {
                perDeviceJson[PROP_DEV_IP] = std::string(it->deviceInfo.deviceIp[0].GetReadableIP());
            } else if (clusterInfo.nicDeploy == NICDeployment::NIC_DEPLOYMENT_HOST && !it->hostIp.IsInvalid()) {
                perServerJson[PROP_HOST_IP] = std::string(it->hostIp.GetReadableIP());
            }
            perServerJson[PROP_DEVICE].push_back(perDeviceJson);
            it = tmpRankList.erase(it);
        } else {
            it++;
        }
    }
    return HCCL_SUCCESS;
}
HcclResult TopoInfoDetect::Struct2JsonRankTable(const RankTable_t &clusterInfo, nlohmann::json& ClusterJson)
{
    nlohmann::json serverListJson;
    ClusterJson[PROP_SERVER_COUNT] = to_string(clusterInfo.serverNum);
    vector<RankInfo_t> tmpRankList = clusterInfo.rankList;
    ClusterJson[PROP_SERVER_LIST] = serverListJson;
    for (u32 i = 0; i < clusterInfo.serverNum; i++) {
        nlohmann::json perServerJson;
        perServerJson[PROP_SERVER_ID] = clusterInfo.serverList[i].serverId;
        nlohmann::json deviceList;
        perServerJson[PROP_DEVICE] = deviceList;
        CHK_RET(TransformDeviceList(clusterInfo, tmpRankList, perServerJson, i));
        ClusterJson[PROP_SERVER_LIST].push_back(perServerJson);
    }

    nlohmann::json superPodListJson;
    CHK_RET(TransformSuperPodList(clusterInfo.rankList, superPodListJson));
    ClusterJson[PROP_SUPER_POD_LIST] = superPodListJson;

    ClusterJson[PROP_STATUS] = "completed";
    ClusterJson[PROP_VERSION] = (localRankInfo_.deviceType == DevType::DEV_TYPE_910_73) ? "1.2" : "1.0";
    return HCCL_SUCCESS;
}

HcclResult TopoInfoDetect::TransformSuperPodList(const std::vector<RankInfo_t> &rankInfo,
    nlohmann::json &superPodListJson) const
{
    // 按照 <super_pod_id, <server_id>> 格式从RankInfo_t中解析super pod信息
    std::map<std::string, std::set<std::string>> superPodMap;
    for (u32 i = 0; i < rankInfo.size(); i++) {
        auto iter = superPodMap.find(rankInfo[i].superPodId);
        if (iter == superPodMap.end()) {
            std::set<std::string> perSuperPod;
            perSuperPod.insert(rankInfo[i].serverId);
            superPodMap.insert(std::pair<std::string, std::set<string>>(rankInfo[i].superPodId, perSuperPod));
        } else {
            // superDeviceId在VerifyClusterSuperPodInfo中已经查重校验过
            iter->second.insert(rankInfo[i].serverId);
        }
    }

    for (auto it = superPodMap.begin(); it != superPodMap.end(); ++it) {
        nlohmann::json superPodIdJson;
        superPodIdJson[PROP_SUPER_POD_ID] = it->first;
        nlohmann::json serverListJson;
        for (auto perServer = it->second.begin(); perServer != it->second.end(); ++perServer) {
            nlohmann::json perServerJson;
            perServerJson[PROP_SERVER_ID] = *perServer;
            serverListJson.push_back(perServerJson);
        }
        superPodIdJson[PROP_SERVER_LIST] = serverListJson;
        superPodListJson.push_back(superPodIdJson);
    }
    return HCCL_SUCCESS;
}
}  // namespace hccl
