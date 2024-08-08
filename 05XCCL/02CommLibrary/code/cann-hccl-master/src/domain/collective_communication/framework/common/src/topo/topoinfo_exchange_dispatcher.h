/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOPOINFO_EXCHANGE_DISPATCHER_H
#define TOPOINFO_EXCHANGE_DISPATCHER_H
 
#include <map>
#include <atomic>
#include <vector>
#include <mutex>
#include <thread>
#include <climits>
#include <condition_variable>

#include "adapter_hccp_common.h"
#include "externalinput_pub.h"
#include "hccl_socket.h"
#include "hccl_common.h"
#include "topoinfo_exchange_server.h"
#include "../json_utils.h"

namespace hccl {
class TopoInfoExchangeDispather {
    // avoid the struct name pollution hccl namespace, so use the struct in class
public:
    struct SendState {
        u32 header;
        u32 indentify          = UINT_MAX;    // 默认UINT_MAX时，不发送indentify
        size_t headerLen       = sizeof(u32); // the header need to send
        size_t headerSended    = 0;           // the header have sended length
        size_t bodyLen         = 0;           // the whole data length
        size_t bodySended      = 0;           // the data have sended
        size_t indentifyLen    = sizeof(u32); // the indentify need to send (MasterInfo mode)
        size_t indentifySended = 0;           // the indentify have sended
        void *data;                           // data pointer
        bool firstSendFlag_ =  true;
 
        HcclResult Send(std::shared_ptr<HcclSocket> socket);
        HcclResult SendHeader(std::shared_ptr<HcclSocket> socket);
        HcclResult SendBody(std::shared_ptr<HcclSocket> socket);
        HcclResult SendIndentify(std::shared_ptr<HcclSocket> socket);
        HcclResult SendHelper(std::shared_ptr<HcclSocket> socket, char *buf, size_t dataLen, size_t &sendedLen);
        bool IsOk()
        {
            return bodyLen != 0 && headerSended == headerLen && bodySended == bodyLen;
        }
    };

    struct FdContext {
        std::shared_ptr<HcclSocket> socket;
        SendState txState;
    };

    using WorkerTask = std::function<HcclResult(void)>;
 
public:
    static constexpr u32 DEFAULT_THREAD_NUM = 1;
    static constexpr u32 MAX_THREAD_NUM = 4;
    static constexpr s32 INVALID_EPOLL_EVENT_FD = -1;
    static constexpr s32 EPOLL_TIMEOUT_MS = 100; // 100ms
    static constexpr s32 LAST_EPOLL_TIMEOUT_MS = 5; // 5ms
    static constexpr s32 RANK_CAPACITY_PER_THREAD = 512;
 
    explicit TopoInfoExchangeDispather(TopoInfoExchangeServer *topoInfoExchangeServer,
        u32 threadNum = DEFAULT_THREAD_NUM)
        : topoInfoExchangeServer_(topoInfoExchangeServer), threadNum_(threadNum)
    {
    }
    ~TopoInfoExchangeDispather();
 
    HcclResult BroadcastRankTable(const std::map<std::string, std::shared_ptr<HcclSocket>> connectSockets,
        const RankTable_t &clusterInfo);
 
private:
    void InitWorkerThread();
    void WorkerWait(int workId);
    void WakeWoker();
    void RunWorkerThread(int workId);
    bool GetTask(WorkerTask &workTask);
    HcclResult PrepareResource(const std::map<std::string, std::shared_ptr<HcclSocket>> connectSockets,
        const RankTable_t &clusterInfo);
    HcclResult SendOnce();
    HcclResult ProcessOneSendEvent(int epollFd, FdHandle &fdHandle);
    HcclResult ProcessSend();
    void CleanResource();
    HcclResult CloseEpollFd();

    TopoInfoExchangeServer *topoInfoExchangeServer_;
    u32 threadNum_ = 1;
    u32 rankNum_   = 0;
    std::vector<std::thread> workerThreads_;
    std::queue<WorkerTask> taskQueue_;
    std::mutex taskQueueMutex_;

    std::unordered_map<FdHandle, FdContext> fdHandleToFdContextMap_;
    std::mutex fdHandleMapMutex_;
    s32 epollFds_ = INVALID_EPOLL_EVENT_FD;
    std::atomic<u32> sendDoneCount_{0};

    std::string rankTableJson_;

    std::mutex wakeMutex_;
    std::atomic<bool> ready_{false};
    std::atomic<bool> stop_{false};
    std::condition_variable wakeManager_;
};
} // namespace hccl

#endif /* TOPOINFO_EXCHANGE_DISPATCHER_H */