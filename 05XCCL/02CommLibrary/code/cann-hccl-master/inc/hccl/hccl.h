/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_H_
#define HCCL_H_

#include <hccl/hccl_types.h>
#include <acl/acl.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * @brief Initialize HCCL.
 *
 * @param clusterInfo A string identifying the cluster info file path, include file name.
 * @param rank A integer identifying the identify for the rank.
 * @param comm A pointer identifying the initialized communication resource.
 * @return HcclResult
 * @see HcclCommDestroy()
 */
extern HcclResult HcclCommInitClusterInfo(const char *clusterInfo, uint32_t rank, HcclComm *comm);

/**
 * @brief Get hccl root info.
 *
 * @param rootInfo A pointer identifying the hccl root info.
 * @return HcclResult
 */
extern HcclResult HcclGetRootInfo(HcclRootInfo *rootInfo);

/**
 * @brief Initialize HCCL with root info.
 *
 * @param nRanks A integer identifying the rank size of the cluster.
 * @param rootInfo A struct identifying the hccl root info.
 * @param rank A integer identifying the identify for the rank.
 * @param comm A pointer identifying the initialized communication resource.
 * @return HcclResult
 * @see HcclCommDestroy()
 */
extern HcclResult HcclCommInitRootInfo(uint32_t nRanks, const HcclRootInfo *rootInfo, uint32_t rank, HcclComm *comm);

/**
 * @brief Initialize HCCL with root info and config params.
 *
 * @param nRanks A integer identifying the rank size of the cluster.
 * @param rootInfo A struct identifying the hccl root info.
 * @param rank A integer identifying the identify for the rank.
 * @param config A pointer identifying config params about the current comm.
 * @param comm A pointer identifying the initialized communication resource.
 * @return HcclResult
 * @see HcclCommDestroy()
 */
extern HcclResult HcclCommInitRootInfoConfig(uint32_t nRanks, const HcclRootInfo *rootInfo, uint32_t rank,
    const HcclCommConfig *config, HcclComm *comm);

/* *

 * @brief Set deterministic calculate
 *
 * @param config A struct identifying the Config
 * @param configValue An interger identifying the identify for the config.
 */

extern HcclResult HcclSetConfig(HcclConfig config, HcclConfigValue configValue);
extern HcclResult HcclGetConfig(HcclConfig config, HcclConfigValue *configValue);

/**

 * @brief get commName.
 *
 * @param commhandle A pointer identifying the initialized communication resource.
 * @param commName The name of commhandle.
 * @return HcclResult
 * @see HcclCommDestroy()
 */
extern HcclResult HcclGetCommName(HcclComm comm, char* commName);


/**
 * @brief AllReduce operator.
 *
 * @param sendBuf A pointer identifying the input data address of the operator.
 * @param recvBuf A pointer identifying the output data address of the operator.
 * @param count An integer(u64) identifying the number of the output data.
 * @param dataType The data type of the operator, must be one of the following types: int8, int16, int32,
float16, float32, bfloat16.
 * @param op The reduction type of the operator, must be one of the following types: sum, min, max, prod.
 * @param comm A pointer identifying the communication resource based on.
 * @param stream A pointer identifying the stream information.
 * @return HcclResult
 */
extern HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream);

/**
 * @brief Broadcast operator.
 *
 * @param buf A pointer identifying the data address of the operator.
 * @param count An integer(u64) identifying the number of the data.
 * @param dataType The data type of the operator, must be one of the following types: int8, int16, int32, int64,
uint8, uint16, uint32, uint64, float16, float32, float64, bfloat16.
 * @param root An integer(u32) identifying the the root rank in the operator.
 * @param comm A pointer identifying the communication resource based on
 * @param stream A pointer identifying the stream information.
 * @return HcclResult
 */
extern HcclResult HcclBroadcast(void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm,
    aclrtStream stream);

/**
 * @brief ReduceScatter operator.
 *
 * @param sendBuf A pointer identifying the input data address of the operator.
 * @param recvBuf A pointer identifying the output data address of the operator.
 * @param recvCount An integer(u64) identifying the number of the output data.
 * @param dataType The data type of the operator, must be one of the following types: int8, int32,
 float16, float32, bfloat16.
 * @param op The reduction type of the operator, must be one of the following types: sum, min, max, prod.
 * @param comm A pointer identifying the communication resource based on.
 * @param stream A pointer identifying the stream information.
 * @return HcclResult
 */
extern HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream);

/**
 * @brief Scatter operator.
 *
 * @param sendBuf A pointer identifying the input data address of the operator.
 * @param recvBuf A pointer identifying the output data address of the operator.
 * @param recvCount An integer(u64) identifying the number of the data.
 * @param dataType The data type of the operator, must be one of the following types: int8, int32, float16, float32.
 * @param root An integer(u32) identifying the the root rank in the operator.
 * @param comm A pointer identifying the communication resource based on
 * @param stream A pointer identifying the stream information.
 * @return HcclResult 
 */
extern HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream);

/**
 * @brief AllGather operator.
 *
 * @param sendBuf A pointer identifying the input data address of the operator.
 * @param recvBuf A pointer identifying the output data address of the operator.
 * @param sendCount An integer(u64) identifying the number of the input data.
 * @param dataType The data type of the operator, must be one of the following types: int8, int16, int32, int64,
uint8, uint16, uint32, uint64, float16, float32, float64, bfloat16.
 * @param comm A pointer identifying the communication resource based on.
 * @param stream A pointer identifying the stream information.
 * @return HcclResult
 */
extern HcclResult HcclAllGather(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType,
    HcclComm comm, aclrtStream stream);
/**
 * @brief Get the rank size of this comm.
 *
 * @param comm A pointer identifying the communication resource based on.
 * @param rankSize  A pointer identifying the rank size.
 * @return HcclResult
 */
extern HcclResult HcclGetRankSize(HcclComm comm, uint32_t *rankSize);

/**
 * @brief Get the rank id of this comm.
 *
 * @param comm A pointer identifying the communication resource based on.
 * @param rankSize  A pointer identifying the rank id.
 * @return HcclResult
 */
extern HcclResult HcclGetRankId(HcclComm comm, uint32_t *rank);
/**
 * @brief Barrier operator.
 *
 * @param comm A pointer identifying the communication resource based on.
 * @param stream A pointer identifying the stream information.
 * @return HcclResult
 */
extern HcclResult HcclBarrier(HcclComm comm, aclrtStream stream);

/**
 * @brief Send operator.
 *
 * @param sendBuff A pointer identifying the input data address of the operator.
 * @param count An integer(u64) identifying the number of the send data.
 * @param dataType The data type of the operator, must be one of the following types: int8, int16, int32, int64,
uint8, uint16, uint32, uint64, float16, float32, float64, bfloat16.
 * @param destRank An integer identifying the destination rank.
 * @param comm A pointer identifying the communication resource based on.
 * @param stream A pointer identifying the stream information.
 * @return HcclResult
 */
extern HcclResult HcclSend(void* sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank,
                           HcclComm comm, aclrtStream stream);
/**
 * @brief Recv operator.
 *
 * @param recvBuff A pointer identifying the output data address of the operator.
 * @param count An integer(u64) identifying the number of the receive data.
 * @param dataType The data type of the operator, must be one of the following types: int8, int16, int32, int64,
uint8, uint16, uint32, uint64, float16, float32, float64, bfloat16.
 * @param srcRank An integer identifying the source rank.
 * @param comm A pointer identifying the communication resource based on.
 * @param stream A pointer identifying the stream information.
 * @return HcclResult
 */
extern HcclResult HcclRecv(void* recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank,
                           HcclComm comm, aclrtStream stream);

/**
 * @brief AlltoAllV operator.
 *
 * @param sendBuff A pointer identifying the input data address of the operator.
 * @param sendCounts Integer array, where entry i specifies the number of elements to send to rank i.
 * @param sdispls Integer array, where entry i specifies the displacement (offset from sendbuf, in units of sendtype)
from which to send data to rank i.
 * @param sendType Datatype of send buffer elements, must be one of the following types: int8, int16, int32, int64,
uint8, uint16, uint32, uint64, float16, float32, float64, bfloat16.
 * @param recvBuf A pointer identifying the output data address of the operator.
 * @param recvCounts Integer array, where entry j specifies the number of elements to receive from rank j.
 * @param rdispls Integer array, where entry j specifies the displacement (offset from recvbuf, in units of recvtype)
 to which data from rank j should be written.
 * @param recvType Datatype of receive buffer elements, must be one of the following types: int8, int16, int32, int64,
uint8, uint16, uint32, uint64, float16, float32, float64.
 * @param comm A pointer identifying the communication resource based on.
 * @param stream A pointer identifying the stream information.
 * @return HcclResult
 */
extern HcclResult HcclAlltoAllV(const void *sendBuf, const void *sendCounts, const void *sdispls, HcclDataType sendType,
                         const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType,
                         HcclComm comm, aclrtStream stream);

/**
 * @brief AlltoAll operator.
 *
 * @param sendBuff A pointer identifying the input data address of the operator.
 * @param sendCount Integer, number of elements to send to each proces.
 * @param sendType Datatype of send buffer elements, must be one of the following types: int8, int16, int32, int64,
uint8, uint16, uint32, uint64, float16, float32, float64, bfloat16.
 * @param recvBuf A pointer identifying the output data address of the operator.
 * @param recvCount Integer, number of elements received from any process.
 * @param recvType Datatype of receive buffer elements, must be one of the following types: int8, int16, int32, int64,
uint8, uint16, uint32, uint64, float16, float32, float64.
 * @param comm A pointer identifying the communication resource based on.
 * @param stream A pointer identifying the stream information.
 * @return HcclResult
 */
extern HcclResult HcclAlltoAll(const void *sendBuf, uint64_t sendCount, HcclDataType sendType,
                               const void *recvBuf, uint64_t recvCount, HcclDataType recvType,
                               HcclComm comm, aclrtStream stream);

/**
 * @brief Reduce operator.
 *
 * @param sendBuf A pointer identifying the input data address of the operator.
 * @param recvBuf A pointer identifying the output data address of the operator.
 * @param count An integer(u64) identifying the number of the output data.
 * @param dataType The data type of the operator, must be one of the following types: int8, int16, int32, float16,
 float32, bfloat16.
 * @param op The reduction type of the operator, must be one of the following types: sum, min, max, prod.
 * @param root An integer(u32) identifying the the root rank in the operator.
 * @param comm A pointer identifying the communication resource based on.
 * @param stream A pointer identifying the stream information.
 * @return HcclResult
 */
extern HcclResult HcclReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType,
                             HcclReduceOp op, uint32_t root, HcclComm comm, aclrtStream stream);

/**
 * @brief Destroy HCCL comm
 *
 * @param comm A pointer identifying the communication resource targetting
 * @return HcclResult
 * @see HcclCommInitClusterInfo()
 */
extern HcclResult HcclCommDestroy(HcclComm comm);

/**
 * @brief Create a single-process multi-npu communication domain. Cross-machine is not supported.
 *
 * @param ndev: the number of NPUs in a communication domain.
 * @param devices: Indicates the NPU list in the communication domain. The value is the device logic ID.
 The communication library creates communication domains in the sequence of devices.
 * @param comms: Generated communication domain handle, size: ndev * sizeof(HcclComm)
 * @return HcclResult
 */
extern HcclResult HcclCommInitAll(uint32_t ndev, int32_t* devices, HcclComm* comms);

/**
 * @brief Get hccl error.
 * @param comm A pointer identifying the communication resource based on.
 * @param asyncError A pointer identifying the communication error.
*/
extern HcclResult HcclGetCommAsyncError(HcclComm comm, HcclResult *asyncError);

/**
 * @brief  convert a hccl errorCode to a string.
 * @param code enum HcclResult.
*/
extern const char *HcclGetErrorString(HcclResult code);

/**
 * @brief  Batch SEND/RECV
 * @param sendRecvInfo A pointer to an send/recv item array.
 * @param itemNum The size of the send/recv item array.
 * @param comm A pointer identifying the communication resource based on.
 * @param stream A pointer identifying the stream information.
*/
extern HcclResult HcclBatchSendRecv(HcclSendRecvItem* sendRecvInfo, uint32_t itemNum, HcclComm comm, aclrtStream stream);


/**
 * @brief Initialize the comm configuration.
 * @param config Pointer to the comm configuration that needs to be initialized.
*/
inline void HcclCommConfigInit(HcclCommConfig *config)
{
    if (config == nullptr) {
        return;
    }

    typedef struct {
        size_t size;
        uint32_t magicWord;
        uint32_t version;
        uint64_t reserved;
    } configInfo_t;

    configInfo_t *info = (configInfo_t *)config;

    info->size = sizeof(HcclCommConfig);
    info->magicWord = HCCL_COMM_CONFIG_MAGIC_WORD;
    info->version = HCCL_COMM_CONFIG_VERSION;
    info->reserved = 0;

    config->hcclBufferSize = HCCL_COMM_DEFAULT_BUFFSIZE;
    config->hcclDeterministic = HCCL_COMM_DEFAULT_DETERMINISTIC;
}
#ifdef __cplusplus
}
#endif // __cplusplus
#endif // HCCL_H
