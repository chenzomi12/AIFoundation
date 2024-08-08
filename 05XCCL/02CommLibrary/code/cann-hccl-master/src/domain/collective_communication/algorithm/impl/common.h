/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMON_H
#define COMMON_H

#include "adapter_hccp_common.h"
#include "hccl_ip_address.h"
#include "hccl_common.h"

// sub stream 相关
constexpr s64 HCCL_SUB_STREAM_NUM_ZERO = 0;  // subStream 数量为0
constexpr s64 HCCL_SUB_STREAM_NUM_DOUBLE_RING = 1;  // subStream 数量为1
constexpr s64 HCCL_SUB_STREAM_NUM_4P_MESH = 2;  // subStream 数量为2
constexpr s64 HCCL_SUB_STREAM_NUM_8P_RING = 3;  // subStream 数量为3
constexpr s64 HCCL_SUB_STREAM_NP_MESH = 2;  // NP MESH场景下subStream数应为ranksize - 2
namespace hccl {
// 通信域内rank信息
using RankInfo = struct TagRankInfo {
public:
    /* * 用户的原始属性 */
    u32 userRank{INVALID_UINT};                                           // 当前rank的user rank ID
    u32 worldRank{INVALID_UINT};                                          // 当前rank的global user rank ID
    u32 localRank{INVALID_UINT};                                          // 当前rank在server内的rank ID
    s32 devicePhyId{-1};                                                  // 当前rank在操作的设备物理编号
    DevType deviceType{DevType::DEV_TYPE_NOSOC};                          // 当前rank在操作的设备属性
    std::vector<HcclIpAddress> nicIp;                                     // 当前rank所归属网卡的IP值(实际建链所用网卡IP)
    std::vector<u32> nicIdx;                                              // 当前rank在server内的所有device网卡位置（devicePhyId）
    NICDeployment nicDeploy{NICDeployment::NIC_DEPLOYMENT_DEVICE};        // 参数平面位置 0:host 1:device other:reserve
    HcclIpAddress hostIp;                                                 // 当前rank使用的host socket addr
    u32 hostPort{INVALID_UINT};                                           // 当前rank使用的host socket port
    std::string serverId{""};                                             // 当前rank所在服务器的唯一标识值 (rank_table中的server id)
    u32 serverIdx{INVALID_UINT};                                          // Server在ranktable中的自然顺序（用户指定）
    u32 superDeviceId{INVALID_UINT};                                      // 当前rank所在的超节点内的device id（sdid）
    std::string superPodId{""};                                         // 当前rank所在超节点id
};

constexpr s64 HCCL_SMALL_COUNT_128_KB = 128 * 1024;  // hccl 910B/310P duo卡单算子小数据量标准，暂定128kb
constexpr s64 HCCL_SMALL_COUNT_32_KB = 32 * 1024;  // hccl小数据量标准，暂定512KB
constexpr s64 HCCL_SMALL_COUNT_GRAPH_64_KB = 64 * 1024;  // hccl图模式小数据量标准，暂定64KB
constexpr s64 HCCL_MEDIUM_COUNT_GRAPH_4_MB = 4 * 1024 * 1024;     // hccl图模式中数据量标准，暂定4MB
constexpr s64 HCCL_SMALL_COUNT_256_KB = 256 * 1024;  // 910B/310p V卡hccl小数据量标准，暂定256KB

constexpr s64 HCCL_ALIGN_COUNT_32_B = 32;  // 910B aiv+rdma，中数据量情况下按照32B对齐
constexpr s64 HCCL_SMALL_COUNT_190_KB = 190 * 1024;  // 910B aiv+rdma，暂定从OutAIVBuffer的512K位置开始存放标记
constexpr s64 HCCL_SMALL_COUNT_512_KB = 512 * 1024;  // 910B aiv+rdma，暂定从OutAIVBuffer的512K位置开始存放标记
constexpr s64 HCCL_SMALL_COUNT_1_MB = 1024 * 1024;  // 910BB aiv+rdma，AIVIN的最后1M空间用作标记区
constexpr s64 HCCL_SMALL_COUNT_2_MB = 2048 * 1024;
constexpr s64 HCCL_SMALL_COUNT_4_MB = 4096 * 1024;
constexpr s64 HCCL_SMALL_COUNT_8_MB = 8192 * 1024;
constexpr s64 HCCL_MID_COUNT_16_MB = 16 * 1024 * 1024;  // 910B aiv+rdma 中数据量支持上限
constexpr s64 HCCL_MID_COUNT_32_MB = 32 * 1024 * 1024;

constexpr u32 HCCL_INTER_SERVER_RING_ALGO_MAX_SUPPORT_SERVER_NUM = 8; // server 间 ring 算法支持的最大server数: 8

constexpr u32 HCCL_ALLTOALLV_P2P_SIZE = 2; // alltoallv不区分全连接、分级的最大ranksize

// a2a前置allgather和a2a注册的notify资源要隔离, tag不能包含HCCL_ALLTOALL字符串
const std::string HCCL_ALLTOALL_PARA_ALLGATHER = "AllgatherForCollectA2APara";

enum class AlgTypeLevel2 {
    ALG_LEVEL2_WHOLE_RING = 0,  // 单层拓扑, 所有leve2均为Whole ring时，组成一个大环
    ALG_LEVEL2_HD,              // 拓扑组合2层, HDR
    ALG_LEVEL2_RING,            // 拓扑组合2层, Ring
    ALG_LEVEL2_RESERVED
};

const std::map<AlgTypeLevel0, std::string> HCCL_ALGO_LEVEL0_NAME_MAP = {
    {AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING, "ring"},
    {AlgTypeLevel0::ALG_LEVEL0_WHOLE_RING, "ring"},
    {AlgTypeLevel0::ALG_LEVEL0_8P_RING, "ring"},
    {AlgTypeLevel0::ALG_LEVEL0_4P_MESH, "fullmesh"},
    {AlgTypeLevel0::ALG_LEVEL0_2P_MESH, "fullmesh"},
    {AlgTypeLevel0::ALG_LEVEL0_1P_MESH, "fullmesh"},
    {AlgTypeLevel0::ALG_LEVEL0_4P_RING, "ring"},
    {AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING, "ring"},
    {AlgTypeLevel0::ALG_LEVEL0_NP_MESH, "fullmesh"},
    {AlgTypeLevel0::ALG_LEVEL0_NP_HD, "HD"},
    {AlgTypeLevel0::ALG_LEVEL0_NP_STAR, "star"},
    {AlgTypeLevel0::ALG_LEVEL0_RESERVED, "null"},
};

const std::map<AlgTypeLevel1, std::string> HCCL_ALGO_LEVEL1_NAME_MAP = {
    {AlgTypeLevel1::ALG_LEVEL1_WHOLE_RING, "ring"},
    {AlgTypeLevel1::ALG_LEVEL1_HD, "H-D"},
    {AlgTypeLevel1::ALG_LEVEL1_RING, "ring"},
    {AlgTypeLevel1::ALG_LEVEL1_PIPELINE, "pipeline"},
    {AlgTypeLevel1::ALG_LEVEL1_NHR, "NHR"},
    {AlgTypeLevel1::ALG_LEVEL1_NHR_V1, "NHR_V1"},
    {AlgTypeLevel1::ALG_LEVEL1_NB, "NB"},
    {AlgTypeLevel1::ALG_LEVEL1_RESERVED, "null"},
};
}  // namespace hccl

struct SendRecvInfo {
    // 存放数据长度和偏移长度
    std::vector<u64> sendLength;
    std::vector<u64> sendOffset;
    std::vector<u64> recvLength;
    std::vector<u64> recvOffset;
    // 存放数据个数和偏移个数
    std::vector<u64> sendCounts;
    std::vector<u64> sendDispls;
    std::vector<u64> recvCounts;
    std::vector<u64> recvDispls;
};

#endif /* COMMON_H */
