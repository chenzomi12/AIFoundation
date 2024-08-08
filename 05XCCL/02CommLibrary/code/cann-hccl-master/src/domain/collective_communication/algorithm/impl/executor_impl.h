/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXECUTOR_IMPL_H
#define EXECUTOR_IMPL_H

/* * 节点间的集合通信操作汇总 */
#include "all_reduce_reduce_broadcast_pub.h"
#include "all_reduce_ring_pub.h"
#include "inc_all_reduce_deter/all_reduce_local_reduce_bcast_pub.h"
#include "all_reduce_mesh_opbase_pub.h"
#include "inc_all_reduce_deter/all_reduce_local_reduce_pub.h"
#include "all_reduce_mesh_oneshot_pub.h"
#include "inc_all_reduce_deter/all_reduce_chunk_mesh_pub.h"
#include "all_reduce_recursive_hd_pub.h"
#include "all_reduce_nhr_pub.h"
#include "all_reduce_nhr_oneshot_pub.h"
#include "all_reduce_nhr_v1_pub.h"
#include "all_reduce_nb_pub.h"
#include "all_reduce_doubling_pub.h"
#include "all_reduce_doubling_direct_pub.h"
#include "reduce_recursive_hd_pub.h"
#include "reduce_ring_pub.h"
#include "reduce_nhr_oneshot_pub.h"
#include "reduce_scatter_ring_pub.h"
#include "reduce_scatter_ring_concurrent_direct_pub.h"
#include "all_gather_ring_concurrent_direct_pub.h"
#include "scatter_ring_concurrent_direct_pub.h"
#include "multi_root_scatter_ring_pub.h"
#include "all_gather_ring_pub.h"
#include "broadcast_ring_pub.h"
#include "bcast_halvingdoubling_pub.h"
#include "bcast_recursive_halvingdoubling_pub.h"
#include "broadcast_nhr_pub.h"
#include "broadcast_nhr_oneshot_pub.h"
#include "broadcast_nhr_v1_pub.h"
#include "broadcast_nb_pub.h"
#include "broadcast_nb_binary_pub.h"
#include "scatter_mesh_pub.h"
#include "scatter_ring_pub.h"
#include "scatter_nhr_pub.h"
#include "scatter_nb_pub.h"
#include "reduce_scatter_mesh_pub.h"
#include "all_gather_mesh_pub.h"
#include "all_gather_mesh_direct.h"
#include "all_gather_mesh_atomic.h"
#include "all_gather_pipeline_pub.h"
#include "all_gather_nhr_pub.h"
#include "all_gather_nhr_v1_pub.h"
#include "all_gather_nb_pub.h"
#include "reduce_scatter_halving_doubling_pub.h"
#include "reduce_scatter_recursive_hd_pub.h"
#include "all_gather_halving_doubling_pub.h"
#include "all_gather_recursive_hd_pub.h"
#include "send_receive_pub.h"
#include "gather_ring_pub.h"
#include "gather_mesh_pub.h"
#include "broadcast_star_pub.h"
#include "gather_star_pub.h"
#include "alltoallv_mesh_read_only_pub.h"
#include "alltoallv_pairwise_pub.h"
#include "alltoallv_staged_calculator_pub.h"
#include "alltoallv_staged_pairwise_pub.h"
#include "alltoallv_staged_mesh_pub.h"
#include "reduce_scatter_mesh_atomic.h"
#include "reduce_scatter_hd_stage_pub.h"
#include "reduce_scatter_local_reduce_pub.h"
#include "reduce_scatter_mesh_atomic_opbase.h"
#include "reduce_scatter_nhr_pub.h"
#include "reduce_scatter_nhr_v1_pub.h"
#include "reduce_scatter_nb_pub.h"
#include "reduce_scatter_pipeline_pub.h"
#include "all_reduce_opbase_pipeline_pub.h"
#include "allltoall_pipeline_mesh_pairwise_ping_pong_pub.h"
#include "allltoall_pipeline_mesh_pairwise_ccl_enough_pub.h"
#include "allltoall_pipeline_base_pub.h"
#include "alltoallv_mesh_read_only_pub.h"

namespace hccl {
}

#endif /* * EXECUTOR_IMPL_H */
