set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_reduce_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_reduce_single_rank_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_reduce_ring_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_reduce_comm_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_reduce_double_ring_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_reduce_double_ring_concurrent_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_reduce_mesh_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_reduce_mesh_mid_count_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_reduce_mesh_oneshot_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_reduce_mesh_opbase_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_reduce_mesh_opbase_pipeline_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_reduce_mesh_small_count_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_reduce_reduce_plus_bcast_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_reduce_mesh_aiv_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_reduce_mesh_opbase_big_count_aiv_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_reduce_small_count_aiv_rdma_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_reduce_mid_count_aiv_rdma_executor.cc
)

target_sources(hccl_alg PRIVATE
    ${src_list}
)

add_subdirectory(310P)