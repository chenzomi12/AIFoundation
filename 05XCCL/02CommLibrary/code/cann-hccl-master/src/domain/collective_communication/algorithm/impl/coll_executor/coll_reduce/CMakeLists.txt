set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_reduce_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_reduce_mesh_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_reduce_ring_plus_hd_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_reduce_double_ring_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_reduce_comm_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_reduce_single_rank_executor.cc
)
 
target_sources(hccl_alg PRIVATE
    ${src_list}
)