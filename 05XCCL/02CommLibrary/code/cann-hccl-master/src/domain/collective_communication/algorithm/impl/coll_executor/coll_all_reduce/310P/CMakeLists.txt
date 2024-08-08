set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_reduce_for_310p_ring_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_reduce_for_310p_doubling_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_reduce_for_310p_doubling_direct_executor.cc
)

target_sources(hccl_alg PRIVATE
    ${src_list}
)