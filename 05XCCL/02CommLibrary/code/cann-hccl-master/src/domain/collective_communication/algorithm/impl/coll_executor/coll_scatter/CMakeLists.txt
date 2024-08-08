set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_scatter_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_scatter_ring_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_scatter_comm_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_scatter_mesh_executor.cc
)

target_sources(hccl_alg PRIVATE
    ${src_list}
)

