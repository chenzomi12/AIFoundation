set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_executor_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_native_executor_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_comm_executor.cc
)

target_sources(hccl_alg PRIVATE
    ${src_list}
)

add_subdirectory(coll_all_gather)
add_subdirectory(coll_all_reduce)
add_subdirectory(coll_reduce_scatter)
add_subdirectory(coll_reduce)
add_subdirectory(coll_send_receive)
add_subdirectory(coll_all_to_all)
add_subdirectory(coll_scatter)
add_subdirectory(coll_broadcast)
add_subdirectory(registry)