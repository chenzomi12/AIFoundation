set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/ccl_buffer_manager.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/hccl_socket_manager.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/op_base_stream_manager.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/offload_stream_manager.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/queue_notify_manager.cc

    ${CMAKE_CURRENT_SOURCE_DIR}/workspace_mem.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/workspace_resource_impl.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/workspace_resource.cc

    ${CMAKE_CURRENT_SOURCE_DIR}/stream_active_manager.cc
)

target_sources(hccl_alg PRIVATE
    ${src_list}
)
