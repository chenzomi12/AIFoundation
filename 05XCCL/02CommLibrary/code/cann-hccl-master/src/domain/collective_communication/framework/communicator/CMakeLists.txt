set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/hccl_comm.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/comm_config.cc
)

target_sources(hccl PRIVATE
    ${src_list}
)

add_subdirectory(impl)