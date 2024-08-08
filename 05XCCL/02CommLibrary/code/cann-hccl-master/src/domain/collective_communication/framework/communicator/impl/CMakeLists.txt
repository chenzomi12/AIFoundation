set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/hccl_communicator.cc
)

target_sources(hccl PRIVATE
    ${src_list}
)

add_subdirectory(resource_manager)