set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/config.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/param_check.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/json_utils.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/hccl_hash_utils.cc
)

target_sources(hccl PRIVATE
    ${src_list}
)

add_subdirectory(opexecounter)
add_subdirectory(topo)
add_subdirectory(task)