set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/hccl_impl.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/hccl_alg.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/hccl_pipeline.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/topo_matcher.cc
)

target_sources(hccl_alg PRIVATE
    ${src_list}
)

add_subdirectory(resource_manager)
add_subdirectory(task)
add_subdirectory(operator)
add_subdirectory(coll_executor)