set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/op_base.cc
)

target_sources(hccl PRIVATE
    ${src_list}
)