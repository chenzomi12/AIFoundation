set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_alg_operator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common_operator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/scatter_operator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_operator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce_operator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_operator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/gather_operator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/all_gather_operator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/broadcast_operator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/send_receive_operator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/alltoall_operator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/batchsendrecv_operator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/send_operator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/receive_operator.cc
)

target_sources(hccl_alg PRIVATE
    ${src_list}
)

add_subdirectory(registry)