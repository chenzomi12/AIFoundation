set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_batch_send_recv_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_send_executor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_receive_executor.cc
)
 
target_sources(hccl_alg PRIVATE
    ${src_list}
)