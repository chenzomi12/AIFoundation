set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/callback_thread_manager.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/hccl_callback_task.cc
)

target_sources(hccl PRIVATE
    ${src_list}
)