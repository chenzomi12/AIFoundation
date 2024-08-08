set(src_list
    threadManage.cc
    parallel_task_loader.cc
    task_loader.cc
)

target_sources(hccl_alg PRIVATE
    ${src_list}
)
