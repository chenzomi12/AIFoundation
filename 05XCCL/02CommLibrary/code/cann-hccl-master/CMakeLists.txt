cmake_minimum_required(VERSION 3.16.0)
project(hccl)

option(BUILD_OPEN_PROJECT "Build open hccl project." ON)

if(BUILD_OPEN_PROJECT)
    include(cmake/config.cmake)
    add_subdirectory(src/domain/collective_communication)

    set(HCCL_HEAD
        inc/hccl/hccl.h
        inc/hccl/hccl_types.h
    )

    install(FILES
        ${HCCL_HEAD}
        DESTINATION ${INSTALL_INCLUDE_DIR}/hccl/ OPTIONAL
    )
endif()