cmake_minimum_required(VERSION 3.16.0)
project(hccl_alg)

set(CMAKE_VERBOSE_MAKEFILE ON)

add_library(hccl_alg SHARED)

set(HCCL_BASE_DIR ${TOP_DIR}/ace/comop/hccl/whole/hccl/)

target_include_directories(hccl_alg PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/base/inc
    ${HCCL_BASE_DIR}/pub_inc
    ${HCCL_BASE_DIR}/pub_inc/hccl
    ${HCCL_BASE_DIR}/inc
    ${HCCL_BASE_DIR}/inc/adapter
    ${TOP_DIR}/inc
    ${TOP_DIR}/open_source/json/include
    ${TOP_DIR}/abl/msprof/inc
    ${TOP_DIR}/ace/npuruntime/inc
    ${TOP_DIR}/ace/npuruntime/acl/inc/external
    ${CMAKE_CURRENT_SOURCE_DIR}/base/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/base/executor/component/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/base/communicator
    ${CMAKE_CURRENT_SOURCE_DIR}/impl
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/resource_manager
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/task
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/operator
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/operator/registry
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/coll_executor
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/coll_executor/registry
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/coll_executor/coll_send_receive
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/coll_executor/coll_all_reduce
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/coll_executor/coll_all_reduce/310P
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/coll_executor/coll_all_to_all
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/coll_executor/coll_all_gather
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/coll_executor/coll_all_gather/310P
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/coll_executor/coll_reduce_scatter
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/coll_executor/coll_reduce_scatter/310P
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/coll_executor/coll_scatter
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/coll_executor/coll_broadcast
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/coll_executor/coll_broadcast/310P
    ${HCCL_BASE_DIR}/../../../../inc/external
    ${HCCL_BASE_DIR}/../../../../inc/external/hccl
    ${HCCL_BASE_DIR}/../../../../inc/hccl
    ${HCCL_BASE_DIR}/../../../../inc
)

if(BUILD_OPEN_PROJECT)
    target_include_directories(hccl_alg PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../pub_inc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../inc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../inc/hccl
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/hccl
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef/common/util/error_manager/
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof/
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime/
    )

    target_link_directories(hccl_alg PRIVATE
        ${ASCEND_CANN_PACKAGE_PATH}/lib64
    )
else()
    target_include_directories(hccl_alg PRIVATE
        ${TOP_DIR}/abl/msprof/inc
        ${TOP_DIR}/ace/npuruntime/inc
        ${TOP_DIR}/inc
        ${TOP_DIR}/metadef/inc/common/util/error_manager/
    )
endif()

target_compile_definitions(hccl_alg PRIVATE
    $<$<STREQUAL:${PRODUCT_SIDE},host>:_GLIBCXX_USE_CXX11_ABI=0>
)

target_compile_options(hccl_alg PRIVATE
    -Werror
    -fno-common
    -fno-strict-aliasing
    -pipe
    -O3
    -std=c++14
    -fstack-protector-all
)

target_link_options(hccl_alg PRIVATE
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
)

target_link_libraries(hccl_alg PRIVATE
    -Wl,--no-as-needed
    c_sec
    -Wl,--no-as-needed
    hccl_alg_research
    hccl_plf
)

add_subdirectory(base)
add_subdirectory(impl)

install(TARGETS hccl_alg
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL
)

if(NOT BUILD_OPEN_PROJECT)
    install_package(
        PACKAGE hccl
        TARGETS hccl_alg
    )
endif()

if(BUILD_OPEN_PROJECT)
    set(install_script_dir ${CMAKE_CURRENT_BINARY_DIR}/install_scripts/)

    add_custom_target(generate_install_script ALL
        COMMAND rm -rf ${install_script_dir}
        COMMAND cp -rf ${ASCEND_CANN_PACKAGE_PATH}/tools/ascend_project/open_install_scripts ${install_script_dir}
        COMMAND chmod -R u+w ${install_script_dir}
        COMMAND echo "base_package=hccl" > ${install_script_dir}/version.info
        COMMAND echo "backup_dir=${CMAKE_PROJECT_NAME}" >> ${install_script_dir}/version.info
        COMMAND echo "Version=${CANN_VERSION}" >> ${install_script_dir}/version.info
    )

    install(DIRECTORY ${install_script_dir}
        DESTINATION .
        FILE_PERMISSIONS OWNER_EXECUTE OWNER_READ GROUP_READ
    )

    set(CPACK_PACKAGE_NAME ${CMAKE_PROJECT_NAME})
    set(CPACK_PACKAGE_VERSION ${CMAKE_PROJECT_VERSION})
    set(CPACK_PACKAGE_DESCRIPTION "CPack hccl project")
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "CPack hccl project")
    set(CPACK_PACKAGE_DIRECTORY ${CMAKE_BINARY_DIR})
    set(CPACK_PACKAGE_FILE_NAME "CANN-hccl_alg-${CANN_VERSION}-linux.${CMAKE_SYSTEM_PROCESSOR}.run")
    set(CPACK_GENERATOR External)
    set(CPACK_CMAKE_GENERATOR "Unix Makefiles")
    set(CPACK_EXTERNAL_ENABLE_STAGING TRUE)
    set(CPACK_EXTERNAL_PACKAGE_SCRIPT ${ASCEND_CANN_PACKAGE_PATH}/tools/ascend_project/cmake/makeself.cmake)
    set(CPACK_EXTERNAL_BUILT_PACKAGES ${CPACK_PACKAGE_DIRECTORY}/_CPack_Packages/Linux/External/${CPACK_PACKAGE_FILE_NAME}/${CPACK_PACKAGE_FILE_NAME})
    include(CPack)
endif()
