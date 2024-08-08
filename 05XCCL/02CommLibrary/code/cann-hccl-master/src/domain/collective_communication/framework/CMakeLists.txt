set(HCCL_PACKET_FLAG 1)

if(BUILD_OPEN_PROJECT)
    set(HCCL_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../)
else()
    set(HCCL_BASE_DIR ${TOP_DIR}/ace/comop/hccl/whole/hccl/)
endif()

add_library(hccl  SHARED)

set(hccl_include_list
    ${HCCL_BASE_DIR}/algorithm/base/inc
    ${HCCL_BASE_DIR}/pub_inc
    ${HCCL_BASE_DIR}/pub_inc/hccl
    ${HCCL_BASE_DIR}/algorithm/base/executor
    ${HCCL_BASE_DIR}/algorithm/base/executor/component/inc
    ${HCCL_BASE_DIR}/algorithm/base/communicator
    ${HCCL_BASE_DIR}/algorithm/impl
    ${HCCL_BASE_DIR}/algorithm/impl/task
    ${HCCL_BASE_DIR}/algorithm/impl/resource_manager
    ${HCCL_BASE_DIR}/algorithm/impl/operator
    ${HCCL_BASE_DIR}/algorithm/impl/coll_executor

    ${HCCL_BASE_DIR}/framework/common/src/
    ${HCCL_BASE_DIR}/framework/common/src/topo
    ${HCCL_BASE_DIR}/framework/common/src/task
    ${HCCL_BASE_DIR}/framework/communicator/impl/
    ${HCCL_BASE_DIR}/framework/communicator/impl/resource_manager
    ${HCCL_BASE_DIR}/framework/hcom/
    ${HCCL_BASE_DIR}/framework/inc/
    ${HCCL_BASE_DIR}/framework/op_base/src
)

if(BUILD_OPEN_PROJECT)
    target_include_directories(hccl PRIVATE
        ${HCCL_BASE_DIR}/../../../inc
        ${HCCL_BASE_DIR}/../../../inc/hccl
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/hccl
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef/common/util/error_manager/
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof/
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime/
        ${THIRD_PARTY_NLOHMANN_PATH}
    )

    target_link_directories(hccl PRIVATE
        ${ASCEND_CANN_PACKAGE_PATH}/lib64
    )
else()
    target_include_directories(hccl PRIVATE
        ${TOP_DIR}/ace/comop/inc/external/hccl/             # ${ASCEND_CANN_PACKAGE_PATH}/include
        ${TOP_DIR}/ace/comop/inc/hccl/                      # ${ASCEND_CANN_PACKAGE_PATH}/include
        ${TOP_DIR}/abl/msprof/inc                           # experiment/msprof/
        ${TOP_DIR}/ace/npuruntime/inc                       # experiment/runtime/
        ${TOP_DIR}/inc/driver                               # experiment/ascend_hal/driver/
        ${TOP_DIR}/inc/external                             # experiment
        ${TOP_DIR}/metadef/inc/common/util/error_manager    # experiment/metadef/inc/common/util/error_manager/
        ${TOP_DIR}/open_source/json/include
    )
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

target_compile_options(hccl PRIVATE
    -Werror
    -fno-common
    -fno-strict-aliasing
    -pipe
    -O3
    -std=c++14
    -fstack-protector-all
)

target_link_options(hccl PRIVATE
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
)

target_compile_definitions(hccl PRIVATE
    $<$<STREQUAL:${PRODUCT_SIDE},host>:_GLIBCXX_USE_CXX11_ABI=0>
)
target_include_directories(hccl PRIVATE
    ${hccl_include_list}
)

target_link_libraries(hccl
        -Wl,--no-as-needed
        c_sec
        hccl_alg
        hccl_plf
        -Wl,--as-needed
        -lrt
        -ldl
        -lpthread
)

add_subdirectory(common)
add_subdirectory(communicator)
add_subdirectory(hcom)
add_subdirectory(op_base)

install(TARGETS hccl
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL
)