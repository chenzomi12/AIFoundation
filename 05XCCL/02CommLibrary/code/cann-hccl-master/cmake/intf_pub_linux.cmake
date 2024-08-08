
add_library(intf_pub_base INTERFACE)

target_compile_options(intf_pub_base INTERFACE
    -fPIC
    -pipe
    $<IF:$<VERSION_GREATER:${CMAKE_C_COMPILER_VERSION},4.8.5>,-fstack-protector-strong,-fstack-protector-all>
)

target_compile_definitions(intf_pub_base INTERFACE
    _GLIBCXX_USE_CXX11_ABI=0
    $<$<CONFIG:Release>:CFG_BUILD_NDEBUG>
    $<$<CONFIG:Debug>:CFG_BUILD_DEBUG>
    LINUX=0
)

target_link_options(intf_pub_base INTERFACE
    $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:-pie>
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
    $<$<CONFIG:Release>:-Wl,--build-id=none>
)

target_link_directories(intf_pub_base INTERFACE
)

target_link_libraries(intf_pub_base INTERFACE
    -pthread
)

add_library(intf_pub INTERFACE)

target_compile_options(intf_pub INTERFACE
    -Wall
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++11>
)

target_link_libraries(intf_pub INTERFACE
    $<BUILD_INTERFACE:intf_pub_base>
)







