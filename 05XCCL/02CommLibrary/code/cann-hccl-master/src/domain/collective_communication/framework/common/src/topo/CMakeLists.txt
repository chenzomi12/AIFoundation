set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/topoinfo_ranktableParser.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/topoinfo_ranktableStandard.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/topoinfo_ranktableConcise.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/topoinfo_ranktableHeterog.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/topoinfo_roletableParser.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/topoinfo_detect.cc 
    ${CMAKE_CURRENT_SOURCE_DIR}/topoinfo_exchange_agent.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/topoinfo_exchange_server.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/topoinfo_exchange_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/topoinfo_exchange_dispatcher.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/topoinfo_parse.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/topoinfo_ranktableOffline.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/hccl_whitelist.cc
)

target_sources(hccl   PRIVATE
    ${src_list}
)