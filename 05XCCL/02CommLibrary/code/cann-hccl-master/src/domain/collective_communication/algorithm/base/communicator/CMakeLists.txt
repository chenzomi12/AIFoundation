set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/comm_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/comm_factory.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/comm_ring.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/comm_p2p.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/comm_mesh.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/comm_halving_doubling.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/comm_star.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/calc_hd_transport_req.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/calc_mesh_transport_req.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/calc_partial_mesh_transport_req.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/calc_ring_transport_req.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/calc_transport_req_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/calc_p2p_transport_req.cc
)

target_sources(hccl_alg PRIVATE
    ${src_list}
)