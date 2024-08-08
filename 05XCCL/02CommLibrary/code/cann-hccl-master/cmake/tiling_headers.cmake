set(ASCENDC_INSTALL_BASE_PATH ${CMAKE_INSTALL_PREFIX}/lib)
file(MAKE_DIRECTORY  ${ASCENDC_INSTALL_BASE_PATH}/tikcpp)
file(MAKE_DIRECTORY  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling)

file(CREATE_LINK ../lib/activation
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/activation SYMBOLIC)
file(CREATE_LINK ../lib/filter
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/filter SYMBOLIC)
file(CREATE_LINK ../lib/index
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/index SYMBOLIC)
file(CREATE_LINK ../lib/kfc
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/kfc SYMBOLIC)
file(CREATE_LINK ../lib/math
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/math SYMBOLIC)
file(CREATE_LINK ../lib/matmul
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/matmul SYMBOLIC)
file(CREATE_LINK ../lib/normalization
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/normalization SYMBOLIC)
file(CREATE_LINK ../lib/pad
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/pad SYMBOLIC)
file(CREATE_LINK ../lib/quantization
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/quantization SYMBOLIC)
file(CREATE_LINK ../lib/pad
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/pad SYMBOLIC)
file(CREATE_LINK ../lib/quantization
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/quantization SYMBOLIC)
file(CREATE_LINK ../lib/reduce
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/reduce SYMBOLIC)
file(CREATE_LINK ../lib/select
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/select SYMBOLIC)
file(CREATE_LINK ../lib/sort
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/sort SYMBOLIC)
file(CREATE_LINK ../lib/transpose
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/transpose SYMBOLIC)

# arithprogression
file(MAKE_DIRECTORY  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/arithprogression)
file(CREATE_LINK ../../lib/index/arithprogression_tiling.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/arithprogression/arithprogression_tiling.h SYMBOLIC)
file(CREATE_LINK ../lib/index/arithprogression_tiling_intf.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/arithprogression_tiling_intf.h SYMBOLIC)

# ascend_antiquant
file(MAKE_DIRECTORY  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/ascend_antiquant)
file(CREATE_LINK ../../lib/quantization/ascend_antiquant_tiling.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/ascend_antiquant/ascend_antiquant_tiling.h SYMBOLIC)
file(CREATE_LINK ../../lib/quantization/ascend_antiquant_tiling_intf.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/ascend_antiquant/ascend_antiquant_tiling_intf.h SYMBOLIC)

# ascend_dequant
file(MAKE_DIRECTORY  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/ascend_dequant)
file(CREATE_LINK ../../lib/quantization/ascend_dequant_tiling.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/ascend_dequant/ascend_dequant_tiling.h SYMBOLIC)
file(CREATE_LINK ../../lib/quantization/ascend_dequant_tiling_intf.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/ascend_dequant/ascend_dequant_tiling_intf.h SYMBOLIC)

# ascend_quant
file(MAKE_DIRECTORY  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/ascend_quant)
file(CREATE_LINK ../../lib/quantization/ascend_quant_tiling.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/ascend_quant/ascend_quant_tiling.h SYMBOLIC)
file(CREATE_LINK ../../lib/quantization/ascend_quant_tiling_intf.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/ascend_quant/ascend_quant_tiling_intf.h SYMBOLIC)

# batchnorm
file(MAKE_DIRECTORY  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/batchnorm)
file(CREATE_LINK ../../lib/normalization/batchnorm_tiling.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/batchnorm/batchnorm_tiling.h SYMBOLIC)
file(CREATE_LINK ../../lib/normalization/batchnorm_tiling_intf.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/batchnorm/batchnorm_tlevel_api/tiling/batchnorm/batchnorm_tilingdata.h SYMBOLIC)

# broadcast
file(MAKE_DIRECTORY  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/broadcast)
file(CREATE_LINK ../../lib/pad/broadcast_tiling_intf.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/broadcast/broadcast_tiling_intf.h SYMBOLIC)

# deepnorm
file(MAKE_DIRECTORY  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/deepnorm)
file(CREATE_LINK ../../lib/normalization/deepnorm_tiling.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/deepnorm/deepnorm_tiling.h SYMBOLIC)
file(CREATE_LINK ../../lib/normalization/deepnorm_tiling_intf.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/deepnorm/deepnorm_tiling_intf.h SYMBOLIC)
file(CREATE_LINK ../../lib/normalization/deepnorm_tilingdata.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/deepnorm/deepnorm_tilingdata.h SYMBOLIC)

# dropout
file(MAKE_DIRECTORY  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/dropout)
file(CREATE_LINK ../../lib/filter/dropout_tiling.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/dropout/dropout_tiling.h SYMBOLIC)
file(CREATE_LINK ../lib/filter/dropout_tiling_intf.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/dropout_tiling_intf.h SYMBOLIC)

# gelu
file(MAKE_DIRECTORY  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/gelu)
file(CREATE_LINK ../../lib/activation/gelu_tiling.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/gelu/gelu_tiling.h SYMBOLIC)
file(CREATE_LINK ../../lib/activation/gelu_tiling_intf.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/gelu/gelu_tiling_intf.h SYMBOLIC)

# layernorm
file(MAKE_DIRECTORY  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/layernorm)
file(CREATE_LINK ../../lib/normalization/layernorm_tiling.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/layernorm/layernorm_tiling.h SYMBOLIC)
file(CREATE_LINK ../../lib/normalization/layernorm_tiling_intf.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/layernorm/layernorm_tiling_intf.h SYMBOLIC)
file(CREATE_LINK ../../lib/normalization/layernorm_tilingdata.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/layernorm/layernorm_tilingdata.h SYMBOLIC)

# layernormgrad
file(MAKE_DIRECTORY  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/layernormgrad)
file(CREATE_LINK ../../lib/normalization/layernorm_grad_tiling.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/layernormgrad/layernorm_grad_tiling.h SYMBOLIC)
file(CREATE_LINK ../../lib/normalization/layernorm_grad_tiling_intf.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/layernormgrad/layernorm_grad_tiling_intf.h SYMBOLIC)
file(CREATE_LINK ../../lib/normalization/layernorm_grad_tilingdata.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/layernormgrad/layernorm_grad_tilingdata.h SYMBOLIC)
file(CREATE_LINK ../../lib/normalization/layernorm_grad_beta_tiling.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/layernormgrad/layernorm_grad_beta_tiling.h SYMBOLIC)
file(CREATE_LINK ../../lib/normalization/layernorm_grad_beta_tiling_intf.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/layernormgrad/layernorm_grad_beta_tiling_intf.h SYMBOLIC)
file(CREATE_LINK ../../lib/normalization/layernorm_grad_beta_tilingdata.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/layernormgrad/layernorm_grad_beta_tilingdata.h SYMBOLIC)

# matmul
file(MAKE_DIRECTORY  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/matrix)
file(CREATE_LINK ../../lib/matmul/bmm_tiling.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/matrix/bmm_tiling.h SYMBOLIC)
file(CREATE_LINK ../../lib/matmul/matmul_tiling.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/matrix/matmul_tiling.h SYMBOLIC)
file(CREATE_LINK ../../lib/matmul/matmul_tiling_base.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/matrix/matmul_tiling_base.h SYMBOLIC)
file(CREATE_LINK ../../lib/matmul/matmul_tilingdata.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/matrix/matmul_tilingdata.h SYMBOLIC)

# mean
file(MAKE_DIRECTORY  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/mean)
file(CREATE_LINK ../../lib/reduce/mean_tiling_intf.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/mean/mean_tiling_intf.h SYMBOLIC)

# reglu
file(MAKE_DIRECTORY  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/reglu)
file(CREATE_LINK ../../lib/activation/reglu_tiling.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/reglu/reglu_tiling.h SYMBOLIC)
file(CREATE_LINK ../../lib/activation/reglu_tiling_intf.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/reglu/reglu_tiling_intf.h SYMBOLIC)

# rmsnorm
file(MAKE_DIRECTORY  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/rmsnorm)
file(CREATE_LINK ../../lib/normalization/rmsnorm_tiling.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/rmsnorm/rmsnorm_tiling.h SYMBOLIC)
file(CREATE_LINK ../../lib/normalization/rmsnorm_tiling_intf.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/rmsnorm/rmsnorm_tiling_intf.h SYMBOLIC)
file(CREATE_LINK ../../lib/normalization/rmsnorm_tilingdata.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/rmsnorm/rmsnorm_tilingdata.h SYMBOLIC)

# sigmoid
file(MAKE_DIRECTORY  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/sigmoid)
file(CREATE_LINK ../../lib/activation/sigmoid_tiling.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/sigmoid/sigmoid_tiling.h SYMBOLIC)
file(CREATE_LINK ../../lib/activation/sigmoid_tiling_intf.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/sigmoid/sigmoid_tiling_intf.h SYMBOLIC)

# silu
file(MAKE_DIRECTORY  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/silu)
file(CREATE_LINK ../../lib/activation/silu_tiling_intf.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/silu/silu_tiling_intf.h SYMBOLIC)

# reduce_xor_sum
file(MAKE_DIRECTORY  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/reduce_xor_sum)
file(CREATE_LINK ../../lib/reduce/reduce_xor_sum_tiling.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/reduce_xor_sum/reduce_xor_sum_tiling.h SYMBOLIC)

# softmax
file(MAKE_DIRECTORY  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/softmax)
file(CREATE_LINK ../../lib/activation/softmax_tiling.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/softmax/softmax_tiling.h SYMBOLIC)
file(CREATE_LINK ../../lib/activation/softmax_tiling_intf.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/softmax/softmax_tiling_intf.h SYMBOLIC)
file(CREATE_LINK ../../lib/activation/softmax_tilingdata.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/softmax/softmax_tilingdata.h SYMBOLIC)

file(CREATE_LINK ../../lib/activation/logsoftmax_tiling_intf.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/softmax/logsoftmax_tiling_intf.h SYMBOLIC)
file(CREATE_LINK ../../lib/activation/logsoftmax_tilingdata.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/softmax/logsoftmax_tilingdata.h SYMBOLIC)

# sum
file(MAKE_DIRECTORY  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/sum)
file(CREATE_LINK ../../lib/reduce/sum_tiling_intf.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/sum/sum_tiling_intf.h SYMBOLIC)

# swiglu
file(MAKE_DIRECTORY  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/swiglu)
file(CREATE_LINK ../../lib/activation/swiglu_tiling_intf.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/swiglu/swiglu_tiling_intf.h SYMBOLIC)

# swish
file(MAKE_DIRECTORY  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/swish)
file(CREATE_LINK ../../lib/activation/swish_tiling_intf.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/swish/swish_tiling_intf.h SYMBOLIC)

# topk
file(MAKE_DIRECTORY  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/topk)
file(CREATE_LINK ../../lib/sort/topk_tiling.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/topk/topk_tiling.h SYMBOLIC)
file(CREATE_LINK ../../lib/sort/topk_tilingdata.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/topk/topk_tilingdata.h SYMBOLIC)
file(CREATE_LINK ../../lib/sort/topk_tiling_intf.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/topk/topk_tiling_intf.h SYMBOLIC)

# xor
file(MAKE_DIRECTORY  ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/xor)
file(CREATE_LINK ../../lib/math/xor_tiling_intf.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/xor/xor_tiling_intf.h SYMBOLIC)

file(CREATE_LINK ../lib/tiling_api.h
        ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling/tiling_api.h SYMBOLIC)

file(CREATE_LINK ../ascendc/include/highlevel_api/tiling ${ASCENDC_INSTALL_BASE_PATH}/tikcpp/tiling SYMBOLIC)
