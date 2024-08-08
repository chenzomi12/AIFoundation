/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "externalinput_pub.h"
#include "comm_config_pub.h"
#include "adapter_error_manager_pub.h"

namespace hccl {
CommConfig::CommConfig()
    : bufferSize_(GetExternalInputCCLBuffSize()), deterministic_(static_cast<u8>(GetExternalInputHcclDeterministic()))
{}

CommConfig::~CommConfig() {}

HcclResult CommConfig::Load(const HcclCommConfig *userConfig, const std::string &id)
{
    // 检查是否为空
    CHK_PTR_NULL(userConfig);
    
    // 读取结构体的size
    const size_t configSize = *(reinterpret_cast<const size_t *>(userConfig));
    HCCL_INFO("[Load] config size[%llu]", configSize);

    CommConfigHandle configHandle;

    // 根据size读取结构体
    s32 sRet = memcpy_s(&configHandle, sizeof(CommConfigHandle), userConfig, configSize);
    CHK_PRT_RET(sRet != EOK,
        HCCL_ERROR("[Load] memcpy comm config fail. errorno[%d] "
        "params:destMaxSize[%u], count[%u]",
        sRet, sizeof(CommConfigHandle), configSize),
        HCCL_E_MEMORY);

    HCCL_INFO("[Load] comm config info of [%s]: configSize[%llu], version[%u]", id.c_str(),
        configHandle.info.configSize, configHandle.info.version);

    // 检查Magic word是否合法
    CHK_RET(CheckMagicWord(configHandle));

    // 根据版本号读取配置，检查配置参数合法性
    CHK_RET(SetConfigByVersion(configHandle));

    HCCL_INFO("[Load] comm config of [%s]: bufferSize[%llu], deterministic[%u]", id.c_str(), bufferSize_,
        deterministic_);

    return HCCL_SUCCESS;
}

HcclResult CommConfig::CheckMagicWord(const CommConfigHandle &config)
{
    if (config.info.magicWord != COMM_CONFIG_MAGIC_WORD) {
        HCCL_ERROR("[CheckMagicWord] Invalid magic word[0x%x]. Please make sure the config has been initialized by "
            "HcclCommConfigInit().",
            config.info.magicWord);
        return HCCL_E_PARA;
    }

    return HCCL_SUCCESS;
}

HcclResult CommConfig::SetConfigByVersion(const CommConfigHandle &config)
{
    // 版本大于等于1，设置CCL buffer、确定性计算配置
    if (config.info.version >= 1) {
        if (config.bufferSize < HCCL_CCL_COMM_BUFFER_MIN) {
            RPT_INPUT_ERR(true, "EI0003", std::vector<std::string>({ "ccl_op", "parameter", "value", "tips" }),
                std::vector<std::string>({
                    "HcclCommInitRootInfoConfig",
                    "hcclBufferSize",
                    std::to_string(config.bufferSize),
                    "Value should be equal to or greater than 1(MB)."
                })
            );
            HCCL_ERROR("[SetConfigByVersion] The configuration of hcclBufferSize[%u(MB)] is invalid, which should be "
                       "greater than %u(MB).",
                config.bufferSize, HCCL_CCL_COMM_BUFFER_MIN);
            return HCCL_E_PARA;
        }
        bufferSize_ = config.bufferSize * HCCL_CCL_COMM_FIXED_CALC_BUFFER_SIZE; // MByte 转 Byte

        if (config.deterministic > 1) {
            RPT_INPUT_ERR(true, "EI0003", std::vector<std::string>({ "ccl_op", "parameter", "value", "tips" }),
                std::vector<std::string>({
                    "HcclCommInitRootInfoConfig",
                    "hcclDeterministic",
                    std::to_string(config.deterministic),
                    "Value should be 0(disable) or 1(enable)."
                })
            );
            HCCL_ERROR(
                "[SetConfigByVersion] The configuration of hcclDeterministic[%u] is invalid, which should be 0 or 1.",
                config.deterministic);
            return HCCL_E_PARA;
        }

        deterministic_ = static_cast<u8>(config.deterministic);     // 前面已保证数值不超过UINT8_MAX，直接进行类型转换
    }

    return HCCL_SUCCESS;
}

u64 CommConfig::GetConfigBufferSize() const
{
    return bufferSize_;
}

u8 CommConfig::GetConfigDeterministic() const
{
    return deterministic_;
}
}