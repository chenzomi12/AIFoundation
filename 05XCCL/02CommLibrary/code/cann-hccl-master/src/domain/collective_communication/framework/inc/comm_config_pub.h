/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_COMM_CONFIG_PUB_H
#define HCCL_COMM_CONFIG_PUB_H

constexpr u32 COMM_CONFIG_MAGIC_WORD = 0xf0f0f0f0;  // Magic word值，用于校验传入的配置结构体是否已经被初始化

// 通信域级别配置参数结构体 - 内部信息
typedef struct CommConfigInfoDef {
    size_t configSize;  // 配置结构体大小
    u32 magicWord;      // Magic word
    u32 version;        // HCCL版本
    char reserved[8];   // 8 byte 保留字段
} CommConfigInfo;

// 通信域级别配置参数结构体 - 外部配置项
typedef struct CommConfigHandleDef {
    CommConfigInfo info;
    u32 bufferSize;     // ccl buffer 大小配置
    u32 deterministic;   // 确定性计算配置
} CommConfigHandle;

namespace hccl {
class CommConfig {
public:
    CommConfig();
    ~CommConfig();

    HcclResult Load(const HcclCommConfig *userConfig, const std::string &id); // 读取通信域配置
    u64 GetConfigBufferSize() const;    // 获取CCL buffer大小配置
    u8 GetConfigDeterministic() const;  // 获取确定性计算配置

private:
    HcclResult CheckMagicWord(const CommConfigHandle& config);      // 检查Magic Word是否合法
    HcclResult SetConfigByVersion(const CommConfigHandle& config);  // 根据版本号读取配置，保证兼容性

    u64 bufferSize_;        // CCL buffer大小配置，单位B
    u8 deterministic_;      // 确定性计算配置：0-关闭，1-开启，其他数字暂时保留
};
}
#endif /* HCCL_COMM_CONFIG_PUB_H */