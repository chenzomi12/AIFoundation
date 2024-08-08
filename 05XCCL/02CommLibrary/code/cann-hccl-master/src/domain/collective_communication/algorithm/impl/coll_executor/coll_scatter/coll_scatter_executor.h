/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_SCATTER_EXECUTOR_H
#define COLL_SCATTER_EXECUTOR_H

#include "coll_native_executor_base.h"
#include "coll_alg_exec_registry.h"

namespace hccl {

// 所有 Scatter Executor 的基类，继承自 NativeExecutor
class CollScatterExecutor : public CollNativeExecutorBase {
public:
    explicit CollScatterExecutor(const HcclDispatcher dispatcher,
                                std::unique_ptr<TopoMatcher> &topoMatcher);
    ~CollScatterExecutor() = default;

    HcclResult Orchestrate(const OpParam& param, const AlgResourceResponse& algRes) override;
protected:
    /* *************** 资源计算 *************** */
    virtual HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport);
    virtual HcclResult CalcTransportMemType(TransportMemType &inputType, TransportMemType &outputType);

    /* *************** 算法编排 *************** */
    // 按Inner、Outer、Level2可继续进行拆分。
    virtual HcclResult KernelRunInner(DeviceMem &inputMem, u64 count, HcclDataType dataType, u32 &commIndex,
        u32 root, u32 &subRoot, CommPlane commLevel, Stream &stream);
    // 用于需要Loop的Executor
    virtual HcclResult RunLoop(const OpParam &param, const AlgResourceResponse &algRes);
    virtual HcclResult RunLoopInner(const OpParam &param, ExecMem &execMem, const AlgResourceResponse &algRes);

    virtual bool IsHugeData(u64 curSize);
    /* *************** 通用工具 *************** */
    virtual HcclResult PrepareDataSlice(u64 dataCount, u32 unitSize, u32 sliceNum,
        std::vector<Slice> &dataSlice);
    virtual HcclResult ReorderSlice(std::vector<Slice> &dataSlice, std::vector<u32> &order);
private:
};

} // namespace hccl

#endif
