/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALG_OPERATOR_BASE_H
#define ALG_OPERATOR_BASE_H

#include <vector>
#include "hccl_impl.h"
#include "parallel_task_loader.h"
#include "dispatcher.h"
#include "ccl_buffer_manager.h"
#include "hccl_opbase_atrace_info_pub.h"
#include "device_capacity.h"
#include "topo_matcher.h"

#include "coll_alg_param.h"
#include "coll_executor_base.h"
namespace hccl {
struct PreProcessMetaInfo {
    HcclCMDType opType;
    std::vector<u64> inputData;
    u64 inputSize;
    u64 outputSize;
};
constexpr u64 MAX_ALLTOALL_MESH_ALGO_RANK_INTRA_MESH = 16;

class CollAlgOperator {
public:
    CollAlgOperator(std::unique_ptr<hcclImpl> &pImpl, std::unique_ptr<TopoMatcher> &topoMatcher, HcclCMDType opType);
    virtual ~CollAlgOperator() = default;

    virtual HcclResult SelectAlg(const std::string& tag,
        const OpParam& param, std::string& algName, std::string& newTag);
    virtual HcclResult CalcResRequest(const std::string& algName,
        const OpParam& param, AlgResourceRequest& resourceRequest);
    virtual HcclResult Orchestrate(const std::string& algName,
        const OpParam& param, const AlgResourceResponse& algResource);
    // batchsendrecv判断是否需要增量建链
    HcclResult CalcIncreLinkRequest(const std::string& algName, const OpParam& param,
        AlgResourceRequest& resourceRequest);

    virtual bool JudgeIfNeedPreProcessAndGetParam(const OpParam& param,
        std::unique_ptr<PreProcessMetaInfo> &preMetaInfo);
    virtual HcclResult PreparePreOpParam(OpParam& preProcessOpParam,
        const std::unique_ptr<PreProcessMetaInfo> &preMetaInfo, Stream &preProcessStream);
    virtual void SetPreProcessResult(HostMem hostCollectBuffer);
    virtual bool CheckNeedRecreateComm(const std::string& algName, u64 lastScratchMemSize);
    static bool NAFullmeshSatisfyHighPerfAlltoallMeshCondition(DevType deviceType, u32 rankSize);
    static bool FullmeshPairwiseSatisfyHighPerfAlltoallMeshCondition(DevType deviceType, u32 rankSize);

protected:
    bool IsAlgTypeLevel0Mesh(AlgTypeLevel0 &originalAlgTypeLevel0) const;
    std::string GenerateNewTagByAlgTypeLevel1(std::string tag, std::string algTypeLevel1Tag) const;
    HcclResult  AutoSelectAlgTypeLevel1(HcclCMDType hcclCMDType, u64 countSize, u64 cclBufferSize,
                                        std::string &algTypeLevel1Tag, bool isInlineReduce = false,
                                        bool isRdmaReduce = false, bool isAivMode = false);
    /**
     * @brief 获取算法类型的Level0算法
     *
     * @param algType : 算法类型
     * @return AlgTypeLevel0
     */
    AlgTypeLevel0 GetLevel0AlgType(const AlgType algType)const;

    /**
     * @brief 获取算法类型的Level1算法
     *
     * @param algType : 算法类型
     * @return AlgTypeLevel1
     */
    AlgTypeLevel1 GetLevel1AlgType(const AlgType algType) const;
    /**
     * @brief 获取算法类型的Level2算法
     *
     * @param algType : 算法类型
     * @return AlgTypeLevel2
     */
    AlgTypeLevel2 GetLevel2AlgType(const AlgType algType) const;
    /**
     * @brief server间是否使用 ring 算法
     *
     * @param algType : 算法类型
     * @return true : 使用 ring算法
     * @return false 不使用 ring 算法
     */
    bool UseInterServerRingAlgo(AlgType algType);
    /**
     * @brief server间是否使用 hd 算法
     *
     * @param algType : 算法类型
     * @return true : 使用 hd算法
     * @return false 不使用 hd 算法
     */
    bool UseInterServerHDAlgo(AlgType algType);
    /**
     * @brief server间是否使用 NB 算法
     *
     * @param algType : 算法类型
     * @return true : 使用 NB
     * @return false 不使用 NB 算法
     */
    bool UseInterServerNBAlgo(AlgType algType);
    /**
     * @brief server间是否使用 NHR 算法
     *
     * @param algType : 算法类型
     * @return true : 使用 NHR
     * @return false 不使用 NHR 算法
     */
    bool UseInterServerNHRAlgo(AlgType algType);
    /**
     * @brief server间是否使用 NHRV1 算法
     *
     * @param algType : 算法类型
     * @return true : 使用 NHRV1
     * @return false 不使用 NHRV1 算法
     */
    bool UseInterServerNHRV1Algo(AlgType algType);

    /**
     * @brief server间是否使用 pipeline 算法
     *
     * @param algType : 算法类型
     * @return true : 使用 pipeline
     * @return false 不使用 pipeline 算法
     */
    bool UseInterServerPipelineAlgo(AlgType algType);

    /**
     * @brief level2间是否使用 ring 算法
     *
     * @param algType : 算法类型
     * @return true : 使用 ring
     * @return false 不使用 ring 算法
     */
    bool UseLevel2RingAlgo(AlgType algType);

    /**
     * @brief 设置server间使用NHR算法
     *
     * @param algType ：算法类型
     * @return HcclResult
     */
    HcclResult SetInterServerNHRAlgo(AlgType &algType) const;

    /**
     * @brief 设置server间使用ring算法(由于bcast/allgather/reducescatter/reduce/send/recv暂不支持server间ring需继续使用HD)
     *
     * @param algType ：算法类型
     * @return HcclResult
     */
    HcclResult SetInterServerHDAlgo(AlgType &algType) const;
    /**
     * @brief 根据server数量设定server间算法类型
     *
     * @param algType ：算法类型
     * @return HcclResult
     */
    HcclResult SetInterServerRingAlgo(AlgType &algType) const;

    bool Is310P3Common()
    {
        return !isHaveCpuRank_ && !Is310PDevice() && deviceType_ == DevType::DEV_TYPE_310P3;
    }
    AlgType algType_;    // 算法类型
    TopoType topoType_;
    bool isAlgoLevel1Default_ = false;
    bool isHaveCpuRank_;
    bool inlineReduceSwitchOn_;
    std::string identifier_;
    OpMode opMode;

    CCLBufferManager &cclBufferManager_;
    const std::unique_ptr<NotifyPool> &notifyPool_;

    u32 serverNum_;
    u32 moduleNum_;
    u32 devNumInLevel2_;
    u32 deviceNumPerServer_;
    u32 deviceNumPerAggregation_;
    bool multiModuleDiffDeviceNumMode_;
    u32 meshAggregationRankSize_;
    bool isDiffDeviceModule_;
    bool isSingleMeshAggregation_ = false;
    bool meshSinglePlane_ = false;
    bool isAllRankSamePlane_ = false;
    bool is310PDuoCard_;
    bool isSupportRdmaLite_ = false;    // 是否支持rdma lite
    u32 userRank_; // 本group中的userrank
    u32 realUserRank_; // world group中的userrank
    u32 userRankSize_;
    u32 devicePhyId_;
    s32 deviceLogicId_;
    DevType deviceType_;
    std::vector<u32> nicList_;
    std::unordered_map<u32, u32> pairLinkCounter_; // server内所有device间的链路类型计数
    std::vector<RankInfo> &rankInfoList_; // world group内rank的信息, 按照rank id递增依次排列
    std::unique_ptr<hcclImpl> &hcclImpl_;
    std::unique_ptr<CollExecutorBase> executor_;
    std::unique_ptr<TopoMatcher> &topoMatcher_;
    HcclDispatcher dispatcher_; // dispatcher放到最后析构
    HcclDispatcher vDispatcher_; // virtualDispatcher放到最后析构
private:

    virtual HcclResult SelectAlgoTypeForReduceScatter(float delay, u64 recvCurSize, float bandWidth,
        AlgTypeLevel1 &algType);
    HcclResult SelectAlgoTypeForAllGather(float delay, u64 sendCurSize, float bandWidth,
        AlgTypeLevel1 &algType);
    HcclResult SelectAlgoTypeForGather(float delay, u64 sendCurSize, float bandWidth,
        AlgTypeLevel1 &algType);
    HcclResult SelectAlgoTypeForAllReduce(float delay, u64 curSize, float bandWidth,
        AlgTypeLevel1 &algType);
    HcclResult SelectAlgoTypeForBroadcast(float delay, u64 curSize, float bandWidth,
        AlgTypeLevel1 &algType);
    HcclResult SelectAlgoTypeForReduce(float delay, u64 curSize, float bandWidth,
        AlgTypeLevel1 &algType);

    HcclResult AppendTag(const AlgTypeLevel1 &algTypeLevel1, std::string &tag);
    HcclResult SelectAlgoForComm(HcclCMDType hcclCMDType, float delay, u64 curSize, float bandWidth,
        AlgTypeLevel1 &algType);
    HcclResult GetDefaultAlgoLevel1V2(HcclCMDType hcclCMDType, u64 curSize, u64 cclBufferSize,
        AlgTypeLevel1 &algType, bool isInlineReduce = false, bool isRdmaReduce = false, bool isAivMode = false);
    void SetAlgoAttr();
    void SetTopoAttr();
    void SetExecutorAttr();

    std::map<HcclCMDType, std::function<HcclResult(float, u64, float, AlgTypeLevel1 &)>> selectFuncMap_ = {
        {HcclCMDType::HCCL_CMD_REDUCE_SCATTER,
            std::bind(&CollAlgOperator::SelectAlgoTypeForReduceScatter, this,
                std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4)},
        {HcclCMDType::HCCL_CMD_ALLGATHER,
            std::bind(&CollAlgOperator::SelectAlgoTypeForAllGather, this,
                std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4)},
        {HcclCMDType::HCCL_CMD_ALLREDUCE,
            std::bind(&CollAlgOperator::SelectAlgoTypeForAllReduce, this,
                std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4)},
    };
};
}   // namespace hccl

#endif /** __ALG_OPERATOR_BASE_H__ */