/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

class SearchPath {
public:
    std::vector<u32> Search(const std::vector<u32> &nicList)
    {
        if (nicList.size() == 0 || nicList.size() == 1) {
            return nicList;
        }

        result_.clear();
        std::vector<bool> arrived(nicList.size(), 0);
        nicSet_.clear();
        for (auto i : nicList) {
            nicSet_.insert(i);
        }

        if (dfs(nicList, arrived, 0, nicList[0])) {
            return result_;
        } else {
            return {};
        }
    }

private:
    bool dfs(const std::vector<u32> &nicList, std::vector<bool> &arrived, u32 idx, u32 nowNicIdx)
    {
        // the last nic, it must reachable to result_[0]
        if (idx == nicList.size() - 1) {
            for (auto i : reachableRank_[nowNicIdx]) {
                if (i == result_[0]) {
                    result_.push_back(nowNicIdx);
                    return true;
                }
            }

            return false;
        }

        arrived[nowNicIdx] = true;
        result_.push_back(nowNicIdx);

        for (auto i : reachableRank_[nowNicIdx]) {
            if (nicSet_.count(i) == 0 || arrived[i]) {
                continue;
            }

            if (dfs(nicList, arrived, idx + 1, i)) {
                return true;
            }
        }

        arrived[nowNicIdx] = false;
        result_.pop_back();
        return false;
    }

    std::vector<u32> result_;
    std::set<u32> nicSet_;
    // 适配910_73设备双轨组网，可通过SIO串联
    std::map<int, std::vector<u32>> reachableRank_ = {
    {0, {1, 2, 4, 6, 8, 10, 12, 14}},
    {1, {0, 3, 5, 7, 9, 11, 13, 15}},
    {2, {3, 4, 6, 8, 10, 12, 14, 0}},
    {3, {2, 5, 7, 9, 11, 13, 15, 1}},
    {4, {5, 6, 8, 10, 12, 14, 0, 2}},
    {5, {4, 7, 9, 11, 13, 15, 1, 3}},
    {6, {7, 8, 10, 12, 14, 0, 2, 4}},
    {7, {6, 9, 11, 13, 15, 1, 3, 5}},
    {8, {9, 10, 12, 14, 0, 2, 4, 6}},
    {9, {8, 11, 13, 15, 1, 3, 5, 7}},
    {10, {11, 12, 14, 0, 2, 4, 6, 8}},
    {11, {10, 13, 15, 1, 3, 5, 7, 9}},
    {12, {13, 14, 0, 2, 4, 6, 8, 10}},
    {13, {12, 15, 1, 3, 5, 7, 9, 11}},
    {14, {15, 0, 2, 4, 6, 8, 10, 12}},
    {15, {14, 1, 3, 5, 7, 9, 11, 13}}};
};