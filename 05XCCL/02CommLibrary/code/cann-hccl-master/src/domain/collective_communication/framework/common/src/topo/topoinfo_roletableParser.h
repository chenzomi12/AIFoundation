/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOPOINFO_ROLETABLE_H
#define TOPOINFO_ROLETABLE_H

#include "log.h"
#include "topoinfo_ranktableParser_pub.h"

namespace hccl {
class TopoinfoRoletable : public TopoInfoRanktableParser {
public:
    explicit TopoinfoRoletable(const std::string &rankTableM);
    ~TopoinfoRoletable() override;

    HcclResult ParserRoleTable(RoleTableInfo &roleTableInfo);
private:
    HcclResult GetSingleNode(const nlohmann::json &NodeListObj, u32 objIndex,
        std::vector<RoleTableNodeInfo> &nodes);
    HcclResult GetServersInfo(std::vector<RoleTableNodeInfo> &servers);
    HcclResult GetClientsInfo(std::vector<RoleTableNodeInfo> &clients);
};
}  // namespace hccl
#endif  // TOPOINFO_ROLETABLE_H
