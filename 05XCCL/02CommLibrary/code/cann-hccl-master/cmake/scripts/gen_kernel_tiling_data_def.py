#!/usr/bin/env python
# coding=utf-8

# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import sys
import os
import re


def gen_tiling(tiling_header_file):
    single_tiling_source = ""
    if not os.path.exists(tiling_header_file):
        print("warning: no userdef tiling header file: ", tiling_header_file)
        return single_tiling_source
    print("generate tiling def header file: ", tiling_header_file)
    pattern = re.compile(r'[(](.*)[)]', re.S)
    with open(tiling_header_file, 'r') as fd:
        lines = fd.readlines()
        for line in lines:
            line = line.strip()
            if (line.startswith('BEGIN_TILING_DATA_DEF')):
                single_tiling_source += '#pragma pack(push, 8)\n'
                single_tiling_source += 'struct '
                struct_def = re.findall(pattern, line)[0]
                single_tiling_source += struct_def + ' {\n'
            elif (line.startswith('TILING_DATA_FIELD_DEF_ARR')):
                field_params = re.findall(pattern, line)[0]
                fds = field_params.split(',')
                single_tiling_source += '    {} {}[{}] = {{}};\n'.format(fds[0].strip(), fds[2].strip(), fds[1].strip())
            elif (line.startswith('TILING_DATA_FIELD_DEF_STRUCT')):
                field_params = re.findall(pattern, line)[0]
                fds = field_params.split(',')
                single_tiling_source += '    {} {};\n'.format(fds[0].strip(), fds[1].strip())
            elif (line.startswith('TILING_DATA_FIELD_DEF')):
                field_params = re.findall(pattern, line)[0]
                fds = field_params.split(',')
                single_tiling_source += '    {} {} = 0;\n'.format(fds[0].strip(), fds[1].strip())
            elif (line.startswith('END_TILING_DATA_DEF')):
                single_tiling_source += '};\n'
                single_tiling_source += '#pragma pack(pop)\n'
    return single_tiling_source



if __name__ == '__main__':
    if len(sys.argv) <= 2:
        raise RuntimeError('arguments must greater than 2')
    res = """#ifndef __TIKCFW_KERNEL_TILING_H_
#define __TIKCFW_KERNEL_TILING_H_

#if defined(ASCENDC_CPU_DEBUG)
#include <cstdint>
#include <cstring>
#endif

"""
    print("[LOG]:  ", sys.argv[1], sys.argv[2])
    file_list = []
    for root, files in os.walk(sys.argv[1]):
        for file in files:
            if file.endswith("tilingdata.h"):
                file_list.append(os.path.join(root, file))
    file_list.sort()
    for file in file_list:
        res += gen_tiling(file)
    res += '#endif\n'

    generate_file = sys.argv[2]
    absolute_file = os.path.abspath(generate_file)
    generate_dir = os.path.dirname(generate_file)
    if not os.path.exists(generate_dir):
        os.makedirs(generate_dir, exist_ok=True)

    with os.fdopen(os.open(absolute_file, os.O_RDWR | os.O_CREAT | os.O_TRUNC), 'w') as ofd:
        ofd.write(res)
