# alloc

## 函数原型<a name="zh-cn_topic_0000001933265292_section10036mcpsimp"></a>

static HostMem alloc\(u64 size, bool isRtsMem = true\)

## 函数功能<a name="zh-cn_topic_0000001933265292_section10039mcpsimp"></a>

根据输入去申请device内存。

## 参数说明<a name="zh-cn_topic_0000001933265292_section10042mcpsimp"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0000001933265292_table10044mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001933265292_row10051mcpsimp"><th class="cellrowborder" valign="top" width="28.71%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001933265292_p10053mcpsimp"><a name="zh-cn_topic_0000001933265292_p10053mcpsimp"></a><a name="zh-cn_topic_0000001933265292_p10053mcpsimp"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="13.86%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001933265292_p10055mcpsimp"><a name="zh-cn_topic_0000001933265292_p10055mcpsimp"></a><a name="zh-cn_topic_0000001933265292_p10055mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="57.43000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001933265292_p10057mcpsimp"><a name="zh-cn_topic_0000001933265292_p10057mcpsimp"></a><a name="zh-cn_topic_0000001933265292_p10057mcpsimp"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001933265292_row10059mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001933265292_p10061mcpsimp"><a name="zh-cn_topic_0000001933265292_p10061mcpsimp"></a><a name="zh-cn_topic_0000001933265292_p10061mcpsimp"></a>u64 size,</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001933265292_p10063mcpsimp"><a name="zh-cn_topic_0000001933265292_p10063mcpsimp"></a><a name="zh-cn_topic_0000001933265292_p10063mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001933265292_p10065mcpsimp"><a name="zh-cn_topic_0000001933265292_p10065mcpsimp"></a><a name="zh-cn_topic_0000001933265292_p10065mcpsimp"></a>内存大小</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001933265292_row10066mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001933265292_p10068mcpsimp"><a name="zh-cn_topic_0000001933265292_p10068mcpsimp"></a><a name="zh-cn_topic_0000001933265292_p10068mcpsimp"></a>bool isRtsMem</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001933265292_p10070mcpsimp"><a name="zh-cn_topic_0000001933265292_p10070mcpsimp"></a><a name="zh-cn_topic_0000001933265292_p10070mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001933265292_p10072mcpsimp"><a name="zh-cn_topic_0000001933265292_p10072mcpsimp"></a><a name="zh-cn_topic_0000001933265292_p10072mcpsimp"></a>是否通过rts申请</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001933265292_section10073mcpsimp"></a>

Host Mem对象。

## 约束说明<a name="zh-cn_topic_0000001933265292_section10076mcpsimp"></a>

无

