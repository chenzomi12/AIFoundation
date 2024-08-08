# alloc

## 函数原型<a name="zh-cn_topic_0000001960344401_section9855mcpsimp"></a>

static DeviceMem alloc\(u64 size, bool level2Address = false\)

## 函数功能<a name="zh-cn_topic_0000001960344401_section9858mcpsimp"></a>

根据输入去申请device内存。

## 参数说明<a name="zh-cn_topic_0000001960344401_section9861mcpsimp"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0000001960344401_table9863mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001960344401_row9870mcpsimp"><th class="cellrowborder" valign="top" width="28.71287128712871%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001960344401_p9872mcpsimp"><a name="zh-cn_topic_0000001960344401_p9872mcpsimp"></a><a name="zh-cn_topic_0000001960344401_p9872mcpsimp"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="13.861386138613863%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001960344401_p9874mcpsimp"><a name="zh-cn_topic_0000001960344401_p9874mcpsimp"></a><a name="zh-cn_topic_0000001960344401_p9874mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="57.42574257425742%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001960344401_p9876mcpsimp"><a name="zh-cn_topic_0000001960344401_p9876mcpsimp"></a><a name="zh-cn_topic_0000001960344401_p9876mcpsimp"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001960344401_row9878mcpsimp"><td class="cellrowborder" valign="top" width="28.71287128712871%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001960344401_p9880mcpsimp"><a name="zh-cn_topic_0000001960344401_p9880mcpsimp"></a><a name="zh-cn_topic_0000001960344401_p9880mcpsimp"></a>u64 size</p>
</td>
<td class="cellrowborder" valign="top" width="13.861386138613863%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001960344401_p9882mcpsimp"><a name="zh-cn_topic_0000001960344401_p9882mcpsimp"></a><a name="zh-cn_topic_0000001960344401_p9882mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.42574257425742%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001960344401_p9884mcpsimp"><a name="zh-cn_topic_0000001960344401_p9884mcpsimp"></a><a name="zh-cn_topic_0000001960344401_p9884mcpsimp"></a>内存大小</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001960344401_row9885mcpsimp"><td class="cellrowborder" valign="top" width="28.71287128712871%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001960344401_p9887mcpsimp"><a name="zh-cn_topic_0000001960344401_p9887mcpsimp"></a><a name="zh-cn_topic_0000001960344401_p9887mcpsimp"></a>bool level2Address</p>
</td>
<td class="cellrowborder" valign="top" width="13.861386138613863%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001960344401_p9889mcpsimp"><a name="zh-cn_topic_0000001960344401_p9889mcpsimp"></a><a name="zh-cn_topic_0000001960344401_p9889mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.42574257425742%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001960344401_p9891mcpsimp"><a name="zh-cn_topic_0000001960344401_p9891mcpsimp"></a><a name="zh-cn_topic_0000001960344401_p9891mcpsimp"></a>是否是二级地址</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001960344401_section9892mcpsimp"></a>

DeviceMem对象。

## 约束说明<a name="zh-cn_topic_0000001960344401_section9895mcpsimp"></a>

无

