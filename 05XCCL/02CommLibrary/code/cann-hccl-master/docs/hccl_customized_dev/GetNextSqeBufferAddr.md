# GetNextSqeBufferAddr 

## 函数原型<a name="zh-cn_topic_0000001963534825_section452mcpsimp"></a>

HcclResult GetNextSqeBufferAddr\(uint8\_t \*&sqeBufferAddr, uint8\_t \*&sqeTypeAddr, uint16\_t &taskId\)

## 函数功能<a name="zh-cn_topic_0000001963534825_section455mcpsimp"></a>

获取sqebuffer。

## 参数说明<a name="zh-cn_topic_0000001963534825_section458mcpsimp"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0000001963534825_table460mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001963534825_row467mcpsimp"><th class="cellrowborder" valign="top" width="28.71287128712871%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001963534825_p469mcpsimp"><a name="zh-cn_topic_0000001963534825_p469mcpsimp"></a><a name="zh-cn_topic_0000001963534825_p469mcpsimp"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="13.861386138613863%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001963534825_p471mcpsimp"><a name="zh-cn_topic_0000001963534825_p471mcpsimp"></a><a name="zh-cn_topic_0000001963534825_p471mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="57.42574257425742%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001963534825_p473mcpsimp"><a name="zh-cn_topic_0000001963534825_p473mcpsimp"></a><a name="zh-cn_topic_0000001963534825_p473mcpsimp"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001963534825_row475mcpsimp"><td class="cellrowborder" valign="top" width="28.71287128712871%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001963534825_p477mcpsimp"><a name="zh-cn_topic_0000001963534825_p477mcpsimp"></a><a name="zh-cn_topic_0000001963534825_p477mcpsimp"></a>uint8_t *&amp;sqeBufferAddr</p>
</td>
<td class="cellrowborder" valign="top" width="13.861386138613863%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001963534825_p479mcpsimp"><a name="zh-cn_topic_0000001963534825_p479mcpsimp"></a><a name="zh-cn_topic_0000001963534825_p479mcpsimp"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="57.42574257425742%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001963534825_p481mcpsimp"><a name="zh-cn_topic_0000001963534825_p481mcpsimp"></a><a name="zh-cn_topic_0000001963534825_p481mcpsimp"></a>sqeBuffer地址</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001963534825_row482mcpsimp"><td class="cellrowborder" valign="top" width="28.71287128712871%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001963534825_p484mcpsimp"><a name="zh-cn_topic_0000001963534825_p484mcpsimp"></a><a name="zh-cn_topic_0000001963534825_p484mcpsimp"></a>uint8_t *&amp;sqeTypeAddr</p>
</td>
<td class="cellrowborder" valign="top" width="13.861386138613863%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001963534825_p486mcpsimp"><a name="zh-cn_topic_0000001963534825_p486mcpsimp"></a><a name="zh-cn_topic_0000001963534825_p486mcpsimp"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="57.42574257425742%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001963534825_p488mcpsimp"><a name="zh-cn_topic_0000001963534825_p488mcpsimp"></a><a name="zh-cn_topic_0000001963534825_p488mcpsimp"></a>sqeType地址</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001963534825_row489mcpsimp"><td class="cellrowborder" valign="top" width="28.71287128712871%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001963534825_p491mcpsimp"><a name="zh-cn_topic_0000001963534825_p491mcpsimp"></a><a name="zh-cn_topic_0000001963534825_p491mcpsimp"></a>uint16_t &amp;taskId</p>
</td>
<td class="cellrowborder" valign="top" width="13.861386138613863%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001963534825_p493mcpsimp"><a name="zh-cn_topic_0000001963534825_p493mcpsimp"></a><a name="zh-cn_topic_0000001963534825_p493mcpsimp"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="57.42574257425742%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001963534825_p495mcpsimp"><a name="zh-cn_topic_0000001963534825_p495mcpsimp"></a><a name="zh-cn_topic_0000001963534825_p495mcpsimp"></a>Task id</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001963534825_section496mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

## 约束说明<a name="zh-cn_topic_0000001963534825_section499mcpsimp"></a>

无

