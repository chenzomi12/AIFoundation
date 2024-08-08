# LaunchTask 

## 函数原型<a name="zh-cn_topic_0000001953703445_section593mcpsimp"></a>

```
HcclResult LaunchTask(HcclDispatcher dispatcherPtr, hccl::Stream &stream)
```

## 功能说明<a name="zh-cn_topic_0000001953703445_section595mcpsimp"></a>

launch task。

## 参数说明<a name="zh-cn_topic_0000001953703445_section597mcpsimp"></a>

<a name="zh-cn_topic_0000001953703445_table598mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001953703445_row604mcpsimp"><th class="cellrowborder" valign="top" width="46%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001953703445_p606mcpsimp"><a name="zh-cn_topic_0000001953703445_p606mcpsimp"></a><a name="zh-cn_topic_0000001953703445_p606mcpsimp"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="22%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001953703445_p608mcpsimp"><a name="zh-cn_topic_0000001953703445_p608mcpsimp"></a><a name="zh-cn_topic_0000001953703445_p608mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="32%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001953703445_p610mcpsimp"><a name="zh-cn_topic_0000001953703445_p610mcpsimp"></a><a name="zh-cn_topic_0000001953703445_p610mcpsimp"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001953703445_row612mcpsimp"><td class="cellrowborder" valign="top" width="46%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001953703445_p614mcpsimp"><a name="zh-cn_topic_0000001953703445_p614mcpsimp"></a><a name="zh-cn_topic_0000001953703445_p614mcpsimp"></a>HcclDispatcher dispatcher</p>
</td>
<td class="cellrowborder" valign="top" width="22%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001953703445_p616mcpsimp"><a name="zh-cn_topic_0000001953703445_p616mcpsimp"></a><a name="zh-cn_topic_0000001953703445_p616mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="32%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001953703445_p618mcpsimp"><a name="zh-cn_topic_0000001953703445_p618mcpsimp"></a><a name="zh-cn_topic_0000001953703445_p618mcpsimp"></a>dispatcher handle</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001953703445_row619mcpsimp"><td class="cellrowborder" valign="top" width="46%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001953703445_p621mcpsimp"><a name="zh-cn_topic_0000001953703445_p621mcpsimp"></a><a name="zh-cn_topic_0000001953703445_p621mcpsimp"></a>hccl::Stream &amp;stream</p>
</td>
<td class="cellrowborder" valign="top" width="22%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001953703445_p623mcpsimp"><a name="zh-cn_topic_0000001953703445_p623mcpsimp"></a><a name="zh-cn_topic_0000001953703445_p623mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="32%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001953703445_p625mcpsimp"><a name="zh-cn_topic_0000001953703445_p625mcpsimp"></a><a name="zh-cn_topic_0000001953703445_p625mcpsimp"></a>stream对象</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001953703445_section626mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS；其他失败。

