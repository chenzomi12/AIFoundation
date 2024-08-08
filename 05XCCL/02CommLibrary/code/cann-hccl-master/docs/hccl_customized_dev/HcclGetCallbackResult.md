# HcclGetCallbackResult 

## 函数原型<a name="zh-cn_topic_0000001926464500_section2620mcpsimp"></a>

```
HcclResult HcclGetCallbackResult(HcclDispatcher dispatcherPtr)
```

## 功能说明<a name="zh-cn_topic_0000001926464500_section2622mcpsimp"></a>

获取callback执行结果。

## 参数说明<a name="zh-cn_topic_0000001926464500_section2624mcpsimp"></a>

<a name="zh-cn_topic_0000001926464500_table2625mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001926464500_row2631mcpsimp"><th class="cellrowborder" valign="top" width="46%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001926464500_p2633mcpsimp"><a name="zh-cn_topic_0000001926464500_p2633mcpsimp"></a><a name="zh-cn_topic_0000001926464500_p2633mcpsimp"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="22%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001926464500_p2635mcpsimp"><a name="zh-cn_topic_0000001926464500_p2635mcpsimp"></a><a name="zh-cn_topic_0000001926464500_p2635mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="32%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001926464500_p2637mcpsimp"><a name="zh-cn_topic_0000001926464500_p2637mcpsimp"></a><a name="zh-cn_topic_0000001926464500_p2637mcpsimp"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001926464500_row2639mcpsimp"><td class="cellrowborder" valign="top" width="46%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001926464500_p2641mcpsimp"><a name="zh-cn_topic_0000001926464500_p2641mcpsimp"></a><a name="zh-cn_topic_0000001926464500_p2641mcpsimp"></a>HcclDispatcher dispatcher</p>
</td>
<td class="cellrowborder" valign="top" width="22%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001926464500_p2643mcpsimp"><a name="zh-cn_topic_0000001926464500_p2643mcpsimp"></a><a name="zh-cn_topic_0000001926464500_p2643mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="32%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001926464500_p2645mcpsimp"><a name="zh-cn_topic_0000001926464500_p2645mcpsimp"></a><a name="zh-cn_topic_0000001926464500_p2645mcpsimp"></a>dispatcher handle</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001926464500_section2646mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS；其他失败。

