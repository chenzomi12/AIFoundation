# RegisterInitTaskCallBack 

## 函数原型<a name="zh-cn_topic_0000001966374597_section2620mcpsimp"></a>

```
void RegisterInitTaskCallBack(HcclResult (*p1)(const HcclDispatcher &, hccl::Stream &))
```

## 功能说明<a name="zh-cn_topic_0000001966374597_section2622mcpsimp"></a>

注册init task callback函数。

## 参数说明<a name="zh-cn_topic_0000001966374597_section2624mcpsimp"></a>

<a name="zh-cn_topic_0000001966374597_table2625mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001966374597_row2631mcpsimp"><th class="cellrowborder" valign="top" width="46%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001966374597_p2633mcpsimp"><a name="zh-cn_topic_0000001966374597_p2633mcpsimp"></a><a name="zh-cn_topic_0000001966374597_p2633mcpsimp"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="22%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001966374597_p2635mcpsimp"><a name="zh-cn_topic_0000001966374597_p2635mcpsimp"></a><a name="zh-cn_topic_0000001966374597_p2635mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="32%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001966374597_p2637mcpsimp"><a name="zh-cn_topic_0000001966374597_p2637mcpsimp"></a><a name="zh-cn_topic_0000001966374597_p2637mcpsimp"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001966374597_row2639mcpsimp"><td class="cellrowborder" valign="top" width="46%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001966374597_p671465712419"><a name="zh-cn_topic_0000001966374597_p671465712419"></a><a name="zh-cn_topic_0000001966374597_p671465712419"></a>HcclResult (*p1)(const HcclDispatcher &amp;, hccl::Stream &amp;)</p>
</td>
<td class="cellrowborder" valign="top" width="22%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001966374597_p2643mcpsimp"><a name="zh-cn_topic_0000001966374597_p2643mcpsimp"></a><a name="zh-cn_topic_0000001966374597_p2643mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="32%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001966374597_p166816372514"><a name="zh-cn_topic_0000001966374597_p166816372514"></a><a name="zh-cn_topic_0000001966374597_p166816372514"></a>init task callback函数指针</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001966374597_section2646mcpsimp"></a>

无。

