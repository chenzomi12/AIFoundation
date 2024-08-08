# IsCtxInitialized 

## 函数原型<a name="zh-cn_topic_0000001926623852_section1752mcpsimp"></a>

```
HcclResult IsCtxInitialized(HcclDispatcher dispatcherPtr, bool *ctxInitFlag)
```

## 功能说明<a name="zh-cn_topic_0000001926623852_section1754mcpsimp"></a>

task是否初始化。

## 参数说明<a name="zh-cn_topic_0000001926623852_section1756mcpsimp"></a>

<a name="zh-cn_topic_0000001926623852_table1757mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001926623852_row1763mcpsimp"><th class="cellrowborder" valign="top" width="46%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001926623852_p1765mcpsimp"><a name="zh-cn_topic_0000001926623852_p1765mcpsimp"></a><a name="zh-cn_topic_0000001926623852_p1765mcpsimp"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="22%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001926623852_p1767mcpsimp"><a name="zh-cn_topic_0000001926623852_p1767mcpsimp"></a><a name="zh-cn_topic_0000001926623852_p1767mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="32%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001926623852_p1769mcpsimp"><a name="zh-cn_topic_0000001926623852_p1769mcpsimp"></a><a name="zh-cn_topic_0000001926623852_p1769mcpsimp"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001926623852_row1771mcpsimp"><td class="cellrowborder" valign="top" width="46%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001926623852_p1773mcpsimp"><a name="zh-cn_topic_0000001926623852_p1773mcpsimp"></a><a name="zh-cn_topic_0000001926623852_p1773mcpsimp"></a>HcclDispatcher dispatcher</p>
</td>
<td class="cellrowborder" valign="top" width="22%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001926623852_p1775mcpsimp"><a name="zh-cn_topic_0000001926623852_p1775mcpsimp"></a><a name="zh-cn_topic_0000001926623852_p1775mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="32%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001926623852_p1777mcpsimp"><a name="zh-cn_topic_0000001926623852_p1777mcpsimp"></a><a name="zh-cn_topic_0000001926623852_p1777mcpsimp"></a>dispatcher handle</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001926623852_row1778mcpsimp"><td class="cellrowborder" valign="top" width="46%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001926623852_p1780mcpsimp"><a name="zh-cn_topic_0000001926623852_p1780mcpsimp"></a><a name="zh-cn_topic_0000001926623852_p1780mcpsimp"></a>bool *ctxInitFlag</p>
</td>
<td class="cellrowborder" valign="top" width="22%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001926623852_p1782mcpsimp"><a name="zh-cn_topic_0000001926623852_p1782mcpsimp"></a><a name="zh-cn_topic_0000001926623852_p1782mcpsimp"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="32%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001926623852_p1784mcpsimp"><a name="zh-cn_topic_0000001926623852_p1784mcpsimp"></a><a name="zh-cn_topic_0000001926623852_p1784mcpsimp"></a>初始化标识</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001926623852_section1785mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS；其他失败。

