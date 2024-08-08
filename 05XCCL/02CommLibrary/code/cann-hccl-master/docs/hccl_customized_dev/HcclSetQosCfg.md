# HcclSetQosCfg 

## 函数原型<a name="zh-cn_topic_0000001926623840_section2104mcpsimp"></a>

```
HcclResult HcclSetQosCfg(HcclDispatcher dispatcherPtr, const u32 qosCfg)
```

## 功能说明<a name="zh-cn_topic_0000001926623840_section2106mcpsimp"></a>

设置qos cfg。

## 参数说明<a name="zh-cn_topic_0000001926623840_section2108mcpsimp"></a>

<a name="zh-cn_topic_0000001926623840_table2109mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001926623840_row2115mcpsimp"><th class="cellrowborder" valign="top" width="46%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001926623840_p2117mcpsimp"><a name="zh-cn_topic_0000001926623840_p2117mcpsimp"></a><a name="zh-cn_topic_0000001926623840_p2117mcpsimp"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="22%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001926623840_p2119mcpsimp"><a name="zh-cn_topic_0000001926623840_p2119mcpsimp"></a><a name="zh-cn_topic_0000001926623840_p2119mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="32%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001926623840_p2121mcpsimp"><a name="zh-cn_topic_0000001926623840_p2121mcpsimp"></a><a name="zh-cn_topic_0000001926623840_p2121mcpsimp"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001926623840_row2123mcpsimp"><td class="cellrowborder" valign="top" width="46%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001926623840_p2125mcpsimp"><a name="zh-cn_topic_0000001926623840_p2125mcpsimp"></a><a name="zh-cn_topic_0000001926623840_p2125mcpsimp"></a>HcclDispatcher dispatcher</p>
</td>
<td class="cellrowborder" valign="top" width="22%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001926623840_p2127mcpsimp"><a name="zh-cn_topic_0000001926623840_p2127mcpsimp"></a><a name="zh-cn_topic_0000001926623840_p2127mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="32%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001926623840_p2129mcpsimp"><a name="zh-cn_topic_0000001926623840_p2129mcpsimp"></a><a name="zh-cn_topic_0000001926623840_p2129mcpsimp"></a>dispatcher handle</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001926623840_row2130mcpsimp"><td class="cellrowborder" valign="top" width="46%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001926623840_p2132mcpsimp"><a name="zh-cn_topic_0000001926623840_p2132mcpsimp"></a><a name="zh-cn_topic_0000001926623840_p2132mcpsimp"></a>const u32 qosCfg</p>
</td>
<td class="cellrowborder" valign="top" width="22%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001926623840_p2134mcpsimp"><a name="zh-cn_topic_0000001926623840_p2134mcpsimp"></a><a name="zh-cn_topic_0000001926623840_p2134mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="32%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001926623840_p2136mcpsimp"><a name="zh-cn_topic_0000001926623840_p2136mcpsimp"></a><a name="zh-cn_topic_0000001926623840_p2136mcpsimp"></a>qos cfg</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001926623840_section2137mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS；其他失败。

