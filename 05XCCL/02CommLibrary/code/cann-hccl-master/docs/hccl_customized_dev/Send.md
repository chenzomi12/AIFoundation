# Send 

## 函数原型<a name="zh-cn_topic_0000001929299770_section2221mcpsimp"></a>

```
HcclResult Send(const void *data, u64 size)
HcclResult Send(const std::string &sendMsg)
```

## 函数功能<a name="zh-cn_topic_0000001929299770_section2224mcpsimp"></a>

Socket send。

## 参数说明<a name="zh-cn_topic_0000001929299770_section2227mcpsimp"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0000001929299770_table2229mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001929299770_row2236mcpsimp"><th class="cellrowborder" valign="top" width="28.71%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001929299770_p2238mcpsimp"><a name="zh-cn_topic_0000001929299770_p2238mcpsimp"></a><a name="zh-cn_topic_0000001929299770_p2238mcpsimp"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="13.86%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001929299770_p2240mcpsimp"><a name="zh-cn_topic_0000001929299770_p2240mcpsimp"></a><a name="zh-cn_topic_0000001929299770_p2240mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="57.43000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001929299770_p2242mcpsimp"><a name="zh-cn_topic_0000001929299770_p2242mcpsimp"></a><a name="zh-cn_topic_0000001929299770_p2242mcpsimp"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001929299770_row2244mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299770_p2246mcpsimp"><a name="zh-cn_topic_0000001929299770_p2246mcpsimp"></a><a name="zh-cn_topic_0000001929299770_p2246mcpsimp"></a>const void *data</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299770_p2248mcpsimp"><a name="zh-cn_topic_0000001929299770_p2248mcpsimp"></a><a name="zh-cn_topic_0000001929299770_p2248mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299770_p2250mcpsimp"><a name="zh-cn_topic_0000001929299770_p2250mcpsimp"></a><a name="zh-cn_topic_0000001929299770_p2250mcpsimp"></a>数据起始地址</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929299770_row2251mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299770_p2253mcpsimp"><a name="zh-cn_topic_0000001929299770_p2253mcpsimp"></a><a name="zh-cn_topic_0000001929299770_p2253mcpsimp"></a>u64 size</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299770_p2255mcpsimp"><a name="zh-cn_topic_0000001929299770_p2255mcpsimp"></a><a name="zh-cn_topic_0000001929299770_p2255mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299770_p2257mcpsimp"><a name="zh-cn_topic_0000001929299770_p2257mcpsimp"></a><a name="zh-cn_topic_0000001929299770_p2257mcpsimp"></a>数据大小</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929299770_row56491455193115"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299770_p2334mcpsimp"><a name="zh-cn_topic_0000001929299770_p2334mcpsimp"></a><a name="zh-cn_topic_0000001929299770_p2334mcpsimp"></a>const std::string &amp;sendMsg</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299770_p2336mcpsimp"><a name="zh-cn_topic_0000001929299770_p2336mcpsimp"></a><a name="zh-cn_topic_0000001929299770_p2336mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299770_p2338mcpsimp"><a name="zh-cn_topic_0000001929299770_p2338mcpsimp"></a><a name="zh-cn_topic_0000001929299770_p2338mcpsimp"></a>发送数据</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001929299770_section2258mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

## 约束说明<a name="zh-cn_topic_0000001929299770_section2261mcpsimp"></a>

无

