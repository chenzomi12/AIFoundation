# HcclGetQosCfg 

## 函数原型<a name="zh-cn_topic_0000001953703437_section853mcpsimp"></a>

```
HcclResult HcclGetQosCfg(HcclDispatcher dispatcherPtr, u32 *qosCfg)
```

## 功能说明<a name="zh-cn_topic_0000001953703437_section855mcpsimp"></a>

获取qos cfg。

## 参数说明<a name="zh-cn_topic_0000001953703437_section857mcpsimp"></a>

<a name="zh-cn_topic_0000001953703437_table858mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001953703437_row864mcpsimp"><th class="cellrowborder" valign="top" width="46%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001953703437_p866mcpsimp"><a name="zh-cn_topic_0000001953703437_p866mcpsimp"></a><a name="zh-cn_topic_0000001953703437_p866mcpsimp"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="22%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001953703437_p868mcpsimp"><a name="zh-cn_topic_0000001953703437_p868mcpsimp"></a><a name="zh-cn_topic_0000001953703437_p868mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="32%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001953703437_p870mcpsimp"><a name="zh-cn_topic_0000001953703437_p870mcpsimp"></a><a name="zh-cn_topic_0000001953703437_p870mcpsimp"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001953703437_row872mcpsimp"><td class="cellrowborder" valign="top" width="46%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001953703437_p874mcpsimp"><a name="zh-cn_topic_0000001953703437_p874mcpsimp"></a><a name="zh-cn_topic_0000001953703437_p874mcpsimp"></a>HcclDispatcher dispatcher</p>
</td>
<td class="cellrowborder" valign="top" width="22%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001953703437_p876mcpsimp"><a name="zh-cn_topic_0000001953703437_p876mcpsimp"></a><a name="zh-cn_topic_0000001953703437_p876mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="32%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001953703437_p878mcpsimp"><a name="zh-cn_topic_0000001953703437_p878mcpsimp"></a><a name="zh-cn_topic_0000001953703437_p878mcpsimp"></a>dispatcher handle</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001953703437_row879mcpsimp"><td class="cellrowborder" valign="top" width="46%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001953703437_p881mcpsimp"><a name="zh-cn_topic_0000001953703437_p881mcpsimp"></a><a name="zh-cn_topic_0000001953703437_p881mcpsimp"></a>u32 *qosCfg</p>
</td>
<td class="cellrowborder" valign="top" width="22%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001953703437_p883mcpsimp"><a name="zh-cn_topic_0000001953703437_p883mcpsimp"></a><a name="zh-cn_topic_0000001953703437_p883mcpsimp"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="32%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001953703437_p885mcpsimp"><a name="zh-cn_topic_0000001953703437_p885mcpsimp"></a><a name="zh-cn_topic_0000001953703437_p885mcpsimp"></a>qos cfg</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001953703437_section886mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS；其他失败。

