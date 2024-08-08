# TxData 

## 函数原型<a name="zh-cn_topic_0000001929299954_section7593mcpsimp"></a>

```
HcclResult TxData(UserMemType dstMemType, u64 dstOffset, const void *src, u64 len, Stream &stream)
```

## 函数功能<a name="zh-cn_topic_0000001929299954_section7596mcpsimp"></a>

发送数据。

## 参数说明<a name="zh-cn_topic_0000001929299954_section7599mcpsimp"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0000001929299954_table7601mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001929299954_row7608mcpsimp"><th class="cellrowborder" valign="top" width="28.71%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001929299954_p7610mcpsimp"><a name="zh-cn_topic_0000001929299954_p7610mcpsimp"></a><a name="zh-cn_topic_0000001929299954_p7610mcpsimp"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="13.86%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001929299954_p7612mcpsimp"><a name="zh-cn_topic_0000001929299954_p7612mcpsimp"></a><a name="zh-cn_topic_0000001929299954_p7612mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="57.43000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001929299954_p7614mcpsimp"><a name="zh-cn_topic_0000001929299954_p7614mcpsimp"></a><a name="zh-cn_topic_0000001929299954_p7614mcpsimp"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001929299954_row7616mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299954_p7618mcpsimp"><a name="zh-cn_topic_0000001929299954_p7618mcpsimp"></a><a name="zh-cn_topic_0000001929299954_p7618mcpsimp"></a>UserMemType dstMemType</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299954_p7620mcpsimp"><a name="zh-cn_topic_0000001929299954_p7620mcpsimp"></a><a name="zh-cn_topic_0000001929299954_p7620mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299954_p7622mcpsimp"><a name="zh-cn_topic_0000001929299954_p7622mcpsimp"></a><a name="zh-cn_topic_0000001929299954_p7622mcpsimp"></a>算法step信息</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929299954_row7623mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299954_p7625mcpsimp"><a name="zh-cn_topic_0000001929299954_p7625mcpsimp"></a><a name="zh-cn_topic_0000001929299954_p7625mcpsimp"></a>u64 dstOffset</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299954_p7627mcpsimp"><a name="zh-cn_topic_0000001929299954_p7627mcpsimp"></a><a name="zh-cn_topic_0000001929299954_p7627mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299954_entry7628mcpsimpp0"><a name="zh-cn_topic_0000001929299954_entry7628mcpsimpp0"></a><a name="zh-cn_topic_0000001929299954_entry7628mcpsimpp0"></a>目的偏移</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929299954_row7629mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299954_p7631mcpsimp"><a name="zh-cn_topic_0000001929299954_p7631mcpsimp"></a><a name="zh-cn_topic_0000001929299954_p7631mcpsimp"></a>const void *src</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299954_p7633mcpsimp"><a name="zh-cn_topic_0000001929299954_p7633mcpsimp"></a><a name="zh-cn_topic_0000001929299954_p7633mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299954_entry7634mcpsimpp0"><a name="zh-cn_topic_0000001929299954_entry7634mcpsimpp0"></a><a name="zh-cn_topic_0000001929299954_entry7634mcpsimpp0"></a>源地址</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929299954_row7635mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299954_p7637mcpsimp"><a name="zh-cn_topic_0000001929299954_p7637mcpsimp"></a><a name="zh-cn_topic_0000001929299954_p7637mcpsimp"></a>u64 len</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299954_p7639mcpsimp"><a name="zh-cn_topic_0000001929299954_p7639mcpsimp"></a><a name="zh-cn_topic_0000001929299954_p7639mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299954_entry7640mcpsimpp0"><a name="zh-cn_topic_0000001929299954_entry7640mcpsimpp0"></a><a name="zh-cn_topic_0000001929299954_entry7640mcpsimpp0"></a>数据长度</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929299954_row7641mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299954_p7643mcpsimp"><a name="zh-cn_topic_0000001929299954_p7643mcpsimp"></a><a name="zh-cn_topic_0000001929299954_p7643mcpsimp"></a>Stream &amp;stream</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299954_p7645mcpsimp"><a name="zh-cn_topic_0000001929299954_p7645mcpsimp"></a><a name="zh-cn_topic_0000001929299954_p7645mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299954_p7647mcpsimp"><a name="zh-cn_topic_0000001929299954_p7647mcpsimp"></a><a name="zh-cn_topic_0000001929299954_p7647mcpsimp"></a>Stream对象</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001929299954_section7648mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

## 约束说明<a name="zh-cn_topic_0000001929299954_section7651mcpsimp"></a>

无

