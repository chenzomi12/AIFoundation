# DataReceivedAck

## 函数原型<a name="zh-cn_topic_0000001929459294_section7334mcpsimp"></a>

```
HcclResult DataReceivedAck(Stream &stream)
```

## 函数功能<a name="zh-cn_topic_0000001929459294_section7337mcpsimp"></a>

接收数据后，发送同步信号到对端。

## 参数说明<a name="zh-cn_topic_0000001929459294_section7340mcpsimp"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0000001929459294_table7342mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001929459294_row7349mcpsimp"><th class="cellrowborder" valign="top" width="28.71%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001929459294_p7351mcpsimp"><a name="zh-cn_topic_0000001929459294_p7351mcpsimp"></a><a name="zh-cn_topic_0000001929459294_p7351mcpsimp"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="13.86%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001929459294_p7353mcpsimp"><a name="zh-cn_topic_0000001929459294_p7353mcpsimp"></a><a name="zh-cn_topic_0000001929459294_p7353mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="57.43000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001929459294_p7355mcpsimp"><a name="zh-cn_topic_0000001929459294_p7355mcpsimp"></a><a name="zh-cn_topic_0000001929459294_p7355mcpsimp"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001929459294_row7357mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929459294_p7359mcpsimp"><a name="zh-cn_topic_0000001929459294_p7359mcpsimp"></a><a name="zh-cn_topic_0000001929459294_p7359mcpsimp"></a>Stream &amp;stream</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929459294_p7361mcpsimp"><a name="zh-cn_topic_0000001929459294_p7361mcpsimp"></a><a name="zh-cn_topic_0000001929459294_p7361mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929459294_p7363mcpsimp"><a name="zh-cn_topic_0000001929459294_p7363mcpsimp"></a><a name="zh-cn_topic_0000001929459294_p7363mcpsimp"></a>Stream对象</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001929459294_section7364mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

## 约束说明<a name="zh-cn_topic_0000001929459294_section7367mcpsimp"></a>

无

