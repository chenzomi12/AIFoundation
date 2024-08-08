# TxAck 

## 函数原型<a name="zh-cn_topic_0000001956458793_section7371mcpsimp"></a>

```
HcclResult TxAck(Stream &stream)
```

## 函数功能<a name="zh-cn_topic_0000001956458793_section7374mcpsimp"></a>

本端发送同步信号到对端。

## 参数说明<a name="zh-cn_topic_0000001956458793_section7377mcpsimp"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0000001956458793_table7379mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001956458793_row7386mcpsimp"><th class="cellrowborder" valign="top" width="28.71%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001956458793_p7388mcpsimp"><a name="zh-cn_topic_0000001956458793_p7388mcpsimp"></a><a name="zh-cn_topic_0000001956458793_p7388mcpsimp"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="13.86%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001956458793_p7390mcpsimp"><a name="zh-cn_topic_0000001956458793_p7390mcpsimp"></a><a name="zh-cn_topic_0000001956458793_p7390mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="57.43000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001956458793_p7392mcpsimp"><a name="zh-cn_topic_0000001956458793_p7392mcpsimp"></a><a name="zh-cn_topic_0000001956458793_p7392mcpsimp"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001956458793_row7394mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956458793_p7396mcpsimp"><a name="zh-cn_topic_0000001956458793_p7396mcpsimp"></a><a name="zh-cn_topic_0000001956458793_p7396mcpsimp"></a>Stream &amp;stream</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956458793_p7398mcpsimp"><a name="zh-cn_topic_0000001956458793_p7398mcpsimp"></a><a name="zh-cn_topic_0000001956458793_p7398mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956458793_p7400mcpsimp"><a name="zh-cn_topic_0000001956458793_p7400mcpsimp"></a><a name="zh-cn_topic_0000001956458793_p7400mcpsimp"></a>Stream对象</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001956458793_section7401mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

## 约束说明<a name="zh-cn_topic_0000001956458793_section7404mcpsimp"></a>

无

