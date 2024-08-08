# TxEnv 

## 函数原型<a name="zh-cn_topic_0000001956458817_section8168mcpsimp"></a>

HcclResult TxEnv\(const void \*ptr, const u64 len, Stream &stream\)

## 函数功能<a name="zh-cn_topic_0000001956458817_section8171mcpsimp"></a>

发送前信息准备。

## 参数说明<a name="zh-cn_topic_0000001956458817_section8174mcpsimp"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0000001956458817_table8176mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001956458817_row8183mcpsimp"><th class="cellrowborder" valign="top" width="28.71%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001956458817_p8185mcpsimp"><a name="zh-cn_topic_0000001956458817_p8185mcpsimp"></a><a name="zh-cn_topic_0000001956458817_p8185mcpsimp"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="13.86%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001956458817_p8187mcpsimp"><a name="zh-cn_topic_0000001956458817_p8187mcpsimp"></a><a name="zh-cn_topic_0000001956458817_p8187mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="57.43000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001956458817_p8189mcpsimp"><a name="zh-cn_topic_0000001956458817_p8189mcpsimp"></a><a name="zh-cn_topic_0000001956458817_p8189mcpsimp"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001956458817_row8191mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956458817_p8193mcpsimp"><a name="zh-cn_topic_0000001956458817_p8193mcpsimp"></a><a name="zh-cn_topic_0000001956458817_p8193mcpsimp"></a>const void *ptr</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956458817_p8195mcpsimp"><a name="zh-cn_topic_0000001956458817_p8195mcpsimp"></a><a name="zh-cn_topic_0000001956458817_p8195mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956458817_p8197mcpsimp"><a name="zh-cn_topic_0000001956458817_p8197mcpsimp"></a><a name="zh-cn_topic_0000001956458817_p8197mcpsimp"></a>算法step信息</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001956458817_row8198mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956458817_p8200mcpsimp"><a name="zh-cn_topic_0000001956458817_p8200mcpsimp"></a><a name="zh-cn_topic_0000001956458817_p8200mcpsimp"></a>const u64 len</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956458817_p8202mcpsimp"><a name="zh-cn_topic_0000001956458817_p8202mcpsimp"></a><a name="zh-cn_topic_0000001956458817_p8202mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956458817_entry8203mcpsimpp0"><a name="zh-cn_topic_0000001956458817_entry8203mcpsimpp0"></a><a name="zh-cn_topic_0000001956458817_entry8203mcpsimpp0"></a>数据长度</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001956458817_row8204mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956458817_p8206mcpsimp"><a name="zh-cn_topic_0000001956458817_p8206mcpsimp"></a><a name="zh-cn_topic_0000001956458817_p8206mcpsimp"></a>Stream &amp;stream</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956458817_p8208mcpsimp"><a name="zh-cn_topic_0000001956458817_p8208mcpsimp"></a><a name="zh-cn_topic_0000001956458817_p8208mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956458817_p8210mcpsimp"><a name="zh-cn_topic_0000001956458817_p8210mcpsimp"></a><a name="zh-cn_topic_0000001956458817_p8210mcpsimp"></a>Stream对象</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001956458817_section8211mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

## 约束说明<a name="zh-cn_topic_0000001956458817_section8214mcpsimp"></a>

无

