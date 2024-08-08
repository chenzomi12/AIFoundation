# RxData 

## 函数原型<a name="zh-cn_topic_0000001929459302_section7655mcpsimp"></a>

```
HcclResult RxData(UserMemType srcMemType, u64 srcOffset, void *dst, u64 len, Stream &stream)
```

## 函数功能<a name="zh-cn_topic_0000001929459302_section7658mcpsimp"></a>

接收数据。

## 参数说明<a name="zh-cn_topic_0000001929459302_section7661mcpsimp"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0000001929459302_table7663mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001929459302_row7670mcpsimp"><th class="cellrowborder" valign="top" width="28.71%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001929459302_p7672mcpsimp"><a name="zh-cn_topic_0000001929459302_p7672mcpsimp"></a><a name="zh-cn_topic_0000001929459302_p7672mcpsimp"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="13.86%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001929459302_p7674mcpsimp"><a name="zh-cn_topic_0000001929459302_p7674mcpsimp"></a><a name="zh-cn_topic_0000001929459302_p7674mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="57.43000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001929459302_p7676mcpsimp"><a name="zh-cn_topic_0000001929459302_p7676mcpsimp"></a><a name="zh-cn_topic_0000001929459302_p7676mcpsimp"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001929459302_row7678mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929459302_p7680mcpsimp"><a name="zh-cn_topic_0000001929459302_p7680mcpsimp"></a><a name="zh-cn_topic_0000001929459302_p7680mcpsimp"></a>UserMemType srcMemType</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929459302_p7682mcpsimp"><a name="zh-cn_topic_0000001929459302_p7682mcpsimp"></a><a name="zh-cn_topic_0000001929459302_p7682mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929459302_p7684mcpsimp"><a name="zh-cn_topic_0000001929459302_p7684mcpsimp"></a><a name="zh-cn_topic_0000001929459302_p7684mcpsimp"></a>算法step信息</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929459302_row7685mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929459302_p7687mcpsimp"><a name="zh-cn_topic_0000001929459302_p7687mcpsimp"></a><a name="zh-cn_topic_0000001929459302_p7687mcpsimp"></a>u64 srcOffset</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929459302_p7689mcpsimp"><a name="zh-cn_topic_0000001929459302_p7689mcpsimp"></a><a name="zh-cn_topic_0000001929459302_p7689mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929459302_entry7690mcpsimpp0"><a name="zh-cn_topic_0000001929459302_entry7690mcpsimpp0"></a><a name="zh-cn_topic_0000001929459302_entry7690mcpsimpp0"></a>源地址偏移</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929459302_row7691mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929459302_p7693mcpsimp"><a name="zh-cn_topic_0000001929459302_p7693mcpsimp"></a><a name="zh-cn_topic_0000001929459302_p7693mcpsimp"></a>void *dst</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929459302_p7695mcpsimp"><a name="zh-cn_topic_0000001929459302_p7695mcpsimp"></a><a name="zh-cn_topic_0000001929459302_p7695mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929459302_entry7696mcpsimpp0"><a name="zh-cn_topic_0000001929459302_entry7696mcpsimpp0"></a><a name="zh-cn_topic_0000001929459302_entry7696mcpsimpp0"></a>目的地址</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929459302_row7697mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929459302_p7699mcpsimp"><a name="zh-cn_topic_0000001929459302_p7699mcpsimp"></a><a name="zh-cn_topic_0000001929459302_p7699mcpsimp"></a>u64 len</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929459302_p7701mcpsimp"><a name="zh-cn_topic_0000001929459302_p7701mcpsimp"></a><a name="zh-cn_topic_0000001929459302_p7701mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929459302_entry7702mcpsimpp0"><a name="zh-cn_topic_0000001929459302_entry7702mcpsimpp0"></a><a name="zh-cn_topic_0000001929459302_entry7702mcpsimpp0"></a>数据长度</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929459302_row7703mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929459302_p7705mcpsimp"><a name="zh-cn_topic_0000001929459302_p7705mcpsimp"></a><a name="zh-cn_topic_0000001929459302_p7705mcpsimp"></a>Stream &amp;stream</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929459302_p7707mcpsimp"><a name="zh-cn_topic_0000001929459302_p7707mcpsimp"></a><a name="zh-cn_topic_0000001929459302_p7707mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929459302_p7709mcpsimp"><a name="zh-cn_topic_0000001929459302_p7709mcpsimp"></a><a name="zh-cn_topic_0000001929459302_p7709mcpsimp"></a>Stream对象</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001929459302_section7710mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

## 约束说明<a name="zh-cn_topic_0000001929459302_section7713mcpsimp"></a>

无

