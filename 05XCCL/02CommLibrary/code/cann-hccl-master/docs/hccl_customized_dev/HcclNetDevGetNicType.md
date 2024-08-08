# HcclNetDevGetNicType 

## 函数原型<a name="zh-cn_topic_0000001956458601_section1704mcpsimp"></a>

```
HcclResult HcclNetDevGetNicType(HcclNetDevCtx netDevCtx, NicType *nicType)
```

## 函数功能<a name="zh-cn_topic_0000001956458601_section1707mcpsimp"></a>

获取网卡类型。

## 参数说明<a name="zh-cn_topic_0000001956458601_section1710mcpsimp"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0000001956458601_table1712mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001956458601_row1719mcpsimp"><th class="cellrowborder" valign="top" width="28.71%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001956458601_p1721mcpsimp"><a name="zh-cn_topic_0000001956458601_p1721mcpsimp"></a><a name="zh-cn_topic_0000001956458601_p1721mcpsimp"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="13.86%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001956458601_p1723mcpsimp"><a name="zh-cn_topic_0000001956458601_p1723mcpsimp"></a><a name="zh-cn_topic_0000001956458601_p1723mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="57.43000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001956458601_p1725mcpsimp"><a name="zh-cn_topic_0000001956458601_p1725mcpsimp"></a><a name="zh-cn_topic_0000001956458601_p1725mcpsimp"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001956458601_row1727mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956458601_p1729mcpsimp"><a name="zh-cn_topic_0000001956458601_p1729mcpsimp"></a><a name="zh-cn_topic_0000001956458601_p1729mcpsimp"></a>HcclNetDevCtx *netDevCtx</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956458601_p1731mcpsimp"><a name="zh-cn_topic_0000001956458601_p1731mcpsimp"></a><a name="zh-cn_topic_0000001956458601_p1731mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956458601_p1733mcpsimp"><a name="zh-cn_topic_0000001956458601_p1733mcpsimp"></a><a name="zh-cn_topic_0000001956458601_p1733mcpsimp"></a>网卡设备handle</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001956458601_row1734mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956458601_p1736mcpsimp"><a name="zh-cn_topic_0000001956458601_p1736mcpsimp"></a><a name="zh-cn_topic_0000001956458601_p1736mcpsimp"></a>NicType *nicType</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956458601_p1738mcpsimp"><a name="zh-cn_topic_0000001956458601_p1738mcpsimp"></a><a name="zh-cn_topic_0000001956458601_p1738mcpsimp"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956458601_p1740mcpsimp"><a name="zh-cn_topic_0000001956458601_p1740mcpsimp"></a><a name="zh-cn_topic_0000001956458601_p1740mcpsimp"></a>网卡类型</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001956458601_section1741mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

## 约束说明<a name="zh-cn_topic_0000001956458601_section1744mcpsimp"></a>

无

