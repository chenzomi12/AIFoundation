# HcclNetOpenDev 

## 函数原型<a name="zh-cn_topic_0000001929299750_section1602mcpsimp"></a>

```
HcclResult HcclNetOpenDev(HcclNetDevCtx *netDevCtx, NicType nicType, s32 devicePhyId, s32 deviceLogicId, hccl::HcclIpAddress localIp)
```

## 函数功能<a name="zh-cn_topic_0000001929299750_section1605mcpsimp"></a>

打开网卡设备。

## 参数说明<a name="zh-cn_topic_0000001929299750_section1608mcpsimp"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0000001929299750_table1610mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001929299750_row1617mcpsimp"><th class="cellrowborder" valign="top" width="28.71%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001929299750_p1619mcpsimp"><a name="zh-cn_topic_0000001929299750_p1619mcpsimp"></a><a name="zh-cn_topic_0000001929299750_p1619mcpsimp"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="13.86%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001929299750_p1621mcpsimp"><a name="zh-cn_topic_0000001929299750_p1621mcpsimp"></a><a name="zh-cn_topic_0000001929299750_p1621mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="57.43000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001929299750_p1623mcpsimp"><a name="zh-cn_topic_0000001929299750_p1623mcpsimp"></a><a name="zh-cn_topic_0000001929299750_p1623mcpsimp"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001929299750_row1625mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299750_p1627mcpsimp"><a name="zh-cn_topic_0000001929299750_p1627mcpsimp"></a><a name="zh-cn_topic_0000001929299750_p1627mcpsimp"></a>HcclNetDevCtx *netDevCtx</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299750_p1629mcpsimp"><a name="zh-cn_topic_0000001929299750_p1629mcpsimp"></a><a name="zh-cn_topic_0000001929299750_p1629mcpsimp"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299750_p1631mcpsimp"><a name="zh-cn_topic_0000001929299750_p1631mcpsimp"></a><a name="zh-cn_topic_0000001929299750_p1631mcpsimp"></a>网卡设备handle</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929299750_row1632mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299750_p1634mcpsimp"><a name="zh-cn_topic_0000001929299750_p1634mcpsimp"></a><a name="zh-cn_topic_0000001929299750_p1634mcpsimp"></a>NicType nicType</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299750_p1636mcpsimp"><a name="zh-cn_topic_0000001929299750_p1636mcpsimp"></a><a name="zh-cn_topic_0000001929299750_p1636mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299750_p1638mcpsimp"><a name="zh-cn_topic_0000001929299750_p1638mcpsimp"></a><a name="zh-cn_topic_0000001929299750_p1638mcpsimp"></a>网卡类型</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929299750_row1639mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299750_p1641mcpsimp"><a name="zh-cn_topic_0000001929299750_p1641mcpsimp"></a><a name="zh-cn_topic_0000001929299750_p1641mcpsimp"></a>s32 devicePhyId</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299750_p1643mcpsimp"><a name="zh-cn_topic_0000001929299750_p1643mcpsimp"></a><a name="zh-cn_topic_0000001929299750_p1643mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299750_p1645mcpsimp"><a name="zh-cn_topic_0000001929299750_p1645mcpsimp"></a><a name="zh-cn_topic_0000001929299750_p1645mcpsimp"></a>Device phy ID</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929299750_row1646mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299750_p1648mcpsimp"><a name="zh-cn_topic_0000001929299750_p1648mcpsimp"></a><a name="zh-cn_topic_0000001929299750_p1648mcpsimp"></a>s32 deviceLogicId</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299750_p1650mcpsimp"><a name="zh-cn_topic_0000001929299750_p1650mcpsimp"></a><a name="zh-cn_topic_0000001929299750_p1650mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299750_p1652mcpsimp"><a name="zh-cn_topic_0000001929299750_p1652mcpsimp"></a><a name="zh-cn_topic_0000001929299750_p1652mcpsimp"></a>Device logic ID</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929299750_row1653mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299750_p1655mcpsimp"><a name="zh-cn_topic_0000001929299750_p1655mcpsimp"></a><a name="zh-cn_topic_0000001929299750_p1655mcpsimp"></a>hccl::HcclIpAddress localIp</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299750_p1657mcpsimp"><a name="zh-cn_topic_0000001929299750_p1657mcpsimp"></a><a name="zh-cn_topic_0000001929299750_p1657mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299750_p1659mcpsimp"><a name="zh-cn_topic_0000001929299750_p1659mcpsimp"></a><a name="zh-cn_topic_0000001929299750_p1659mcpsimp"></a>网卡ip信息</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001929299750_section1660mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

## 约束说明<a name="zh-cn_topic_0000001929299750_section1663mcpsimp"></a>

无

