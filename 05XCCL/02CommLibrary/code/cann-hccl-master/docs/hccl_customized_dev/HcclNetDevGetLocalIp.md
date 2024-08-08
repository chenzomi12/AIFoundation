# HcclNetDevGetLocalIp 

## 函数原型<a name="zh-cn_topic_0000001956618401_section1748mcpsimp"></a>

```
HcclResult HcclNetDevGetLocalIp(HcclNetDevCtx netDevCtx, hccl::HcclIpAddress &localIp)
```

## 函数功能<a name="zh-cn_topic_0000001956618401_section1751mcpsimp"></a>

获取对应的网卡ip。

## 参数说明<a name="zh-cn_topic_0000001956618401_section1754mcpsimp"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0000001956618401_table1756mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001956618401_row1763mcpsimp"><th class="cellrowborder" valign="top" width="28.71%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001956618401_p1765mcpsimp"><a name="zh-cn_topic_0000001956618401_p1765mcpsimp"></a><a name="zh-cn_topic_0000001956618401_p1765mcpsimp"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="13.86%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001956618401_p1767mcpsimp"><a name="zh-cn_topic_0000001956618401_p1767mcpsimp"></a><a name="zh-cn_topic_0000001956618401_p1767mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="57.43000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001956618401_p1769mcpsimp"><a name="zh-cn_topic_0000001956618401_p1769mcpsimp"></a><a name="zh-cn_topic_0000001956618401_p1769mcpsimp"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001956618401_row1771mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956618401_p1773mcpsimp"><a name="zh-cn_topic_0000001956618401_p1773mcpsimp"></a><a name="zh-cn_topic_0000001956618401_p1773mcpsimp"></a>HcclNetDevCtx *netDevCtx</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956618401_p1775mcpsimp"><a name="zh-cn_topic_0000001956618401_p1775mcpsimp"></a><a name="zh-cn_topic_0000001956618401_p1775mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956618401_p1777mcpsimp"><a name="zh-cn_topic_0000001956618401_p1777mcpsimp"></a><a name="zh-cn_topic_0000001956618401_p1777mcpsimp"></a>网卡设备handle</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001956618401_row1778mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956618401_p1780mcpsimp"><a name="zh-cn_topic_0000001956618401_p1780mcpsimp"></a><a name="zh-cn_topic_0000001956618401_p1780mcpsimp"></a>hccl::HcclIpAddress &amp;localIp</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956618401_p1782mcpsimp"><a name="zh-cn_topic_0000001956618401_p1782mcpsimp"></a><a name="zh-cn_topic_0000001956618401_p1782mcpsimp"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956618401_p1784mcpsimp"><a name="zh-cn_topic_0000001956618401_p1784mcpsimp"></a><a name="zh-cn_topic_0000001956618401_p1784mcpsimp"></a>Ip信息</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001956618401_section1785mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

## 约束说明<a name="zh-cn_topic_0000001956618401_section1788mcpsimp"></a>

无

