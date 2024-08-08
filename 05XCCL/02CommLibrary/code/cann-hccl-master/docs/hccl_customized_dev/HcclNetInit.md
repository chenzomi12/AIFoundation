# HcclNetInit 

## 函数原型<a name="zh-cn_topic_0000001956458597_section1493mcpsimp"></a>

```
HcclResult HcclNetInit(NICDeployment nicDeploy, s32 devicePhyId, s32 deviceLogicId, bool enableWhitelistFlag)
```

## 函数功能<a name="zh-cn_topic_0000001956458597_section1496mcpsimp"></a>

初始化网络功能。

## 参数说明<a name="zh-cn_topic_0000001956458597_section1499mcpsimp"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0000001956458597_table1501mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001956458597_row1508mcpsimp"><th class="cellrowborder" valign="top" width="28.71%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001956458597_p1510mcpsimp"><a name="zh-cn_topic_0000001956458597_p1510mcpsimp"></a><a name="zh-cn_topic_0000001956458597_p1510mcpsimp"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="13.86%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001956458597_p1512mcpsimp"><a name="zh-cn_topic_0000001956458597_p1512mcpsimp"></a><a name="zh-cn_topic_0000001956458597_p1512mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="57.43000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001956458597_p1514mcpsimp"><a name="zh-cn_topic_0000001956458597_p1514mcpsimp"></a><a name="zh-cn_topic_0000001956458597_p1514mcpsimp"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001956458597_row1516mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956458597_p1518mcpsimp"><a name="zh-cn_topic_0000001956458597_p1518mcpsimp"></a><a name="zh-cn_topic_0000001956458597_p1518mcpsimp"></a>NICDeployment nicDeploy</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956458597_p1520mcpsimp"><a name="zh-cn_topic_0000001956458597_p1520mcpsimp"></a><a name="zh-cn_topic_0000001956458597_p1520mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956458597_p1522mcpsimp"><a name="zh-cn_topic_0000001956458597_p1522mcpsimp"></a><a name="zh-cn_topic_0000001956458597_p1522mcpsimp"></a>网卡部署位置</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001956458597_row1523mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956458597_p1525mcpsimp"><a name="zh-cn_topic_0000001956458597_p1525mcpsimp"></a><a name="zh-cn_topic_0000001956458597_p1525mcpsimp"></a>s32 devicePhyId</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956458597_p1527mcpsimp"><a name="zh-cn_topic_0000001956458597_p1527mcpsimp"></a><a name="zh-cn_topic_0000001956458597_p1527mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956458597_p1529mcpsimp"><a name="zh-cn_topic_0000001956458597_p1529mcpsimp"></a><a name="zh-cn_topic_0000001956458597_p1529mcpsimp"></a>Device phy ID</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001956458597_row1530mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956458597_p1532mcpsimp"><a name="zh-cn_topic_0000001956458597_p1532mcpsimp"></a><a name="zh-cn_topic_0000001956458597_p1532mcpsimp"></a>s32 deviceLogicId</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956458597_p1534mcpsimp"><a name="zh-cn_topic_0000001956458597_p1534mcpsimp"></a><a name="zh-cn_topic_0000001956458597_p1534mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956458597_p1536mcpsimp"><a name="zh-cn_topic_0000001956458597_p1536mcpsimp"></a><a name="zh-cn_topic_0000001956458597_p1536mcpsimp"></a>Device logic ID</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001956458597_row1537mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956458597_p1539mcpsimp"><a name="zh-cn_topic_0000001956458597_p1539mcpsimp"></a><a name="zh-cn_topic_0000001956458597_p1539mcpsimp"></a>bool enableWhitelistFlag</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956458597_p1541mcpsimp"><a name="zh-cn_topic_0000001956458597_p1541mcpsimp"></a><a name="zh-cn_topic_0000001956458597_p1541mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956458597_p1543mcpsimp"><a name="zh-cn_topic_0000001956458597_p1543mcpsimp"></a><a name="zh-cn_topic_0000001956458597_p1543mcpsimp"></a>是否开启白名单校验</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001956458597_section1544mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

## 约束说明<a name="zh-cn_topic_0000001956458597_section1547mcpsimp"></a>

无

