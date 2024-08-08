# HcclNetDeInit 

## 函数原型<a name="zh-cn_topic_0000001956618397_section1551mcpsimp"></a>

```
HcclResult HcclNetDeInit(NICDeployment nicDeploy, s32 devicePhyId, s32 deviceLogicId)
```

## 函数功能<a name="zh-cn_topic_0000001956618397_section1554mcpsimp"></a>

销毁网络功能。

## 参数说明<a name="zh-cn_topic_0000001956618397_section1557mcpsimp"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0000001956618397_table1559mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001956618397_row1566mcpsimp"><th class="cellrowborder" valign="top" width="28.71%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001956618397_p1568mcpsimp"><a name="zh-cn_topic_0000001956618397_p1568mcpsimp"></a><a name="zh-cn_topic_0000001956618397_p1568mcpsimp"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="13.86%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001956618397_p1570mcpsimp"><a name="zh-cn_topic_0000001956618397_p1570mcpsimp"></a><a name="zh-cn_topic_0000001956618397_p1570mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="57.43000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001956618397_p1572mcpsimp"><a name="zh-cn_topic_0000001956618397_p1572mcpsimp"></a><a name="zh-cn_topic_0000001956618397_p1572mcpsimp"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001956618397_row1574mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956618397_p1576mcpsimp"><a name="zh-cn_topic_0000001956618397_p1576mcpsimp"></a><a name="zh-cn_topic_0000001956618397_p1576mcpsimp"></a>NICDeployment nicDeploy</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956618397_p1578mcpsimp"><a name="zh-cn_topic_0000001956618397_p1578mcpsimp"></a><a name="zh-cn_topic_0000001956618397_p1578mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956618397_p1580mcpsimp"><a name="zh-cn_topic_0000001956618397_p1580mcpsimp"></a><a name="zh-cn_topic_0000001956618397_p1580mcpsimp"></a>网卡部署位置</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001956618397_row1581mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956618397_p1583mcpsimp"><a name="zh-cn_topic_0000001956618397_p1583mcpsimp"></a><a name="zh-cn_topic_0000001956618397_p1583mcpsimp"></a>s32 devicePhyId</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956618397_p1585mcpsimp"><a name="zh-cn_topic_0000001956618397_p1585mcpsimp"></a><a name="zh-cn_topic_0000001956618397_p1585mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956618397_p1587mcpsimp"><a name="zh-cn_topic_0000001956618397_p1587mcpsimp"></a><a name="zh-cn_topic_0000001956618397_p1587mcpsimp"></a>Device phy ID</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001956618397_row1588mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956618397_p1590mcpsimp"><a name="zh-cn_topic_0000001956618397_p1590mcpsimp"></a><a name="zh-cn_topic_0000001956618397_p1590mcpsimp"></a>s32 deviceLogicId</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956618397_p1592mcpsimp"><a name="zh-cn_topic_0000001956618397_p1592mcpsimp"></a><a name="zh-cn_topic_0000001956618397_p1592mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956618397_p1594mcpsimp"><a name="zh-cn_topic_0000001956618397_p1594mcpsimp"></a><a name="zh-cn_topic_0000001956618397_p1594mcpsimp"></a>Device logic ID</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001956618397_section1595mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

## 约束说明<a name="zh-cn_topic_0000001956618397_section1598mcpsimp"></a>

无

