# HcclIpAddress 

## 函数原型<a name="zh-cn_topic_0000001929299730_section1066mcpsimp"></a>

```
HcclIpAddress()
HcclIpAddress(u32 address)
HcclIpAddress(s32 family, const union HcclInAddr &address)
HcclIpAddress(const struct in_addr &address)
HcclIpAddress(const struct in6_addr &address)
HcclIpAddress(const std::string &address)
```

## 函数功能<a name="zh-cn_topic_0000001929299730_section1069mcpsimp"></a>

构造IpAddress。

## 参数说明<a name="zh-cn_topic_0000001929299730_section1072mcpsimp"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0000001929299730_table1090mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001929299730_row1097mcpsimp"><th class="cellrowborder" valign="top" width="28.71%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001929299730_p1099mcpsimp"><a name="zh-cn_topic_0000001929299730_p1099mcpsimp"></a><a name="zh-cn_topic_0000001929299730_p1099mcpsimp"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="13.86%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001929299730_p1101mcpsimp"><a name="zh-cn_topic_0000001929299730_p1101mcpsimp"></a><a name="zh-cn_topic_0000001929299730_p1101mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="57.43000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001929299730_p1103mcpsimp"><a name="zh-cn_topic_0000001929299730_p1103mcpsimp"></a><a name="zh-cn_topic_0000001929299730_p1103mcpsimp"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001929299730_row1105mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299730_p1107mcpsimp"><a name="zh-cn_topic_0000001929299730_p1107mcpsimp"></a><a name="zh-cn_topic_0000001929299730_p1107mcpsimp"></a>u32 address</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299730_p1109mcpsimp"><a name="zh-cn_topic_0000001929299730_p1109mcpsimp"></a><a name="zh-cn_topic_0000001929299730_p1109mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299730_p1111mcpsimp"><a name="zh-cn_topic_0000001929299730_p1111mcpsimp"></a><a name="zh-cn_topic_0000001929299730_p1111mcpsimp"></a>U32表示的Ip</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929299730_row1521183310149"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299730_p1144mcpsimp"><a name="zh-cn_topic_0000001929299730_p1144mcpsimp"></a><a name="zh-cn_topic_0000001929299730_p1144mcpsimp"></a>s32 family</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299730_p1146mcpsimp"><a name="zh-cn_topic_0000001929299730_p1146mcpsimp"></a><a name="zh-cn_topic_0000001929299730_p1146mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299730_p1148mcpsimp"><a name="zh-cn_topic_0000001929299730_p1148mcpsimp"></a><a name="zh-cn_topic_0000001929299730_p1148mcpsimp"></a>Ip地址族</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929299730_row5940133501415"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299730_p1151mcpsimp"><a name="zh-cn_topic_0000001929299730_p1151mcpsimp"></a><a name="zh-cn_topic_0000001929299730_p1151mcpsimp"></a>const union HcclInAddr &amp;address</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299730_p1153mcpsimp"><a name="zh-cn_topic_0000001929299730_p1153mcpsimp"></a><a name="zh-cn_topic_0000001929299730_p1153mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299730_p1155mcpsimp"><a name="zh-cn_topic_0000001929299730_p1155mcpsimp"></a><a name="zh-cn_topic_0000001929299730_p1155mcpsimp"></a>Ip信息</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929299730_row1463125612143"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299730_p1188mcpsimp"><a name="zh-cn_topic_0000001929299730_p1188mcpsimp"></a><a name="zh-cn_topic_0000001929299730_p1188mcpsimp"></a>const struct in_addr &amp;address</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299730_p1190mcpsimp"><a name="zh-cn_topic_0000001929299730_p1190mcpsimp"></a><a name="zh-cn_topic_0000001929299730_p1190mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299730_p1192mcpsimp"><a name="zh-cn_topic_0000001929299730_p1192mcpsimp"></a><a name="zh-cn_topic_0000001929299730_p1192mcpsimp"></a>Ipv4信息</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929299730_row48579601510"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299730_p27521820101510"><a name="zh-cn_topic_0000001929299730_p27521820101510"></a><a name="zh-cn_topic_0000001929299730_p27521820101510"></a>const struct in6_addr &amp;address</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299730_p1227mcpsimp"><a name="zh-cn_topic_0000001929299730_p1227mcpsimp"></a><a name="zh-cn_topic_0000001929299730_p1227mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299730_p1229mcpsimp"><a name="zh-cn_topic_0000001929299730_p1229mcpsimp"></a><a name="zh-cn_topic_0000001929299730_p1229mcpsimp"></a>Ipv6信息</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929299730_row7730257158"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299730_p1262mcpsimp"><a name="zh-cn_topic_0000001929299730_p1262mcpsimp"></a><a name="zh-cn_topic_0000001929299730_p1262mcpsimp"></a>const std::string &amp;address</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299730_p1264mcpsimp"><a name="zh-cn_topic_0000001929299730_p1264mcpsimp"></a><a name="zh-cn_topic_0000001929299730_p1264mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299730_p1266mcpsimp"><a name="zh-cn_topic_0000001929299730_p1266mcpsimp"></a><a name="zh-cn_topic_0000001929299730_p1266mcpsimp"></a>字符串类型的Ip地址</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001929299730_section1075mcpsimp"></a>

无。

## 约束说明<a name="zh-cn_topic_0000001929299730_section1078mcpsimp"></a>

无。

