# HcclSocket 

## 函数原型<a name="zh-cn_topic_0000001956458609_section1935mcpsimp"></a>

```
HcclSocket(const std::string &tag, HcclNetDevCtx netDevCtx, const HcclIpAddress &remoteIp, u32 remotePort, HcclSocketRole localRole)
HcclSocket(HcclNetDevCtx netDevCtx, u32 localPort)
```

## 函数功能<a name="zh-cn_topic_0000001956458609_section1938mcpsimp"></a>

构造socket对象。

## 参数说明<a name="zh-cn_topic_0000001956458609_section1941mcpsimp"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0000001956458609_table1943mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001956458609_row1950mcpsimp"><th class="cellrowborder" valign="top" width="28.71%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001956458609_p1952mcpsimp"><a name="zh-cn_topic_0000001956458609_p1952mcpsimp"></a><a name="zh-cn_topic_0000001956458609_p1952mcpsimp"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="13.86%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001956458609_p1954mcpsimp"><a name="zh-cn_topic_0000001956458609_p1954mcpsimp"></a><a name="zh-cn_topic_0000001956458609_p1954mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="57.43000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001956458609_p1956mcpsimp"><a name="zh-cn_topic_0000001956458609_p1956mcpsimp"></a><a name="zh-cn_topic_0000001956458609_p1956mcpsimp"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001956458609_row1958mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956458609_p1960mcpsimp"><a name="zh-cn_topic_0000001956458609_p1960mcpsimp"></a><a name="zh-cn_topic_0000001956458609_p1960mcpsimp"></a>const std::string &amp;tag</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956458609_p1962mcpsimp"><a name="zh-cn_topic_0000001956458609_p1962mcpsimp"></a><a name="zh-cn_topic_0000001956458609_p1962mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956458609_p1964mcpsimp"><a name="zh-cn_topic_0000001956458609_p1964mcpsimp"></a><a name="zh-cn_topic_0000001956458609_p1964mcpsimp"></a>Tag标识</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001956458609_row1965mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956458609_p1967mcpsimp"><a name="zh-cn_topic_0000001956458609_p1967mcpsimp"></a><a name="zh-cn_topic_0000001956458609_p1967mcpsimp"></a>HcclNetDevCtx netDevCtx</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956458609_p1969mcpsimp"><a name="zh-cn_topic_0000001956458609_p1969mcpsimp"></a><a name="zh-cn_topic_0000001956458609_p1969mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956458609_p1971mcpsimp"><a name="zh-cn_topic_0000001956458609_p1971mcpsimp"></a><a name="zh-cn_topic_0000001956458609_p1971mcpsimp"></a>网卡设备handle</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001956458609_row1972mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956458609_p1974mcpsimp"><a name="zh-cn_topic_0000001956458609_p1974mcpsimp"></a><a name="zh-cn_topic_0000001956458609_p1974mcpsimp"></a>const HcclIpAddress &amp;remoteIp</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956458609_p1976mcpsimp"><a name="zh-cn_topic_0000001956458609_p1976mcpsimp"></a><a name="zh-cn_topic_0000001956458609_p1976mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956458609_p1978mcpsimp"><a name="zh-cn_topic_0000001956458609_p1978mcpsimp"></a><a name="zh-cn_topic_0000001956458609_p1978mcpsimp"></a>对端IP信息</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001956458609_row1979mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956458609_p1981mcpsimp"><a name="zh-cn_topic_0000001956458609_p1981mcpsimp"></a><a name="zh-cn_topic_0000001956458609_p1981mcpsimp"></a>u32 remotePort</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956458609_p1983mcpsimp"><a name="zh-cn_topic_0000001956458609_p1983mcpsimp"></a><a name="zh-cn_topic_0000001956458609_p1983mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956458609_p1985mcpsimp"><a name="zh-cn_topic_0000001956458609_p1985mcpsimp"></a><a name="zh-cn_topic_0000001956458609_p1985mcpsimp"></a>对端prot</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001956458609_row1986mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956458609_p1988mcpsimp"><a name="zh-cn_topic_0000001956458609_p1988mcpsimp"></a><a name="zh-cn_topic_0000001956458609_p1988mcpsimp"></a>HcclSocketRole localRole</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956458609_p1990mcpsimp"><a name="zh-cn_topic_0000001956458609_p1990mcpsimp"></a><a name="zh-cn_topic_0000001956458609_p1990mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956458609_p1992mcpsimp"><a name="zh-cn_topic_0000001956458609_p1992mcpsimp"></a><a name="zh-cn_topic_0000001956458609_p1992mcpsimp"></a>本地建链角色</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001956458609_row157084526295"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956458609_p2025mcpsimp"><a name="zh-cn_topic_0000001956458609_p2025mcpsimp"></a><a name="zh-cn_topic_0000001956458609_p2025mcpsimp"></a>HcclNetDevCtx netDevCtx</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956458609_p2027mcpsimp"><a name="zh-cn_topic_0000001956458609_p2027mcpsimp"></a><a name="zh-cn_topic_0000001956458609_p2027mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956458609_p2029mcpsimp"><a name="zh-cn_topic_0000001956458609_p2029mcpsimp"></a><a name="zh-cn_topic_0000001956458609_p2029mcpsimp"></a>网卡设备handle</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001956458609_row58581954162913"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956458609_p2032mcpsimp"><a name="zh-cn_topic_0000001956458609_p2032mcpsimp"></a><a name="zh-cn_topic_0000001956458609_p2032mcpsimp"></a>u32 localPort</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956458609_p2034mcpsimp"><a name="zh-cn_topic_0000001956458609_p2034mcpsimp"></a><a name="zh-cn_topic_0000001956458609_p2034mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956458609_p2036mcpsimp"><a name="zh-cn_topic_0000001956458609_p2036mcpsimp"></a><a name="zh-cn_topic_0000001956458609_p2036mcpsimp"></a>本端port</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001956458609_section1993mcpsimp"></a>

无。

## 约束说明<a name="zh-cn_topic_0000001956458609_section1996mcpsimp"></a>

无

