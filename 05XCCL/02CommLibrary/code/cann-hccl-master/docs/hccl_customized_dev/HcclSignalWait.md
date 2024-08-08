# HcclSignalWait 

## 函数原型<a name="zh-cn_topic_0000001926464496_section1953mcpsimp"></a>

```
HcclResult HcclSignalWait(HcclDispatcher dispatcherPtr, HcclRtNotify signal, hccl::Stream &stream, u32 userRank, u32 remoteUserRank, s32 stage, bool inchip)
```

## 功能说明<a name="zh-cn_topic_0000001926464496_section1955mcpsimp"></a>

notify wait。

## 参数说明<a name="zh-cn_topic_0000001926464496_section1957mcpsimp"></a>

<a name="zh-cn_topic_0000001926464496_table1958mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001926464496_row1964mcpsimp"><th class="cellrowborder" valign="top" width="46%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001926464496_p1966mcpsimp"><a name="zh-cn_topic_0000001926464496_p1966mcpsimp"></a><a name="zh-cn_topic_0000001926464496_p1966mcpsimp"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="22%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001926464496_p1968mcpsimp"><a name="zh-cn_topic_0000001926464496_p1968mcpsimp"></a><a name="zh-cn_topic_0000001926464496_p1968mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="32%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001926464496_p1970mcpsimp"><a name="zh-cn_topic_0000001926464496_p1970mcpsimp"></a><a name="zh-cn_topic_0000001926464496_p1970mcpsimp"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001926464496_row1972mcpsimp"><td class="cellrowborder" valign="top" width="46%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001926464496_p1974mcpsimp"><a name="zh-cn_topic_0000001926464496_p1974mcpsimp"></a><a name="zh-cn_topic_0000001926464496_p1974mcpsimp"></a>HcclDispatcher dispatcher</p>
</td>
<td class="cellrowborder" valign="top" width="22%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001926464496_p1976mcpsimp"><a name="zh-cn_topic_0000001926464496_p1976mcpsimp"></a><a name="zh-cn_topic_0000001926464496_p1976mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="32%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001926464496_p1978mcpsimp"><a name="zh-cn_topic_0000001926464496_p1978mcpsimp"></a><a name="zh-cn_topic_0000001926464496_p1978mcpsimp"></a>dispatcher handle</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001926464496_row1979mcpsimp"><td class="cellrowborder" valign="top" width="46%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001926464496_p1981mcpsimp"><a name="zh-cn_topic_0000001926464496_p1981mcpsimp"></a><a name="zh-cn_topic_0000001926464496_p1981mcpsimp"></a>HcclRtNotify signal</p>
</td>
<td class="cellrowborder" valign="top" width="22%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001926464496_p1983mcpsimp"><a name="zh-cn_topic_0000001926464496_p1983mcpsimp"></a><a name="zh-cn_topic_0000001926464496_p1983mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="32%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001926464496_p1985mcpsimp"><a name="zh-cn_topic_0000001926464496_p1985mcpsimp"></a><a name="zh-cn_topic_0000001926464496_p1985mcpsimp"></a>rt notify</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001926464496_row1986mcpsimp"><td class="cellrowborder" valign="top" width="46%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001926464496_p1988mcpsimp"><a name="zh-cn_topic_0000001926464496_p1988mcpsimp"></a><a name="zh-cn_topic_0000001926464496_p1988mcpsimp"></a>hccl::Stream &amp;stream</p>
</td>
<td class="cellrowborder" valign="top" width="22%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001926464496_p1990mcpsimp"><a name="zh-cn_topic_0000001926464496_p1990mcpsimp"></a><a name="zh-cn_topic_0000001926464496_p1990mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="32%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001926464496_p1992mcpsimp"><a name="zh-cn_topic_0000001926464496_p1992mcpsimp"></a><a name="zh-cn_topic_0000001926464496_p1992mcpsimp"></a>stream对象</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001926464496_row1993mcpsimp"><td class="cellrowborder" valign="top" width="46%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001926464496_p1995mcpsimp"><a name="zh-cn_topic_0000001926464496_p1995mcpsimp"></a><a name="zh-cn_topic_0000001926464496_p1995mcpsimp"></a>u32 userRank</p>
</td>
<td class="cellrowborder" valign="top" width="22%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001926464496_p1997mcpsimp"><a name="zh-cn_topic_0000001926464496_p1997mcpsimp"></a><a name="zh-cn_topic_0000001926464496_p1997mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="32%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001926464496_p1999mcpsimp"><a name="zh-cn_topic_0000001926464496_p1999mcpsimp"></a><a name="zh-cn_topic_0000001926464496_p1999mcpsimp"></a>本端world rank</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001926464496_row2000mcpsimp"><td class="cellrowborder" valign="top" width="46%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001926464496_p2002mcpsimp"><a name="zh-cn_topic_0000001926464496_p2002mcpsimp"></a><a name="zh-cn_topic_0000001926464496_p2002mcpsimp"></a>u32 remoteUserRank</p>
</td>
<td class="cellrowborder" valign="top" width="22%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001926464496_p2004mcpsimp"><a name="zh-cn_topic_0000001926464496_p2004mcpsimp"></a><a name="zh-cn_topic_0000001926464496_p2004mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="32%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001926464496_p2006mcpsimp"><a name="zh-cn_topic_0000001926464496_p2006mcpsimp"></a><a name="zh-cn_topic_0000001926464496_p2006mcpsimp"></a>对端world rank</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001926464496_row2007mcpsimp"><td class="cellrowborder" valign="top" width="46%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001926464496_p2009mcpsimp"><a name="zh-cn_topic_0000001926464496_p2009mcpsimp"></a><a name="zh-cn_topic_0000001926464496_p2009mcpsimp"></a>s32 stage</p>
</td>
<td class="cellrowborder" valign="top" width="22%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001926464496_p2011mcpsimp"><a name="zh-cn_topic_0000001926464496_p2011mcpsimp"></a><a name="zh-cn_topic_0000001926464496_p2011mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="32%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001926464496_p2013mcpsimp"><a name="zh-cn_topic_0000001926464496_p2013mcpsimp"></a><a name="zh-cn_topic_0000001926464496_p2013mcpsimp"></a>算法stage</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001926464496_row2014mcpsimp"><td class="cellrowborder" valign="top" width="46%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001926464496_p2016mcpsimp"><a name="zh-cn_topic_0000001926464496_p2016mcpsimp"></a><a name="zh-cn_topic_0000001926464496_p2016mcpsimp"></a>bool inchip</p>
</td>
<td class="cellrowborder" valign="top" width="22%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001926464496_p2018mcpsimp"><a name="zh-cn_topic_0000001926464496_p2018mcpsimp"></a><a name="zh-cn_topic_0000001926464496_p2018mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="32%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001926464496_p2020mcpsimp"><a name="zh-cn_topic_0000001926464496_p2020mcpsimp"></a><a name="zh-cn_topic_0000001926464496_p2020mcpsimp"></a>是否跨片</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001926464496_section2021mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS；其他失败。

