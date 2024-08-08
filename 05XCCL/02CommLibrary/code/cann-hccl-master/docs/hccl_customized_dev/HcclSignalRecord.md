# HcclSignalRecord 

## 函数原型<a name="zh-cn_topic_0000001926623848_section515mcpsimp"></a>

```
HcclResult HcclSignalRecord(HcclDispatcher dispatcherPtr, HcclRtNotify signal, hccl::Stream &stream, u32 userRank, u64 offset, s32 stage, bool inchip, u64 signalAddr)
```

## 功能说明<a name="zh-cn_topic_0000001926623848_section517mcpsimp"></a>

notify record。

## 参数说明<a name="zh-cn_topic_0000001926623848_section519mcpsimp"></a>

<a name="zh-cn_topic_0000001926623848_table520mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001926623848_row526mcpsimp"><th class="cellrowborder" valign="top" width="46%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001926623848_p528mcpsimp"><a name="zh-cn_topic_0000001926623848_p528mcpsimp"></a><a name="zh-cn_topic_0000001926623848_p528mcpsimp"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="22%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001926623848_p530mcpsimp"><a name="zh-cn_topic_0000001926623848_p530mcpsimp"></a><a name="zh-cn_topic_0000001926623848_p530mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="32%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001926623848_p532mcpsimp"><a name="zh-cn_topic_0000001926623848_p532mcpsimp"></a><a name="zh-cn_topic_0000001926623848_p532mcpsimp"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001926623848_row534mcpsimp"><td class="cellrowborder" valign="top" width="46%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001926623848_p536mcpsimp"><a name="zh-cn_topic_0000001926623848_p536mcpsimp"></a><a name="zh-cn_topic_0000001926623848_p536mcpsimp"></a>HcclDispatcher dispatcher</p>
</td>
<td class="cellrowborder" valign="top" width="22%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001926623848_p538mcpsimp"><a name="zh-cn_topic_0000001926623848_p538mcpsimp"></a><a name="zh-cn_topic_0000001926623848_p538mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="32%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001926623848_p540mcpsimp"><a name="zh-cn_topic_0000001926623848_p540mcpsimp"></a><a name="zh-cn_topic_0000001926623848_p540mcpsimp"></a>dispatcher handle</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001926623848_row541mcpsimp"><td class="cellrowborder" valign="top" width="46%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001926623848_p543mcpsimp"><a name="zh-cn_topic_0000001926623848_p543mcpsimp"></a><a name="zh-cn_topic_0000001926623848_p543mcpsimp"></a>HcclRtNotify signal</p>
</td>
<td class="cellrowborder" valign="top" width="22%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001926623848_p545mcpsimp"><a name="zh-cn_topic_0000001926623848_p545mcpsimp"></a><a name="zh-cn_topic_0000001926623848_p545mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="32%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001926623848_p547mcpsimp"><a name="zh-cn_topic_0000001926623848_p547mcpsimp"></a><a name="zh-cn_topic_0000001926623848_p547mcpsimp"></a>rt notify</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001926623848_row548mcpsimp"><td class="cellrowborder" valign="top" width="46%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001926623848_p550mcpsimp"><a name="zh-cn_topic_0000001926623848_p550mcpsimp"></a><a name="zh-cn_topic_0000001926623848_p550mcpsimp"></a>hccl::Stream &amp;stream</p>
</td>
<td class="cellrowborder" valign="top" width="22%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001926623848_p552mcpsimp"><a name="zh-cn_topic_0000001926623848_p552mcpsimp"></a><a name="zh-cn_topic_0000001926623848_p552mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="32%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001926623848_p554mcpsimp"><a name="zh-cn_topic_0000001926623848_p554mcpsimp"></a><a name="zh-cn_topic_0000001926623848_p554mcpsimp"></a>stream对象</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001926623848_row555mcpsimp"><td class="cellrowborder" valign="top" width="46%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001926623848_p557mcpsimp"><a name="zh-cn_topic_0000001926623848_p557mcpsimp"></a><a name="zh-cn_topic_0000001926623848_p557mcpsimp"></a>u32 userRank</p>
</td>
<td class="cellrowborder" valign="top" width="22%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001926623848_p559mcpsimp"><a name="zh-cn_topic_0000001926623848_p559mcpsimp"></a><a name="zh-cn_topic_0000001926623848_p559mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="32%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001926623848_p561mcpsimp"><a name="zh-cn_topic_0000001926623848_p561mcpsimp"></a><a name="zh-cn_topic_0000001926623848_p561mcpsimp"></a>本端world rank</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001926623848_row562mcpsimp"><td class="cellrowborder" valign="top" width="46%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001926623848_p564mcpsimp"><a name="zh-cn_topic_0000001926623848_p564mcpsimp"></a><a name="zh-cn_topic_0000001926623848_p564mcpsimp"></a>u64 offset</p>
</td>
<td class="cellrowborder" valign="top" width="22%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001926623848_p566mcpsimp"><a name="zh-cn_topic_0000001926623848_p566mcpsimp"></a><a name="zh-cn_topic_0000001926623848_p566mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="32%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001926623848_p568mcpsimp"><a name="zh-cn_topic_0000001926623848_p568mcpsimp"></a><a name="zh-cn_topic_0000001926623848_p568mcpsimp"></a>notify offset</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001926623848_row569mcpsimp"><td class="cellrowborder" valign="top" width="46%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001926623848_p571mcpsimp"><a name="zh-cn_topic_0000001926623848_p571mcpsimp"></a><a name="zh-cn_topic_0000001926623848_p571mcpsimp"></a>s32 stage</p>
</td>
<td class="cellrowborder" valign="top" width="22%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001926623848_p573mcpsimp"><a name="zh-cn_topic_0000001926623848_p573mcpsimp"></a><a name="zh-cn_topic_0000001926623848_p573mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="32%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001926623848_p575mcpsimp"><a name="zh-cn_topic_0000001926623848_p575mcpsimp"></a><a name="zh-cn_topic_0000001926623848_p575mcpsimp"></a>算法stage</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001926623848_row576mcpsimp"><td class="cellrowborder" valign="top" width="46%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001926623848_p578mcpsimp"><a name="zh-cn_topic_0000001926623848_p578mcpsimp"></a><a name="zh-cn_topic_0000001926623848_p578mcpsimp"></a>bool inchip</p>
</td>
<td class="cellrowborder" valign="top" width="22%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001926623848_p580mcpsimp"><a name="zh-cn_topic_0000001926623848_p580mcpsimp"></a><a name="zh-cn_topic_0000001926623848_p580mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="32%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001926623848_p582mcpsimp"><a name="zh-cn_topic_0000001926623848_p582mcpsimp"></a><a name="zh-cn_topic_0000001926623848_p582mcpsimp"></a>是否跨片</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001926623848_row583mcpsimp"><td class="cellrowborder" valign="top" width="46%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001926623848_p585mcpsimp"><a name="zh-cn_topic_0000001926623848_p585mcpsimp"></a><a name="zh-cn_topic_0000001926623848_p585mcpsimp"></a>u64 signalAddr</p>
</td>
<td class="cellrowborder" valign="top" width="22%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001926623848_p587mcpsimp"><a name="zh-cn_topic_0000001926623848_p587mcpsimp"></a><a name="zh-cn_topic_0000001926623848_p587mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="32%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001926623848_p589mcpsimp"><a name="zh-cn_topic_0000001926623848_p589mcpsimp"></a><a name="zh-cn_topic_0000001926623848_p589mcpsimp"></a>notify address</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001926623848_section590mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS；其他失败。

