# Transport 

## 函数原型<a name="zh-cn_topic_0000001956458765_section6586mcpsimp"></a>

```
Transport(TransportBase *pimpl)
Transport(TransportType type, TransportPara& para, const HcclDispatcher dispatcher, const std::unique_ptr<NotifyPool> &notifyPool, MachinePara &machinePara,  const TransportDeviceP2pData &transDevP2pData = TransportDeviceP2pData())
```

## 函数功能<a name="zh-cn_topic_0000001956458765_section6589mcpsimp"></a>

Transport构造函数。

## 参数说明<a name="zh-cn_topic_0000001956458765_section6592mcpsimp"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0000001956458765_table6631mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001956458765_row6638mcpsimp"><th class="cellrowborder" valign="top" width="28.71%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001956458765_p6640mcpsimp"><a name="zh-cn_topic_0000001956458765_p6640mcpsimp"></a><a name="zh-cn_topic_0000001956458765_p6640mcpsimp"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="13.86%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001956458765_p6642mcpsimp"><a name="zh-cn_topic_0000001956458765_p6642mcpsimp"></a><a name="zh-cn_topic_0000001956458765_p6642mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="57.43000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001956458765_p6644mcpsimp"><a name="zh-cn_topic_0000001956458765_p6644mcpsimp"></a><a name="zh-cn_topic_0000001956458765_p6644mcpsimp"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001956458765_row185821731172111"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956458765_p6611mcpsimp"><a name="zh-cn_topic_0000001956458765_p6611mcpsimp"></a><a name="zh-cn_topic_0000001956458765_p6611mcpsimp"></a>TransportBase *pimpl</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956458765_p6613mcpsimp"><a name="zh-cn_topic_0000001956458765_p6613mcpsimp"></a><a name="zh-cn_topic_0000001956458765_p6613mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956458765_p6615mcpsimp"><a name="zh-cn_topic_0000001956458765_p6615mcpsimp"></a><a name="zh-cn_topic_0000001956458765_p6615mcpsimp"></a>TransportBase指针</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001956458765_row6646mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956458765_p6648mcpsimp"><a name="zh-cn_topic_0000001956458765_p6648mcpsimp"></a><a name="zh-cn_topic_0000001956458765_p6648mcpsimp"></a>TransportType type</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956458765_p6650mcpsimp"><a name="zh-cn_topic_0000001956458765_p6650mcpsimp"></a><a name="zh-cn_topic_0000001956458765_p6650mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956458765_p6652mcpsimp"><a name="zh-cn_topic_0000001956458765_p6652mcpsimp"></a><a name="zh-cn_topic_0000001956458765_p6652mcpsimp"></a>Transport类型</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001956458765_row6653mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956458765_p6655mcpsimp"><a name="zh-cn_topic_0000001956458765_p6655mcpsimp"></a><a name="zh-cn_topic_0000001956458765_p6655mcpsimp"></a>TransportPara&amp; para</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956458765_p6657mcpsimp"><a name="zh-cn_topic_0000001956458765_p6657mcpsimp"></a><a name="zh-cn_topic_0000001956458765_p6657mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956458765_p6659mcpsimp"><a name="zh-cn_topic_0000001956458765_p6659mcpsimp"></a><a name="zh-cn_topic_0000001956458765_p6659mcpsimp"></a>Transport参数</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001956458765_row6660mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956458765_p6662mcpsimp"><a name="zh-cn_topic_0000001956458765_p6662mcpsimp"></a><a name="zh-cn_topic_0000001956458765_p6662mcpsimp"></a>const HcclDispatcher dispatcher</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956458765_p6664mcpsimp"><a name="zh-cn_topic_0000001956458765_p6664mcpsimp"></a><a name="zh-cn_topic_0000001956458765_p6664mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956458765_p6666mcpsimp"><a name="zh-cn_topic_0000001956458765_p6666mcpsimp"></a><a name="zh-cn_topic_0000001956458765_p6666mcpsimp"></a>Dispatcher handle</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001956458765_row6667mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956458765_p6669mcpsimp"><a name="zh-cn_topic_0000001956458765_p6669mcpsimp"></a><a name="zh-cn_topic_0000001956458765_p6669mcpsimp"></a>const std::unique_ptr&lt;NotifyPool&gt; &amp;notifyPool</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956458765_p6671mcpsimp"><a name="zh-cn_topic_0000001956458765_p6671mcpsimp"></a><a name="zh-cn_topic_0000001956458765_p6671mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956458765_p6673mcpsimp"><a name="zh-cn_topic_0000001956458765_p6673mcpsimp"></a><a name="zh-cn_topic_0000001956458765_p6673mcpsimp"></a>Notify pool指针</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001956458765_row6674mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956458765_p6676mcpsimp"><a name="zh-cn_topic_0000001956458765_p6676mcpsimp"></a><a name="zh-cn_topic_0000001956458765_p6676mcpsimp"></a>MachinePara &amp;machinePara</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956458765_p6678mcpsimp"><a name="zh-cn_topic_0000001956458765_p6678mcpsimp"></a><a name="zh-cn_topic_0000001956458765_p6678mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956458765_entry6679mcpsimpp0"><a name="zh-cn_topic_0000001956458765_entry6679mcpsimpp0"></a><a name="zh-cn_topic_0000001956458765_entry6679mcpsimpp0"></a>建链相关参数</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001956458765_row6680mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956458765_p6682mcpsimp"><a name="zh-cn_topic_0000001956458765_p6682mcpsimp"></a><a name="zh-cn_topic_0000001956458765_p6682mcpsimp"></a>const TransportDeviceP2pData &amp;transDevP2pData</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956458765_p6684mcpsimp"><a name="zh-cn_topic_0000001956458765_p6684mcpsimp"></a><a name="zh-cn_topic_0000001956458765_p6684mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956458765_entry6685mcpsimpp0"><a name="zh-cn_topic_0000001956458765_entry6685mcpsimpp0"></a><a name="zh-cn_topic_0000001956458765_entry6685mcpsimpp0"></a>device侧相关数据</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001956458765_section6616mcpsimp"></a>

无。

## 约束说明<a name="zh-cn_topic_0000001956458765_section6619mcpsimp"></a>

无

