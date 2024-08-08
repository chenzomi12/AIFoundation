# TransDataDef 

## 函数原型<a name="zh-cn_topic_0000001939004906_section151mcpsimp"></a>

```
TransDataDef()
TransDataDef(u64 srcBuf, u64 dstBuf, u64 count, HcclDataType dataType, bool errorFlag = false, u32 tableId = DEFAULT_TABLE_ID_VALUE, s64 globalStep = DEFAULT_GLOBAL_STEP_VALUE)
```

## 函数功能<a name="zh-cn_topic_0000001939004906_section154mcpsimp"></a>

TransDataDef构造函数。

## 参数说明<a name="zh-cn_topic_0000001939004906_section157mcpsimp"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0000001939004906_table175mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001939004906_row182mcpsimp"><th class="cellrowborder" valign="top" width="28.71%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001939004906_p184mcpsimp"><a name="zh-cn_topic_0000001939004906_p184mcpsimp"></a><a name="zh-cn_topic_0000001939004906_p184mcpsimp"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="13.86%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001939004906_p186mcpsimp"><a name="zh-cn_topic_0000001939004906_p186mcpsimp"></a><a name="zh-cn_topic_0000001939004906_p186mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="57.43000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001939004906_p188mcpsimp"><a name="zh-cn_topic_0000001939004906_p188mcpsimp"></a><a name="zh-cn_topic_0000001939004906_p188mcpsimp"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001939004906_row190mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001939004906_p192mcpsimp"><a name="zh-cn_topic_0000001939004906_p192mcpsimp"></a><a name="zh-cn_topic_0000001939004906_p192mcpsimp"></a>u64 srcBuf</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001939004906_p194mcpsimp"><a name="zh-cn_topic_0000001939004906_p194mcpsimp"></a><a name="zh-cn_topic_0000001939004906_p194mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001939004906_p196mcpsimp"><a name="zh-cn_topic_0000001939004906_p196mcpsimp"></a><a name="zh-cn_topic_0000001939004906_p196mcpsimp"></a>源地址</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001939004906_row197mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001939004906_p199mcpsimp"><a name="zh-cn_topic_0000001939004906_p199mcpsimp"></a><a name="zh-cn_topic_0000001939004906_p199mcpsimp"></a>u64 dstBuf</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001939004906_p201mcpsimp"><a name="zh-cn_topic_0000001939004906_p201mcpsimp"></a><a name="zh-cn_topic_0000001939004906_p201mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001939004906_p203mcpsimp"><a name="zh-cn_topic_0000001939004906_p203mcpsimp"></a><a name="zh-cn_topic_0000001939004906_p203mcpsimp"></a>目的地址</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001939004906_row204mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001939004906_p206mcpsimp"><a name="zh-cn_topic_0000001939004906_p206mcpsimp"></a><a name="zh-cn_topic_0000001939004906_p206mcpsimp"></a>u64 count</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001939004906_p208mcpsimp"><a name="zh-cn_topic_0000001939004906_p208mcpsimp"></a><a name="zh-cn_topic_0000001939004906_p208mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001939004906_p210mcpsimp"><a name="zh-cn_topic_0000001939004906_p210mcpsimp"></a><a name="zh-cn_topic_0000001939004906_p210mcpsimp"></a>数据量</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001939004906_row211mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001939004906_p213mcpsimp"><a name="zh-cn_topic_0000001939004906_p213mcpsimp"></a><a name="zh-cn_topic_0000001939004906_p213mcpsimp"></a>HcclDataType dataType</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001939004906_p215mcpsimp"><a name="zh-cn_topic_0000001939004906_p215mcpsimp"></a><a name="zh-cn_topic_0000001939004906_p215mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001939004906_p217mcpsimp"><a name="zh-cn_topic_0000001939004906_p217mcpsimp"></a><a name="zh-cn_topic_0000001939004906_p217mcpsimp"></a>数据类型</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001939004906_row218mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001939004906_p220mcpsimp"><a name="zh-cn_topic_0000001939004906_p220mcpsimp"></a><a name="zh-cn_topic_0000001939004906_p220mcpsimp"></a>bool errorFlag</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001939004906_p222mcpsimp"><a name="zh-cn_topic_0000001939004906_p222mcpsimp"></a><a name="zh-cn_topic_0000001939004906_p222mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001939004906_p224mcpsimp"><a name="zh-cn_topic_0000001939004906_p224mcpsimp"></a><a name="zh-cn_topic_0000001939004906_p224mcpsimp"></a>Error标记</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001939004906_row225mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001939004906_p227mcpsimp"><a name="zh-cn_topic_0000001939004906_p227mcpsimp"></a><a name="zh-cn_topic_0000001939004906_p227mcpsimp"></a>u32 tableId</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001939004906_p229mcpsimp"><a name="zh-cn_topic_0000001939004906_p229mcpsimp"></a><a name="zh-cn_topic_0000001939004906_p229mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001939004906_p231mcpsimp"><a name="zh-cn_topic_0000001939004906_p231mcpsimp"></a><a name="zh-cn_topic_0000001939004906_p231mcpsimp"></a>Table id</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001939004906_row232mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001939004906_p234mcpsimp"><a name="zh-cn_topic_0000001939004906_p234mcpsimp"></a><a name="zh-cn_topic_0000001939004906_p234mcpsimp"></a>s64 globalStep</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001939004906_p236mcpsimp"><a name="zh-cn_topic_0000001939004906_p236mcpsimp"></a><a name="zh-cn_topic_0000001939004906_p236mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001939004906_p238mcpsimp"><a name="zh-cn_topic_0000001939004906_p238mcpsimp"></a><a name="zh-cn_topic_0000001939004906_p238mcpsimp"></a>全局step</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001939004906_section160mcpsimp"></a>

无。

## 约束说明<a name="zh-cn_topic_0000001939004906_section163mcpsimp"></a>

无。

