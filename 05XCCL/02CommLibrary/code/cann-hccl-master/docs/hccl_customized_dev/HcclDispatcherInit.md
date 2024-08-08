# HcclDispatcherInit 

## 函数原型<a name="zh-cn_topic_0000001926464484_section2685mcpsimp"></a>

```
HcclResult HcclDispatcherInit(DispatcherType type, const s32 deviceLogicId, const std::shared_ptr<hccl::ProfilerManager> &profilerManager, HcclDispatcher *dispatcher)
```

## 功能说明<a name="zh-cn_topic_0000001926464484_section2687mcpsimp"></a>

初始化dispatcher。

## 参数说明<a name="zh-cn_topic_0000001926464484_section2689mcpsimp"></a>

<a name="zh-cn_topic_0000001926464484_table2690mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001926464484_row2696mcpsimp"><th class="cellrowborder" valign="top" width="62%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001926464484_p2698mcpsimp"><a name="zh-cn_topic_0000001926464484_p2698mcpsimp"></a><a name="zh-cn_topic_0000001926464484_p2698mcpsimp"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="15%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001926464484_p2700mcpsimp"><a name="zh-cn_topic_0000001926464484_p2700mcpsimp"></a><a name="zh-cn_topic_0000001926464484_p2700mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="23%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001926464484_p2702mcpsimp"><a name="zh-cn_topic_0000001926464484_p2702mcpsimp"></a><a name="zh-cn_topic_0000001926464484_p2702mcpsimp"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001926464484_row2704mcpsimp"><td class="cellrowborder" valign="top" width="62%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001926464484_p2706mcpsimp"><a name="zh-cn_topic_0000001926464484_p2706mcpsimp"></a><a name="zh-cn_topic_0000001926464484_p2706mcpsimp"></a>DispatcherType type</p>
</td>
<td class="cellrowborder" valign="top" width="15%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001926464484_p2708mcpsimp"><a name="zh-cn_topic_0000001926464484_p2708mcpsimp"></a><a name="zh-cn_topic_0000001926464484_p2708mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="23%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001926464484_p2710mcpsimp"><a name="zh-cn_topic_0000001926464484_p2710mcpsimp"></a><a name="zh-cn_topic_0000001926464484_p2710mcpsimp"></a>dispatcher 类型</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001926464484_row2711mcpsimp"><td class="cellrowborder" valign="top" width="62%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001926464484_p2713mcpsimp"><a name="zh-cn_topic_0000001926464484_p2713mcpsimp"></a><a name="zh-cn_topic_0000001926464484_p2713mcpsimp"></a>const s32 deviceLogicId</p>
</td>
<td class="cellrowborder" valign="top" width="15%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001926464484_p2715mcpsimp"><a name="zh-cn_topic_0000001926464484_p2715mcpsimp"></a><a name="zh-cn_topic_0000001926464484_p2715mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="23%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001926464484_p2717mcpsimp"><a name="zh-cn_topic_0000001926464484_p2717mcpsimp"></a><a name="zh-cn_topic_0000001926464484_p2717mcpsimp"></a>deviceLogicId</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001926464484_row2718mcpsimp"><td class="cellrowborder" valign="top" width="62%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001926464484_p2720mcpsimp"><a name="zh-cn_topic_0000001926464484_p2720mcpsimp"></a><a name="zh-cn_topic_0000001926464484_p2720mcpsimp"></a>const std::shared_ptr&lt;hccl::ProfilerManager&gt; &amp;profilerManager</p>
</td>
<td class="cellrowborder" valign="top" width="15%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001926464484_p2722mcpsimp"><a name="zh-cn_topic_0000001926464484_p2722mcpsimp"></a><a name="zh-cn_topic_0000001926464484_p2722mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="23%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001926464484_p2724mcpsimp"><a name="zh-cn_topic_0000001926464484_p2724mcpsimp"></a><a name="zh-cn_topic_0000001926464484_p2724mcpsimp"></a>ProfilerManager</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001926464484_row2725mcpsimp"><td class="cellrowborder" valign="top" width="62%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001926464484_p2727mcpsimp"><a name="zh-cn_topic_0000001926464484_p2727mcpsimp"></a><a name="zh-cn_topic_0000001926464484_p2727mcpsimp"></a>HcclDispatcher *dispatcher</p>
</td>
<td class="cellrowborder" valign="top" width="15%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001926464484_p2729mcpsimp"><a name="zh-cn_topic_0000001926464484_p2729mcpsimp"></a><a name="zh-cn_topic_0000001926464484_p2729mcpsimp"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="23%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001926464484_p2731mcpsimp"><a name="zh-cn_topic_0000001926464484_p2731mcpsimp"></a><a name="zh-cn_topic_0000001926464484_p2731mcpsimp"></a>dispatcher handle</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001926464484_section2732mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS；其他失败。

