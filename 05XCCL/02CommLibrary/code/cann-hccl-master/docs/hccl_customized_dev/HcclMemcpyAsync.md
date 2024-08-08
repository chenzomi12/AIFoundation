# HcclMemcpyAsync 

## 函数原型<a name="zh-cn_topic_0000001953703441_section2491mcpsimp"></a>

```
HcclResult HcclMemcpyAsync(HcclDispatcher dispatcherPtr, void *dst, const uint64_t destMax, const void *src, const uint64_t count, const HcclRtMemcpyKind kind, hccl::Stream &stream, const u32 remoteUserRank, hccl::LinkType linkType)
```

## 功能说明<a name="zh-cn_topic_0000001953703441_section2493mcpsimp"></a>

异步内存copy。

## 参数说明<a name="zh-cn_topic_0000001953703441_section2495mcpsimp"></a>

<a name="zh-cn_topic_0000001953703441_table2496mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001953703441_row2502mcpsimp"><th class="cellrowborder" valign="top" width="50%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001953703441_p2504mcpsimp"><a name="zh-cn_topic_0000001953703441_p2504mcpsimp"></a><a name="zh-cn_topic_0000001953703441_p2504mcpsimp"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="21%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001953703441_p2506mcpsimp"><a name="zh-cn_topic_0000001953703441_p2506mcpsimp"></a><a name="zh-cn_topic_0000001953703441_p2506mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="28.999999999999996%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001953703441_p2508mcpsimp"><a name="zh-cn_topic_0000001953703441_p2508mcpsimp"></a><a name="zh-cn_topic_0000001953703441_p2508mcpsimp"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001953703441_row2510mcpsimp"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001953703441_p2512mcpsimp"><a name="zh-cn_topic_0000001953703441_p2512mcpsimp"></a><a name="zh-cn_topic_0000001953703441_p2512mcpsimp"></a>HcclDispatcher dispatcher</p>
</td>
<td class="cellrowborder" valign="top" width="21%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001953703441_p2514mcpsimp"><a name="zh-cn_topic_0000001953703441_p2514mcpsimp"></a><a name="zh-cn_topic_0000001953703441_p2514mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="28.999999999999996%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001953703441_p2516mcpsimp"><a name="zh-cn_topic_0000001953703441_p2516mcpsimp"></a><a name="zh-cn_topic_0000001953703441_p2516mcpsimp"></a>dispatcher handle</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001953703441_row2517mcpsimp"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001953703441_p2519mcpsimp"><a name="zh-cn_topic_0000001953703441_p2519mcpsimp"></a><a name="zh-cn_topic_0000001953703441_p2519mcpsimp"></a>void *dst</p>
</td>
<td class="cellrowborder" valign="top" width="21%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001953703441_p2521mcpsimp"><a name="zh-cn_topic_0000001953703441_p2521mcpsimp"></a><a name="zh-cn_topic_0000001953703441_p2521mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="28.999999999999996%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001953703441_p2523mcpsimp"><a name="zh-cn_topic_0000001953703441_p2523mcpsimp"></a><a name="zh-cn_topic_0000001953703441_p2523mcpsimp"></a>dst内存地址</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001953703441_row2524mcpsimp"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001953703441_p2526mcpsimp"><a name="zh-cn_topic_0000001953703441_p2526mcpsimp"></a><a name="zh-cn_topic_0000001953703441_p2526mcpsimp"></a>const uint64_t destMax</p>
</td>
<td class="cellrowborder" valign="top" width="21%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001953703441_p2528mcpsimp"><a name="zh-cn_topic_0000001953703441_p2528mcpsimp"></a><a name="zh-cn_topic_0000001953703441_p2528mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="28.999999999999996%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001953703441_p2530mcpsimp"><a name="zh-cn_topic_0000001953703441_p2530mcpsimp"></a><a name="zh-cn_topic_0000001953703441_p2530mcpsimp"></a>dst内存大小</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001953703441_row2531mcpsimp"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001953703441_p2533mcpsimp"><a name="zh-cn_topic_0000001953703441_p2533mcpsimp"></a><a name="zh-cn_topic_0000001953703441_p2533mcpsimp"></a>const void *src</p>
</td>
<td class="cellrowborder" valign="top" width="21%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001953703441_p2535mcpsimp"><a name="zh-cn_topic_0000001953703441_p2535mcpsimp"></a><a name="zh-cn_topic_0000001953703441_p2535mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="28.999999999999996%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001953703441_p2537mcpsimp"><a name="zh-cn_topic_0000001953703441_p2537mcpsimp"></a><a name="zh-cn_topic_0000001953703441_p2537mcpsimp"></a>src内存地址</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001953703441_row2538mcpsimp"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001953703441_p2540mcpsimp"><a name="zh-cn_topic_0000001953703441_p2540mcpsimp"></a><a name="zh-cn_topic_0000001953703441_p2540mcpsimp"></a>const uint64_t count</p>
</td>
<td class="cellrowborder" valign="top" width="21%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001953703441_p2542mcpsimp"><a name="zh-cn_topic_0000001953703441_p2542mcpsimp"></a><a name="zh-cn_topic_0000001953703441_p2542mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="28.999999999999996%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001953703441_p2544mcpsimp"><a name="zh-cn_topic_0000001953703441_p2544mcpsimp"></a><a name="zh-cn_topic_0000001953703441_p2544mcpsimp"></a>src内存大小</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001953703441_row2545mcpsimp"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001953703441_p2547mcpsimp"><a name="zh-cn_topic_0000001953703441_p2547mcpsimp"></a><a name="zh-cn_topic_0000001953703441_p2547mcpsimp"></a>const HcclRtMemcpyKind kind</p>
</td>
<td class="cellrowborder" valign="top" width="21%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001953703441_p2549mcpsimp"><a name="zh-cn_topic_0000001953703441_p2549mcpsimp"></a><a name="zh-cn_topic_0000001953703441_p2549mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="28.999999999999996%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001953703441_p2551mcpsimp"><a name="zh-cn_topic_0000001953703441_p2551mcpsimp"></a><a name="zh-cn_topic_0000001953703441_p2551mcpsimp"></a>内存copy类型</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001953703441_row2552mcpsimp"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001953703441_p2554mcpsimp"><a name="zh-cn_topic_0000001953703441_p2554mcpsimp"></a><a name="zh-cn_topic_0000001953703441_p2554mcpsimp"></a>hccl::Stream &amp;stream</p>
</td>
<td class="cellrowborder" valign="top" width="21%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001953703441_p2556mcpsimp"><a name="zh-cn_topic_0000001953703441_p2556mcpsimp"></a><a name="zh-cn_topic_0000001953703441_p2556mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="28.999999999999996%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001953703441_p2558mcpsimp"><a name="zh-cn_topic_0000001953703441_p2558mcpsimp"></a><a name="zh-cn_topic_0000001953703441_p2558mcpsimp"></a>stream对象</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001953703441_row2559mcpsimp"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001953703441_p2561mcpsimp"><a name="zh-cn_topic_0000001953703441_p2561mcpsimp"></a><a name="zh-cn_topic_0000001953703441_p2561mcpsimp"></a>u32 remoteUserRank</p>
</td>
<td class="cellrowborder" valign="top" width="21%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001953703441_p2563mcpsimp"><a name="zh-cn_topic_0000001953703441_p2563mcpsimp"></a><a name="zh-cn_topic_0000001953703441_p2563mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="28.999999999999996%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001953703441_p2565mcpsimp"><a name="zh-cn_topic_0000001953703441_p2565mcpsimp"></a><a name="zh-cn_topic_0000001953703441_p2565mcpsimp"></a>对端world rank</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001953703441_row2566mcpsimp"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001953703441_p2568mcpsimp"><a name="zh-cn_topic_0000001953703441_p2568mcpsimp"></a><a name="zh-cn_topic_0000001953703441_p2568mcpsimp"></a>hccl::LinkType inLinkType</p>
</td>
<td class="cellrowborder" valign="top" width="21%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001953703441_p2570mcpsimp"><a name="zh-cn_topic_0000001953703441_p2570mcpsimp"></a><a name="zh-cn_topic_0000001953703441_p2570mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="28.999999999999996%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001953703441_p2572mcpsimp"><a name="zh-cn_topic_0000001953703441_p2572mcpsimp"></a><a name="zh-cn_topic_0000001953703441_p2572mcpsimp"></a>链路类型</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001953703441_section2573mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS；其他失败。

