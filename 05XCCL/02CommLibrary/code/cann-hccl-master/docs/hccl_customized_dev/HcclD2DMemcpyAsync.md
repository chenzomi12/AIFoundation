# HcclD2DMemcpyAsync 

## 函数原型<a name="zh-cn_topic_0000001926464492_section2911mcpsimp"></a>

```
HcclResult HcclD2DMemcpyAsync(HcclDispatcher dispatcherPtr, hccl::DeviceMem &dst, const hccl::DeviceMem &src, hccl::Stream &stream, u32 remoteUserRank = INVALID_VALUE_RANKID, hccl::LinkType inLinkType = hccl::LinkType::LINK_ONCHIP)
```

## 功能说明<a name="zh-cn_topic_0000001926464492_section2913mcpsimp"></a>

异步device间内存copy。

## 参数说明<a name="zh-cn_topic_0000001926464492_section2915mcpsimp"></a>

<a name="zh-cn_topic_0000001926464492_table2916mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001926464492_row2922mcpsimp"><th class="cellrowborder" valign="top" width="48.484848484848484%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001926464492_p2924mcpsimp"><a name="zh-cn_topic_0000001926464492_p2924mcpsimp"></a><a name="zh-cn_topic_0000001926464492_p2924mcpsimp"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="21.21212121212121%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001926464492_p2926mcpsimp"><a name="zh-cn_topic_0000001926464492_p2926mcpsimp"></a><a name="zh-cn_topic_0000001926464492_p2926mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="30.303030303030305%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001926464492_p2928mcpsimp"><a name="zh-cn_topic_0000001926464492_p2928mcpsimp"></a><a name="zh-cn_topic_0000001926464492_p2928mcpsimp"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001926464492_row2930mcpsimp"><td class="cellrowborder" valign="top" width="48.484848484848484%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001926464492_p2932mcpsimp"><a name="zh-cn_topic_0000001926464492_p2932mcpsimp"></a><a name="zh-cn_topic_0000001926464492_p2932mcpsimp"></a>HcclDispatcher dispatcher</p>
</td>
<td class="cellrowborder" valign="top" width="21.21212121212121%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001926464492_p2934mcpsimp"><a name="zh-cn_topic_0000001926464492_p2934mcpsimp"></a><a name="zh-cn_topic_0000001926464492_p2934mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="30.303030303030305%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001926464492_p2936mcpsimp"><a name="zh-cn_topic_0000001926464492_p2936mcpsimp"></a><a name="zh-cn_topic_0000001926464492_p2936mcpsimp"></a>dispatcher handle</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001926464492_row2937mcpsimp"><td class="cellrowborder" valign="top" width="48.484848484848484%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001926464492_p2939mcpsimp"><a name="zh-cn_topic_0000001926464492_p2939mcpsimp"></a><a name="zh-cn_topic_0000001926464492_p2939mcpsimp"></a>hccl::DeviceMem &amp;dst</p>
</td>
<td class="cellrowborder" valign="top" width="21.21212121212121%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001926464492_p2941mcpsimp"><a name="zh-cn_topic_0000001926464492_p2941mcpsimp"></a><a name="zh-cn_topic_0000001926464492_p2941mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="30.303030303030305%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001926464492_p2943mcpsimp"><a name="zh-cn_topic_0000001926464492_p2943mcpsimp"></a><a name="zh-cn_topic_0000001926464492_p2943mcpsimp"></a>dst内存对象</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001926464492_row2944mcpsimp"><td class="cellrowborder" valign="top" width="48.484848484848484%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001926464492_p2946mcpsimp"><a name="zh-cn_topic_0000001926464492_p2946mcpsimp"></a><a name="zh-cn_topic_0000001926464492_p2946mcpsimp"></a>const hccl::DeviceMem &amp;src</p>
</td>
<td class="cellrowborder" valign="top" width="21.21212121212121%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001926464492_p2948mcpsimp"><a name="zh-cn_topic_0000001926464492_p2948mcpsimp"></a><a name="zh-cn_topic_0000001926464492_p2948mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="30.303030303030305%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001926464492_p2950mcpsimp"><a name="zh-cn_topic_0000001926464492_p2950mcpsimp"></a><a name="zh-cn_topic_0000001926464492_p2950mcpsimp"></a>src内存对象</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001926464492_row2951mcpsimp"><td class="cellrowborder" valign="top" width="48.484848484848484%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001926464492_p2953mcpsimp"><a name="zh-cn_topic_0000001926464492_p2953mcpsimp"></a><a name="zh-cn_topic_0000001926464492_p2953mcpsimp"></a>hccl::Stream &amp;stream</p>
</td>
<td class="cellrowborder" valign="top" width="21.21212121212121%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001926464492_p2955mcpsimp"><a name="zh-cn_topic_0000001926464492_p2955mcpsimp"></a><a name="zh-cn_topic_0000001926464492_p2955mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="30.303030303030305%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001926464492_p2957mcpsimp"><a name="zh-cn_topic_0000001926464492_p2957mcpsimp"></a><a name="zh-cn_topic_0000001926464492_p2957mcpsimp"></a>stream对象</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001926464492_row2958mcpsimp"><td class="cellrowborder" valign="top" width="48.484848484848484%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001926464492_p2960mcpsimp"><a name="zh-cn_topic_0000001926464492_p2960mcpsimp"></a><a name="zh-cn_topic_0000001926464492_p2960mcpsimp"></a>u32 remoteUserRank</p>
</td>
<td class="cellrowborder" valign="top" width="21.21212121212121%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001926464492_p2962mcpsimp"><a name="zh-cn_topic_0000001926464492_p2962mcpsimp"></a><a name="zh-cn_topic_0000001926464492_p2962mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="30.303030303030305%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001926464492_p2964mcpsimp"><a name="zh-cn_topic_0000001926464492_p2964mcpsimp"></a><a name="zh-cn_topic_0000001926464492_p2964mcpsimp"></a>对端world rank</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001926464492_row2965mcpsimp"><td class="cellrowborder" valign="top" width="48.484848484848484%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001926464492_p2967mcpsimp"><a name="zh-cn_topic_0000001926464492_p2967mcpsimp"></a><a name="zh-cn_topic_0000001926464492_p2967mcpsimp"></a>hccl::LinkType inLinkType</p>
</td>
<td class="cellrowborder" valign="top" width="21.21212121212121%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001926464492_p2969mcpsimp"><a name="zh-cn_topic_0000001926464492_p2969mcpsimp"></a><a name="zh-cn_topic_0000001926464492_p2969mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="30.303030303030305%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001926464492_p2971mcpsimp"><a name="zh-cn_topic_0000001926464492_p2971mcpsimp"></a><a name="zh-cn_topic_0000001926464492_p2971mcpsimp"></a>链路类型</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001926464492_section2972mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS；其他失败。

