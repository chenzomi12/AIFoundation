# InitTask 

## 函数原型<a name="zh-cn_topic_0000001953823225_section2068mcpsimp"></a>

```
HcclDispatcher dispatcherPtr, hccl::Stream &stream, const hccl::HcclOpMetaInfo &opMetaInfo)
```

## 功能说明<a name="zh-cn_topic_0000001953823225_section2070mcpsimp"></a>

初始化task。

## 参数说明<a name="zh-cn_topic_0000001953823225_section2072mcpsimp"></a>

<a name="zh-cn_topic_0000001953823225_table2073mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001953823225_row2079mcpsimp"><th class="cellrowborder" valign="top" width="56.99999999999999%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001953823225_p2081mcpsimp"><a name="zh-cn_topic_0000001953823225_p2081mcpsimp"></a><a name="zh-cn_topic_0000001953823225_p2081mcpsimp"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="18%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001953823225_p2083mcpsimp"><a name="zh-cn_topic_0000001953823225_p2083mcpsimp"></a><a name="zh-cn_topic_0000001953823225_p2083mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="25%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001953823225_p2085mcpsimp"><a name="zh-cn_topic_0000001953823225_p2085mcpsimp"></a><a name="zh-cn_topic_0000001953823225_p2085mcpsimp"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001953823225_row2087mcpsimp"><td class="cellrowborder" valign="top" width="56.99999999999999%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001953823225_p2089mcpsimp"><a name="zh-cn_topic_0000001953823225_p2089mcpsimp"></a><a name="zh-cn_topic_0000001953823225_p2089mcpsimp"></a>HcclDispatcher dispatcher</p>
</td>
<td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001953823225_p2091mcpsimp"><a name="zh-cn_topic_0000001953823225_p2091mcpsimp"></a><a name="zh-cn_topic_0000001953823225_p2091mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="25%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001953823225_p2093mcpsimp"><a name="zh-cn_topic_0000001953823225_p2093mcpsimp"></a><a name="zh-cn_topic_0000001953823225_p2093mcpsimp"></a>dispatcher handle</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001953823225_row3363104742310"><td class="cellrowborder" valign="top" width="56.99999999999999%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001953823225_p1736344772313"><a name="zh-cn_topic_0000001953823225_p1736344772313"></a><a name="zh-cn_topic_0000001953823225_p1736344772313"></a>hccl::Stream &amp;stream</p>
</td>
<td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001953823225_p1936418472235"><a name="zh-cn_topic_0000001953823225_p1936418472235"></a><a name="zh-cn_topic_0000001953823225_p1936418472235"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="25%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001953823225_p436444752319"><a name="zh-cn_topic_0000001953823225_p436444752319"></a><a name="zh-cn_topic_0000001953823225_p436444752319"></a>stream对象</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001953823225_row2094mcpsimp"><td class="cellrowborder" valign="top" width="56.99999999999999%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001953823225_p2096mcpsimp"><a name="zh-cn_topic_0000001953823225_p2096mcpsimp"></a><a name="zh-cn_topic_0000001953823225_p2096mcpsimp"></a>const hccl::HcclOpMetaInfo &amp;opMetaInfo</p>
</td>
<td class="cellrowborder" valign="top" width="18%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001953823225_p2098mcpsimp"><a name="zh-cn_topic_0000001953823225_p2098mcpsimp"></a><a name="zh-cn_topic_0000001953823225_p2098mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="25%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001953823225_p2100mcpsimp"><a name="zh-cn_topic_0000001953823225_p2100mcpsimp"></a><a name="zh-cn_topic_0000001953823225_p2100mcpsimp"></a>opinfo</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001953823225_section2101mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS；其他失败。

