# HcclSetGlobalWorkSpace 

## 函数原型<a name="zh-cn_topic_0000001953823213_section292mcpsimp"></a>

```
HcclResult HcclSetGlobalWorkSpace(HcclDispatcher dispatcherPtr, std::vector<void *> &globalWorkSpaceAddr)
```

## 功能说明<a name="zh-cn_topic_0000001953823213_section294mcpsimp"></a>

设置global workspace mem。

## 参数说明<a name="zh-cn_topic_0000001953823213_section296mcpsimp"></a>

<a name="zh-cn_topic_0000001953823213_table297mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001953823213_row303mcpsimp"><th class="cellrowborder" valign="top" width="55.00000000000001%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001953823213_p305mcpsimp"><a name="zh-cn_topic_0000001953823213_p305mcpsimp"></a><a name="zh-cn_topic_0000001953823213_p305mcpsimp"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="16%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001953823213_p307mcpsimp"><a name="zh-cn_topic_0000001953823213_p307mcpsimp"></a><a name="zh-cn_topic_0000001953823213_p307mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="28.999999999999996%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001953823213_p309mcpsimp"><a name="zh-cn_topic_0000001953823213_p309mcpsimp"></a><a name="zh-cn_topic_0000001953823213_p309mcpsimp"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001953823213_row311mcpsimp"><td class="cellrowborder" valign="top" width="55.00000000000001%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001953823213_p313mcpsimp"><a name="zh-cn_topic_0000001953823213_p313mcpsimp"></a><a name="zh-cn_topic_0000001953823213_p313mcpsimp"></a>HcclDispatcher dispatcher</p>
</td>
<td class="cellrowborder" valign="top" width="16%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001953823213_p315mcpsimp"><a name="zh-cn_topic_0000001953823213_p315mcpsimp"></a><a name="zh-cn_topic_0000001953823213_p315mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="28.999999999999996%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001953823213_p317mcpsimp"><a name="zh-cn_topic_0000001953823213_p317mcpsimp"></a><a name="zh-cn_topic_0000001953823213_p317mcpsimp"></a>dispatcher handle</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001953823213_row318mcpsimp"><td class="cellrowborder" valign="top" width="55.00000000000001%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001953823213_p320mcpsimp"><a name="zh-cn_topic_0000001953823213_p320mcpsimp"></a><a name="zh-cn_topic_0000001953823213_p320mcpsimp"></a>std::vector&lt;void *&gt; &amp;globalWorkSpaceAddr</p>
</td>
<td class="cellrowborder" valign="top" width="16%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001953823213_p322mcpsimp"><a name="zh-cn_topic_0000001953823213_p322mcpsimp"></a><a name="zh-cn_topic_0000001953823213_p322mcpsimp"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="28.999999999999996%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001953823213_p324mcpsimp"><a name="zh-cn_topic_0000001953823213_p324mcpsimp"></a><a name="zh-cn_topic_0000001953823213_p324mcpsimp"></a>global workspace mem</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001953823213_section325mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS；其他失败。

