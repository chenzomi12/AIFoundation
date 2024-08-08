# HcclGetNotifyWaitMode 

## 函数原型<a name="zh-cn_topic_0000001926623844_section479mcpsimp"></a>

```
HcclResult HcclGetNotifyWaitMode(HcclDispatcher dispatcherPtr, SyncMode *notifyWaitMode)
```

## 功能说明<a name="zh-cn_topic_0000001926623844_section481mcpsimp"></a>

获取notify wait工作模式。

## 参数说明<a name="zh-cn_topic_0000001926623844_section483mcpsimp"></a>

<a name="zh-cn_topic_0000001926623844_table484mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001926623844_row490mcpsimp"><th class="cellrowborder" valign="top" width="46%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001926623844_p492mcpsimp"><a name="zh-cn_topic_0000001926623844_p492mcpsimp"></a><a name="zh-cn_topic_0000001926623844_p492mcpsimp"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="20%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001926623844_p494mcpsimp"><a name="zh-cn_topic_0000001926623844_p494mcpsimp"></a><a name="zh-cn_topic_0000001926623844_p494mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="34%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001926623844_p496mcpsimp"><a name="zh-cn_topic_0000001926623844_p496mcpsimp"></a><a name="zh-cn_topic_0000001926623844_p496mcpsimp"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001926623844_row498mcpsimp"><td class="cellrowborder" valign="top" width="46%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001926623844_p500mcpsimp"><a name="zh-cn_topic_0000001926623844_p500mcpsimp"></a><a name="zh-cn_topic_0000001926623844_p500mcpsimp"></a>HcclDispatcher dispatcher</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001926623844_p502mcpsimp"><a name="zh-cn_topic_0000001926623844_p502mcpsimp"></a><a name="zh-cn_topic_0000001926623844_p502mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="34%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001926623844_p504mcpsimp"><a name="zh-cn_topic_0000001926623844_p504mcpsimp"></a><a name="zh-cn_topic_0000001926623844_p504mcpsimp"></a>dispatcher handle</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001926623844_row505mcpsimp"><td class="cellrowborder" valign="top" width="46%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001926623844_p507mcpsimp"><a name="zh-cn_topic_0000001926623844_p507mcpsimp"></a><a name="zh-cn_topic_0000001926623844_p507mcpsimp"></a>SyncMode *notifyWaitMode</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001926623844_p509mcpsimp"><a name="zh-cn_topic_0000001926623844_p509mcpsimp"></a><a name="zh-cn_topic_0000001926623844_p509mcpsimp"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="34%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001926623844_p511mcpsimp"><a name="zh-cn_topic_0000001926623844_p511mcpsimp"></a><a name="zh-cn_topic_0000001926623844_p511mcpsimp"></a>notify wait工作模式</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001926623844_section512mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS；其他失败。

