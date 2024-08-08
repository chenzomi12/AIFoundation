# HcclSetNotifyWaitMode 

## 函数原型<a name="zh-cn_topic_0000001953823217_section1046mcpsimp"></a>

```
HcclResult HcclSetNotifyWaitMode(HcclDispatcher dispatcherPtr, const SyncMode notifyWaitMode)
```

## 功能说明<a name="zh-cn_topic_0000001953823217_section1048mcpsimp"></a>

设置notify wait工作模式。

## 参数说明<a name="zh-cn_topic_0000001953823217_section1050mcpsimp"></a>

<a name="zh-cn_topic_0000001953823217_table1051mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001953823217_row1057mcpsimp"><th class="cellrowborder" valign="top" width="49%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001953823217_p1059mcpsimp"><a name="zh-cn_topic_0000001953823217_p1059mcpsimp"></a><a name="zh-cn_topic_0000001953823217_p1059mcpsimp"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="19%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001953823217_p1061mcpsimp"><a name="zh-cn_topic_0000001953823217_p1061mcpsimp"></a><a name="zh-cn_topic_0000001953823217_p1061mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="32%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001953823217_p1063mcpsimp"><a name="zh-cn_topic_0000001953823217_p1063mcpsimp"></a><a name="zh-cn_topic_0000001953823217_p1063mcpsimp"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001953823217_row1065mcpsimp"><td class="cellrowborder" valign="top" width="49%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001953823217_p1067mcpsimp"><a name="zh-cn_topic_0000001953823217_p1067mcpsimp"></a><a name="zh-cn_topic_0000001953823217_p1067mcpsimp"></a>HcclDispatcher dispatcher</p>
</td>
<td class="cellrowborder" valign="top" width="19%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001953823217_p1069mcpsimp"><a name="zh-cn_topic_0000001953823217_p1069mcpsimp"></a><a name="zh-cn_topic_0000001953823217_p1069mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="32%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001953823217_p1071mcpsimp"><a name="zh-cn_topic_0000001953823217_p1071mcpsimp"></a><a name="zh-cn_topic_0000001953823217_p1071mcpsimp"></a>dispatcher handle</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001953823217_row1072mcpsimp"><td class="cellrowborder" valign="top" width="49%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001953823217_p1074mcpsimp"><a name="zh-cn_topic_0000001953823217_p1074mcpsimp"></a><a name="zh-cn_topic_0000001953823217_p1074mcpsimp"></a>const SyncMode notifyWaitMode</p>
</td>
<td class="cellrowborder" valign="top" width="19%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001953823217_p1076mcpsimp"><a name="zh-cn_topic_0000001953823217_p1076mcpsimp"></a><a name="zh-cn_topic_0000001953823217_p1076mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="32%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001953823217_p1078mcpsimp"><a name="zh-cn_topic_0000001953823217_p1078mcpsimp"></a><a name="zh-cn_topic_0000001953823217_p1078mcpsimp"></a>notify wait工作模式</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001953823217_section1079mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS；其他失败。

