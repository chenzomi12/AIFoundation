# Init 

## 函数原型<a name="zh-cn_topic_0000001956458641_section2942mcpsimp"></a>

```
HcclResult Init(const NotifyLoadType type = NotifyLoadType::HOST_NOTIFY)
HcclResult Init(const HcclSignalInfo &notifyInfo, const NotifyLoadType type = NotifyLoadType::DEVICE_NOTIFY)
```

## 函数功能<a name="zh-cn_topic_0000001956458641_section2945mcpsimp"></a>

Notify初始化。

## 参数说明<a name="zh-cn_topic_0000001956458641_section2948mcpsimp"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0000001956458641_table2950mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001956458641_row2957mcpsimp"><th class="cellrowborder" valign="top" width="28.71%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001956458641_p2959mcpsimp"><a name="zh-cn_topic_0000001956458641_p2959mcpsimp"></a><a name="zh-cn_topic_0000001956458641_p2959mcpsimp"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="13.86%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001956458641_p2961mcpsimp"><a name="zh-cn_topic_0000001956458641_p2961mcpsimp"></a><a name="zh-cn_topic_0000001956458641_p2961mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="57.43000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001956458641_p2963mcpsimp"><a name="zh-cn_topic_0000001956458641_p2963mcpsimp"></a><a name="zh-cn_topic_0000001956458641_p2963mcpsimp"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001956458641_row2965mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956458641_p2967mcpsimp"><a name="zh-cn_topic_0000001956458641_p2967mcpsimp"></a><a name="zh-cn_topic_0000001956458641_p2967mcpsimp"></a>const NotifyLoadType type</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956458641_p2969mcpsimp"><a name="zh-cn_topic_0000001956458641_p2969mcpsimp"></a><a name="zh-cn_topic_0000001956458641_p2969mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956458641_p2971mcpsimp"><a name="zh-cn_topic_0000001956458641_p2971mcpsimp"></a><a name="zh-cn_topic_0000001956458641_p2971mcpsimp"></a>Notify任务类型</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001956458641_row106247278352"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956458641_p3004mcpsimp"><a name="zh-cn_topic_0000001956458641_p3004mcpsimp"></a><a name="zh-cn_topic_0000001956458641_p3004mcpsimp"></a>const HcclSignalInfo &amp;notifyInfo</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956458641_p3006mcpsimp"><a name="zh-cn_topic_0000001956458641_p3006mcpsimp"></a><a name="zh-cn_topic_0000001956458641_p3006mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956458641_p3008mcpsimp"><a name="zh-cn_topic_0000001956458641_p3008mcpsimp"></a><a name="zh-cn_topic_0000001956458641_p3008mcpsimp"></a>Notify信息</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001956458641_row1528814306358"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956458641_p3011mcpsimp"><a name="zh-cn_topic_0000001956458641_p3011mcpsimp"></a><a name="zh-cn_topic_0000001956458641_p3011mcpsimp"></a>const NotifyLoadType type</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956458641_p3013mcpsimp"><a name="zh-cn_topic_0000001956458641_p3013mcpsimp"></a><a name="zh-cn_topic_0000001956458641_p3013mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956458641_p3015mcpsimp"><a name="zh-cn_topic_0000001956458641_p3015mcpsimp"></a><a name="zh-cn_topic_0000001956458641_p3015mcpsimp"></a>Notify任务类型</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001956458641_section2972mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

## 约束说明<a name="zh-cn_topic_0000001956458641_section2975mcpsimp"></a>

无

