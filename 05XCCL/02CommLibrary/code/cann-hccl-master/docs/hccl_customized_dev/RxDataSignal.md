# RxDataSignal 

## 函数原型<a name="zh-cn_topic_0000001956618561_section6762mcpsimp"></a>

```
HcclResult RxDataSignal(Stream &stream)
```

## 函数功能<a name="zh-cn_topic_0000001956618561_section6765mcpsimp"></a>

本端等待对端的同步信号。

## 参数说明<a name="zh-cn_topic_0000001956618561_section6768mcpsimp"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0000001956618561_table6770mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001956618561_row6777mcpsimp"><th class="cellrowborder" valign="top" width="28.71%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001956618561_p6779mcpsimp"><a name="zh-cn_topic_0000001956618561_p6779mcpsimp"></a><a name="zh-cn_topic_0000001956618561_p6779mcpsimp"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="13.86%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001956618561_p6781mcpsimp"><a name="zh-cn_topic_0000001956618561_p6781mcpsimp"></a><a name="zh-cn_topic_0000001956618561_p6781mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="57.43000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001956618561_p6783mcpsimp"><a name="zh-cn_topic_0000001956618561_p6783mcpsimp"></a><a name="zh-cn_topic_0000001956618561_p6783mcpsimp"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001956618561_row6785mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956618561_p6787mcpsimp"><a name="zh-cn_topic_0000001956618561_p6787mcpsimp"></a><a name="zh-cn_topic_0000001956618561_p6787mcpsimp"></a>Stream &amp;stream</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956618561_p6789mcpsimp"><a name="zh-cn_topic_0000001956618561_p6789mcpsimp"></a><a name="zh-cn_topic_0000001956618561_p6789mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956618561_p6791mcpsimp"><a name="zh-cn_topic_0000001956618561_p6791mcpsimp"></a><a name="zh-cn_topic_0000001956618561_p6791mcpsimp"></a>Stream对象</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001956618561_section6792mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

## 约束说明<a name="zh-cn_topic_0000001956618561_section6795mcpsimp"></a>

无

