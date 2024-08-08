# RxAck 

## 函数原型<a name="zh-cn_topic_0000001956618581_section7408mcpsimp"></a>

```
HcclResult RxAck(Stream &stream)
```

## 函数功能<a name="zh-cn_topic_0000001956618581_section7411mcpsimp"></a>

本端等待对端的同步信号。

## 参数说明<a name="zh-cn_topic_0000001956618581_section7414mcpsimp"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0000001956618581_table7416mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001956618581_row7423mcpsimp"><th class="cellrowborder" valign="top" width="28.71%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001956618581_p7425mcpsimp"><a name="zh-cn_topic_0000001956618581_p7425mcpsimp"></a><a name="zh-cn_topic_0000001956618581_p7425mcpsimp"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="13.86%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001956618581_p7427mcpsimp"><a name="zh-cn_topic_0000001956618581_p7427mcpsimp"></a><a name="zh-cn_topic_0000001956618581_p7427mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="57.43000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001956618581_p7429mcpsimp"><a name="zh-cn_topic_0000001956618581_p7429mcpsimp"></a><a name="zh-cn_topic_0000001956618581_p7429mcpsimp"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001956618581_row7431mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956618581_p7433mcpsimp"><a name="zh-cn_topic_0000001956618581_p7433mcpsimp"></a><a name="zh-cn_topic_0000001956618581_p7433mcpsimp"></a>Stream &amp;stream</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956618581_p7435mcpsimp"><a name="zh-cn_topic_0000001956618581_p7435mcpsimp"></a><a name="zh-cn_topic_0000001956618581_p7435mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956618581_p7437mcpsimp"><a name="zh-cn_topic_0000001956618581_p7437mcpsimp"></a><a name="zh-cn_topic_0000001956618581_p7437mcpsimp"></a>Stream对象</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001956618581_section7438mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

## 约束说明<a name="zh-cn_topic_0000001956618581_section7441mcpsimp"></a>

无

