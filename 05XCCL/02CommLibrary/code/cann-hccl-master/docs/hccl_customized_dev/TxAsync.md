# TxAsync 

## 函数原型<a name="zh-cn_topic_0000001929299926_section6799mcpsimp"></a>

```
HcclResult TxAsync(UserMemType dstMemType, u64 dstOffset, const void *src, u64 len, Stream &stream)

HcclResult TxAsync(std::vector<TxMemoryInfo>& txMems, Stream &stream)
```

## 函数功能<a name="zh-cn_topic_0000001929299926_section6802mcpsimp"></a>

异步发送数据，将本端src地址的数据发送到远端指定类型地址中。

## 参数说明<a name="zh-cn_topic_0000001929299926_section6805mcpsimp"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0000001929299926_table6807mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001929299926_row6814mcpsimp"><th class="cellrowborder" valign="top" width="28.71%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001929299926_p6816mcpsimp"><a name="zh-cn_topic_0000001929299926_p6816mcpsimp"></a><a name="zh-cn_topic_0000001929299926_p6816mcpsimp"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="13.86%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001929299926_p6818mcpsimp"><a name="zh-cn_topic_0000001929299926_p6818mcpsimp"></a><a name="zh-cn_topic_0000001929299926_p6818mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="57.43000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001929299926_p6820mcpsimp"><a name="zh-cn_topic_0000001929299926_p6820mcpsimp"></a><a name="zh-cn_topic_0000001929299926_p6820mcpsimp"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001929299926_row6822mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299926_p6824mcpsimp"><a name="zh-cn_topic_0000001929299926_p6824mcpsimp"></a><a name="zh-cn_topic_0000001929299926_p6824mcpsimp"></a>UserMemType dstMemType</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299926_p6826mcpsimp"><a name="zh-cn_topic_0000001929299926_p6826mcpsimp"></a><a name="zh-cn_topic_0000001929299926_p6826mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299926_p6828mcpsimp"><a name="zh-cn_topic_0000001929299926_p6828mcpsimp"></a><a name="zh-cn_topic_0000001929299926_p6828mcpsimp"></a>对端用户内存类型</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929299926_row6829mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299926_p6831mcpsimp"><a name="zh-cn_topic_0000001929299926_p6831mcpsimp"></a><a name="zh-cn_topic_0000001929299926_p6831mcpsimp"></a>u64 dstOffset</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299926_p6833mcpsimp"><a name="zh-cn_topic_0000001929299926_p6833mcpsimp"></a><a name="zh-cn_topic_0000001929299926_p6833mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299926_p6835mcpsimp"><a name="zh-cn_topic_0000001929299926_p6835mcpsimp"></a><a name="zh-cn_topic_0000001929299926_p6835mcpsimp"></a>对端内存偏移</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929299926_row6836mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299926_p6838mcpsimp"><a name="zh-cn_topic_0000001929299926_p6838mcpsimp"></a><a name="zh-cn_topic_0000001929299926_p6838mcpsimp"></a>const void *src</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299926_p6840mcpsimp"><a name="zh-cn_topic_0000001929299926_p6840mcpsimp"></a><a name="zh-cn_topic_0000001929299926_p6840mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299926_p6842mcpsimp"><a name="zh-cn_topic_0000001929299926_p6842mcpsimp"></a><a name="zh-cn_topic_0000001929299926_p6842mcpsimp"></a>源地址</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929299926_row6843mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299926_p6845mcpsimp"><a name="zh-cn_topic_0000001929299926_p6845mcpsimp"></a><a name="zh-cn_topic_0000001929299926_p6845mcpsimp"></a>u64 len</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299926_p6847mcpsimp"><a name="zh-cn_topic_0000001929299926_p6847mcpsimp"></a><a name="zh-cn_topic_0000001929299926_p6847mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299926_p6849mcpsimp"><a name="zh-cn_topic_0000001929299926_p6849mcpsimp"></a><a name="zh-cn_topic_0000001929299926_p6849mcpsimp"></a>发送数据大小</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929299926_row6850mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299926_p6852mcpsimp"><a name="zh-cn_topic_0000001929299926_p6852mcpsimp"></a><a name="zh-cn_topic_0000001929299926_p6852mcpsimp"></a>Stream &amp;stream</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299926_p6854mcpsimp"><a name="zh-cn_topic_0000001929299926_p6854mcpsimp"></a><a name="zh-cn_topic_0000001929299926_p6854mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299926_p6856mcpsimp"><a name="zh-cn_topic_0000001929299926_p6856mcpsimp"></a><a name="zh-cn_topic_0000001929299926_p6856mcpsimp"></a>Stream对象</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929299926_row44061755112218"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299926_p6889mcpsimp"><a name="zh-cn_topic_0000001929299926_p6889mcpsimp"></a><a name="zh-cn_topic_0000001929299926_p6889mcpsimp"></a>std::vector&lt;TxMemoryInfo&gt;&amp; txMems</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299926_p6891mcpsimp"><a name="zh-cn_topic_0000001929299926_p6891mcpsimp"></a><a name="zh-cn_topic_0000001929299926_p6891mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299926_p6893mcpsimp"><a name="zh-cn_topic_0000001929299926_p6893mcpsimp"></a><a name="zh-cn_topic_0000001929299926_p6893mcpsimp"></a>发送内存信息</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929299926_row8456175882218"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299926_p6896mcpsimp"><a name="zh-cn_topic_0000001929299926_p6896mcpsimp"></a><a name="zh-cn_topic_0000001929299926_p6896mcpsimp"></a>Stream &amp;stream</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299926_p6898mcpsimp"><a name="zh-cn_topic_0000001929299926_p6898mcpsimp"></a><a name="zh-cn_topic_0000001929299926_p6898mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299926_p6900mcpsimp"><a name="zh-cn_topic_0000001929299926_p6900mcpsimp"></a><a name="zh-cn_topic_0000001929299926_p6900mcpsimp"></a>Stream对象</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001929299926_section6857mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

## 约束说明<a name="zh-cn_topic_0000001929299926_section6860mcpsimp"></a>

无

