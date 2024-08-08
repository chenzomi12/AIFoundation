# range 

## 函数原型<a name="zh-cn_topic_0000001960344405_section9975mcpsimp"></a>

DeviceMem range\(u64 offset, u64 size\) const

## 函数功能<a name="zh-cn_topic_0000001960344405_section9978mcpsimp"></a>

在当前mem实例中截取一段形成新的Mem实例。

## 参数说明<a name="zh-cn_topic_0000001960344405_section9981mcpsimp"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0000001960344405_table9983mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001960344405_row9990mcpsimp"><th class="cellrowborder" valign="top" width="28.71287128712871%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001960344405_p9992mcpsimp"><a name="zh-cn_topic_0000001960344405_p9992mcpsimp"></a><a name="zh-cn_topic_0000001960344405_p9992mcpsimp"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="13.861386138613863%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001960344405_p9994mcpsimp"><a name="zh-cn_topic_0000001960344405_p9994mcpsimp"></a><a name="zh-cn_topic_0000001960344405_p9994mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="57.42574257425742%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001960344405_p9996mcpsimp"><a name="zh-cn_topic_0000001960344405_p9996mcpsimp"></a><a name="zh-cn_topic_0000001960344405_p9996mcpsimp"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001960344405_row9998mcpsimp"><td class="cellrowborder" valign="top" width="28.71287128712871%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001960344405_p10000mcpsimp"><a name="zh-cn_topic_0000001960344405_p10000mcpsimp"></a><a name="zh-cn_topic_0000001960344405_p10000mcpsimp"></a>u64 offset</p>
</td>
<td class="cellrowborder" valign="top" width="13.861386138613863%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001960344405_p10002mcpsimp"><a name="zh-cn_topic_0000001960344405_p10002mcpsimp"></a><a name="zh-cn_topic_0000001960344405_p10002mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.42574257425742%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001960344405_p10004mcpsimp"><a name="zh-cn_topic_0000001960344405_p10004mcpsimp"></a><a name="zh-cn_topic_0000001960344405_p10004mcpsimp"></a>偏移大小</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001960344405_row10005mcpsimp"><td class="cellrowborder" valign="top" width="28.71287128712871%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001960344405_p10007mcpsimp"><a name="zh-cn_topic_0000001960344405_p10007mcpsimp"></a><a name="zh-cn_topic_0000001960344405_p10007mcpsimp"></a>u64 size</p>
</td>
<td class="cellrowborder" valign="top" width="13.861386138613863%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001960344405_p10009mcpsimp"><a name="zh-cn_topic_0000001960344405_p10009mcpsimp"></a><a name="zh-cn_topic_0000001960344405_p10009mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.42574257425742%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001960344405_p10011mcpsimp"><a name="zh-cn_topic_0000001960344405_p10011mcpsimp"></a><a name="zh-cn_topic_0000001960344405_p10011mcpsimp"></a>新实例大小</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001960344405_section10012mcpsimp"></a>

Device Mem对象。

## 约束说明<a name="zh-cn_topic_0000001960344405_section10015mcpsimp"></a>

无

