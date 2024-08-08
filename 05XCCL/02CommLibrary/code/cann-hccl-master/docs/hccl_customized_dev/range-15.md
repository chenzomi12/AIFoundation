# range 

## 函数原型<a name="zh-cn_topic_0000001933265328_section10156mcpsimp"></a>

HostMem range\(u64 offset, u64 size\) const

## 函数功能<a name="zh-cn_topic_0000001933265328_section10159mcpsimp"></a>

在当前mem实例中截取一段形成新的Mem实例。

## 参数说明<a name="zh-cn_topic_0000001933265328_section10162mcpsimp"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0000001933265328_table10164mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001933265328_row10171mcpsimp"><th class="cellrowborder" valign="top" width="28.71287128712871%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001933265328_p10173mcpsimp"><a name="zh-cn_topic_0000001933265328_p10173mcpsimp"></a><a name="zh-cn_topic_0000001933265328_p10173mcpsimp"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="13.861386138613863%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001933265328_p10175mcpsimp"><a name="zh-cn_topic_0000001933265328_p10175mcpsimp"></a><a name="zh-cn_topic_0000001933265328_p10175mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="57.42574257425742%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001933265328_p10177mcpsimp"><a name="zh-cn_topic_0000001933265328_p10177mcpsimp"></a><a name="zh-cn_topic_0000001933265328_p10177mcpsimp"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001933265328_row10179mcpsimp"><td class="cellrowborder" valign="top" width="28.71287128712871%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001933265328_p10181mcpsimp"><a name="zh-cn_topic_0000001933265328_p10181mcpsimp"></a><a name="zh-cn_topic_0000001933265328_p10181mcpsimp"></a>u64 offset</p>
</td>
<td class="cellrowborder" valign="top" width="13.861386138613863%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001933265328_p10183mcpsimp"><a name="zh-cn_topic_0000001933265328_p10183mcpsimp"></a><a name="zh-cn_topic_0000001933265328_p10183mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.42574257425742%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001933265328_p10185mcpsimp"><a name="zh-cn_topic_0000001933265328_p10185mcpsimp"></a><a name="zh-cn_topic_0000001933265328_p10185mcpsimp"></a>偏移大小</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001933265328_row10186mcpsimp"><td class="cellrowborder" valign="top" width="28.71287128712871%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001933265328_p10188mcpsimp"><a name="zh-cn_topic_0000001933265328_p10188mcpsimp"></a><a name="zh-cn_topic_0000001933265328_p10188mcpsimp"></a>u64 size</p>
</td>
<td class="cellrowborder" valign="top" width="13.861386138613863%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001933265328_p10190mcpsimp"><a name="zh-cn_topic_0000001933265328_p10190mcpsimp"></a><a name="zh-cn_topic_0000001933265328_p10190mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.42574257425742%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001933265328_p10192mcpsimp"><a name="zh-cn_topic_0000001933265328_p10192mcpsimp"></a><a name="zh-cn_topic_0000001933265328_p10192mcpsimp"></a>新实例大小</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001933265328_section10193mcpsimp"></a>

Host Mem对象。

## 约束说明<a name="zh-cn_topic_0000001933265328_section10196mcpsimp"></a>

无

