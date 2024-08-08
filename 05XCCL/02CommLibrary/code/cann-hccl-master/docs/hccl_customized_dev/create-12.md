# create

## 函数原型<a name="zh-cn_topic_0000001960344441_section10080mcpsimp"></a>

static HostMem create\(void \*ptr, u64 size\)

## 函数功能<a name="zh-cn_topic_0000001960344441_section10083mcpsimp"></a>

用输入地址和大小构造HostMem对象，不会去申请、释放host内存。

## 参数说明<a name="zh-cn_topic_0000001960344441_section10086mcpsimp"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0000001960344441_table10088mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001960344441_row10095mcpsimp"><th class="cellrowborder" valign="top" width="28.71287128712871%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001960344441_p10097mcpsimp"><a name="zh-cn_topic_0000001960344441_p10097mcpsimp"></a><a name="zh-cn_topic_0000001960344441_p10097mcpsimp"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="13.861386138613863%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001960344441_p10099mcpsimp"><a name="zh-cn_topic_0000001960344441_p10099mcpsimp"></a><a name="zh-cn_topic_0000001960344441_p10099mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="57.42574257425742%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001960344441_p10101mcpsimp"><a name="zh-cn_topic_0000001960344441_p10101mcpsimp"></a><a name="zh-cn_topic_0000001960344441_p10101mcpsimp"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001960344441_row10103mcpsimp"><td class="cellrowborder" valign="top" width="28.71287128712871%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001960344441_p10105mcpsimp"><a name="zh-cn_topic_0000001960344441_p10105mcpsimp"></a><a name="zh-cn_topic_0000001960344441_p10105mcpsimp"></a>void *ptr</p>
</td>
<td class="cellrowborder" valign="top" width="13.861386138613863%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001960344441_p10107mcpsimp"><a name="zh-cn_topic_0000001960344441_p10107mcpsimp"></a><a name="zh-cn_topic_0000001960344441_p10107mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.42574257425742%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001960344441_p10109mcpsimp"><a name="zh-cn_topic_0000001960344441_p10109mcpsimp"></a><a name="zh-cn_topic_0000001960344441_p10109mcpsimp"></a>内存地址</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001960344441_row10110mcpsimp"><td class="cellrowborder" valign="top" width="28.71287128712871%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001960344441_p10112mcpsimp"><a name="zh-cn_topic_0000001960344441_p10112mcpsimp"></a><a name="zh-cn_topic_0000001960344441_p10112mcpsimp"></a>u64 size</p>
</td>
<td class="cellrowborder" valign="top" width="13.861386138613863%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001960344441_p10114mcpsimp"><a name="zh-cn_topic_0000001960344441_p10114mcpsimp"></a><a name="zh-cn_topic_0000001960344441_p10114mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.42574257425742%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001960344441_p10116mcpsimp"><a name="zh-cn_topic_0000001960344441_p10116mcpsimp"></a><a name="zh-cn_topic_0000001960344441_p10116mcpsimp"></a>内存大小</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001960344441_section10117mcpsimp"></a>

Host Mem对象。

## 约束说明<a name="zh-cn_topic_0000001960344441_section10120mcpsimp"></a>

无

