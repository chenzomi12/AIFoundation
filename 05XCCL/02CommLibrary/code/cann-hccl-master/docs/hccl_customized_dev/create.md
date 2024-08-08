# create

## 函数原型<a name="zh-cn_topic_0000001960184589_section9899mcpsimp"></a>

static DeviceMem create\(void \*ptr, u64 size\)

## 函数功能<a name="zh-cn_topic_0000001960184589_section9902mcpsimp"></a>

用输入地址和大小构造DeviceMem对象，不会去申请、释放device内存。

## 参数说明<a name="zh-cn_topic_0000001960184589_section9905mcpsimp"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0000001960184589_table9907mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001960184589_row9914mcpsimp"><th class="cellrowborder" valign="top" width="28.71287128712871%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001960184589_p9916mcpsimp"><a name="zh-cn_topic_0000001960184589_p9916mcpsimp"></a><a name="zh-cn_topic_0000001960184589_p9916mcpsimp"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="13.861386138613863%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001960184589_p9918mcpsimp"><a name="zh-cn_topic_0000001960184589_p9918mcpsimp"></a><a name="zh-cn_topic_0000001960184589_p9918mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="57.42574257425742%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001960184589_p9920mcpsimp"><a name="zh-cn_topic_0000001960184589_p9920mcpsimp"></a><a name="zh-cn_topic_0000001960184589_p9920mcpsimp"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001960184589_row9922mcpsimp"><td class="cellrowborder" valign="top" width="28.71287128712871%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001960184589_p9924mcpsimp"><a name="zh-cn_topic_0000001960184589_p9924mcpsimp"></a><a name="zh-cn_topic_0000001960184589_p9924mcpsimp"></a>void *ptr</p>
</td>
<td class="cellrowborder" valign="top" width="13.861386138613863%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001960184589_p9926mcpsimp"><a name="zh-cn_topic_0000001960184589_p9926mcpsimp"></a><a name="zh-cn_topic_0000001960184589_p9926mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.42574257425742%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001960184589_p9928mcpsimp"><a name="zh-cn_topic_0000001960184589_p9928mcpsimp"></a><a name="zh-cn_topic_0000001960184589_p9928mcpsimp"></a>内存地址</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001960184589_row9929mcpsimp"><td class="cellrowborder" valign="top" width="28.71287128712871%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001960184589_p9931mcpsimp"><a name="zh-cn_topic_0000001960184589_p9931mcpsimp"></a><a name="zh-cn_topic_0000001960184589_p9931mcpsimp"></a>u64 size</p>
</td>
<td class="cellrowborder" valign="top" width="13.861386138613863%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001960184589_p9933mcpsimp"><a name="zh-cn_topic_0000001960184589_p9933mcpsimp"></a><a name="zh-cn_topic_0000001960184589_p9933mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.42574257425742%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001960184589_p9935mcpsimp"><a name="zh-cn_topic_0000001960184589_p9935mcpsimp"></a><a name="zh-cn_topic_0000001960184589_p9935mcpsimp"></a>内存大小</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001960184589_section9936mcpsimp"></a>

DeviceMem对象。

## 约束说明<a name="zh-cn_topic_0000001960184589_section9939mcpsimp"></a>

无

