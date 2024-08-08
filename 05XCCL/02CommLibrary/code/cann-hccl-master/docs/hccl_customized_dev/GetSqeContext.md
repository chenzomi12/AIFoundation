# GetSqeContext 

## 函数原型<a name="zh-cn_topic_0000001963694625_section540mcpsimp"></a>

HcclResult GetSqeContext\(std::shared\_ptr<HcclSqeContext\> &sqeContext\)

## 函数功能<a name="zh-cn_topic_0000001963694625_section543mcpsimp"></a>

获取sqe context。

## 参数说明<a name="zh-cn_topic_0000001963694625_section546mcpsimp"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0000001963694625_table548mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001963694625_row555mcpsimp"><th class="cellrowborder" valign="top" width="28.71287128712871%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001963694625_p557mcpsimp"><a name="zh-cn_topic_0000001963694625_p557mcpsimp"></a><a name="zh-cn_topic_0000001963694625_p557mcpsimp"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="13.861386138613863%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001963694625_p559mcpsimp"><a name="zh-cn_topic_0000001963694625_p559mcpsimp"></a><a name="zh-cn_topic_0000001963694625_p559mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="57.42574257425742%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001963694625_p561mcpsimp"><a name="zh-cn_topic_0000001963694625_p561mcpsimp"></a><a name="zh-cn_topic_0000001963694625_p561mcpsimp"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001963694625_row563mcpsimp"><td class="cellrowborder" valign="top" width="28.71287128712871%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001963694625_p565mcpsimp"><a name="zh-cn_topic_0000001963694625_p565mcpsimp"></a><a name="zh-cn_topic_0000001963694625_p565mcpsimp"></a>std::shared_ptr&lt;HcclSqeContext&gt; &amp;sqeContext</p>
</td>
<td class="cellrowborder" valign="top" width="13.861386138613863%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001963694625_p567mcpsimp"><a name="zh-cn_topic_0000001963694625_p567mcpsimp"></a><a name="zh-cn_topic_0000001963694625_p567mcpsimp"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="57.42574257425742%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001963694625_p569mcpsimp"><a name="zh-cn_topic_0000001963694625_p569mcpsimp"></a><a name="zh-cn_topic_0000001963694625_p569mcpsimp"></a>Sqe context</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001963694625_section570mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

## 约束说明<a name="zh-cn_topic_0000001963694625_section573mcpsimp"></a>

无。

