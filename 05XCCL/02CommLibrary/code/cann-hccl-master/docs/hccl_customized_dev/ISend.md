# ISend 

## 函数原型<a name="zh-cn_topic_0000001929299774_section2383mcpsimp"></a>

```
HcclResult ISend(void *data, u64 size, u64& compSize)
```

## 函数功能<a name="zh-cn_topic_0000001929299774_section2386mcpsimp"></a>

非阻塞发送。

## 参数说明<a name="zh-cn_topic_0000001929299774_section2389mcpsimp"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0000001929299774_table2391mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001929299774_row2398mcpsimp"><th class="cellrowborder" valign="top" width="28.71%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001929299774_p2400mcpsimp"><a name="zh-cn_topic_0000001929299774_p2400mcpsimp"></a><a name="zh-cn_topic_0000001929299774_p2400mcpsimp"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="13.86%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001929299774_p2402mcpsimp"><a name="zh-cn_topic_0000001929299774_p2402mcpsimp"></a><a name="zh-cn_topic_0000001929299774_p2402mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="57.43000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001929299774_p2404mcpsimp"><a name="zh-cn_topic_0000001929299774_p2404mcpsimp"></a><a name="zh-cn_topic_0000001929299774_p2404mcpsimp"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001929299774_row2406mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299774_p2408mcpsimp"><a name="zh-cn_topic_0000001929299774_p2408mcpsimp"></a><a name="zh-cn_topic_0000001929299774_p2408mcpsimp"></a>void *data</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299774_p2410mcpsimp"><a name="zh-cn_topic_0000001929299774_p2410mcpsimp"></a><a name="zh-cn_topic_0000001929299774_p2410mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299774_p2412mcpsimp"><a name="zh-cn_topic_0000001929299774_p2412mcpsimp"></a><a name="zh-cn_topic_0000001929299774_p2412mcpsimp"></a>发送数据起始地址</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929299774_row2413mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299774_p2415mcpsimp"><a name="zh-cn_topic_0000001929299774_p2415mcpsimp"></a><a name="zh-cn_topic_0000001929299774_p2415mcpsimp"></a>u64 size</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299774_p2417mcpsimp"><a name="zh-cn_topic_0000001929299774_p2417mcpsimp"></a><a name="zh-cn_topic_0000001929299774_p2417mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299774_p2419mcpsimp"><a name="zh-cn_topic_0000001929299774_p2419mcpsimp"></a><a name="zh-cn_topic_0000001929299774_p2419mcpsimp"></a>发送数据大小</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929299774_row2420mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299774_p2422mcpsimp"><a name="zh-cn_topic_0000001929299774_p2422mcpsimp"></a><a name="zh-cn_topic_0000001929299774_p2422mcpsimp"></a>u64&amp; compSize</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299774_p2424mcpsimp"><a name="zh-cn_topic_0000001929299774_p2424mcpsimp"></a><a name="zh-cn_topic_0000001929299774_p2424mcpsimp"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299774_p2426mcpsimp"><a name="zh-cn_topic_0000001929299774_p2426mcpsimp"></a><a name="zh-cn_topic_0000001929299774_p2426mcpsimp"></a>实际发送数据大小</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001929299774_section2427mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

## 约束说明<a name="zh-cn_topic_0000001929299774_section2430mcpsimp"></a>

无

