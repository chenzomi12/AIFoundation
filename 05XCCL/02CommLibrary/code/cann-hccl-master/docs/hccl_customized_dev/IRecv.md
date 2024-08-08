# IRecv 

## 函数原型<a name="zh-cn_topic_0000001929459150_section2434mcpsimp"></a>

```
HcclResult IRecv(void *recvBuf, u32 recvBufLen, u64& compSize)
```

## 函数功能<a name="zh-cn_topic_0000001929459150_section2437mcpsimp"></a>

非阻塞接收。

## 参数说明<a name="zh-cn_topic_0000001929459150_section2440mcpsimp"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0000001929459150_table2442mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001929459150_row2449mcpsimp"><th class="cellrowborder" valign="top" width="28.71%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001929459150_p2451mcpsimp"><a name="zh-cn_topic_0000001929459150_p2451mcpsimp"></a><a name="zh-cn_topic_0000001929459150_p2451mcpsimp"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="13.86%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001929459150_p2453mcpsimp"><a name="zh-cn_topic_0000001929459150_p2453mcpsimp"></a><a name="zh-cn_topic_0000001929459150_p2453mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="57.43000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001929459150_p2455mcpsimp"><a name="zh-cn_topic_0000001929459150_p2455mcpsimp"></a><a name="zh-cn_topic_0000001929459150_p2455mcpsimp"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001929459150_row2457mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929459150_p2459mcpsimp"><a name="zh-cn_topic_0000001929459150_p2459mcpsimp"></a><a name="zh-cn_topic_0000001929459150_p2459mcpsimp"></a>void *recvBuf</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929459150_p2461mcpsimp"><a name="zh-cn_topic_0000001929459150_p2461mcpsimp"></a><a name="zh-cn_topic_0000001929459150_p2461mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929459150_p2463mcpsimp"><a name="zh-cn_topic_0000001929459150_p2463mcpsimp"></a><a name="zh-cn_topic_0000001929459150_p2463mcpsimp"></a>接收数据起始地址</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929459150_row2464mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929459150_p2466mcpsimp"><a name="zh-cn_topic_0000001929459150_p2466mcpsimp"></a><a name="zh-cn_topic_0000001929459150_p2466mcpsimp"></a>u32 recvBufLen</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929459150_p2468mcpsimp"><a name="zh-cn_topic_0000001929459150_p2468mcpsimp"></a><a name="zh-cn_topic_0000001929459150_p2468mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929459150_p2470mcpsimp"><a name="zh-cn_topic_0000001929459150_p2470mcpsimp"></a><a name="zh-cn_topic_0000001929459150_p2470mcpsimp"></a>接收数据buffer大小</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929459150_row2471mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929459150_p2473mcpsimp"><a name="zh-cn_topic_0000001929459150_p2473mcpsimp"></a><a name="zh-cn_topic_0000001929459150_p2473mcpsimp"></a>u64&amp; compSize</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929459150_p2475mcpsimp"><a name="zh-cn_topic_0000001929459150_p2475mcpsimp"></a><a name="zh-cn_topic_0000001929459150_p2475mcpsimp"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929459150_p2477mcpsimp"><a name="zh-cn_topic_0000001929459150_p2477mcpsimp"></a><a name="zh-cn_topic_0000001929459150_p2477mcpsimp"></a>实际接收数据量</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001929459150_section2478mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

## 约束说明<a name="zh-cn_topic_0000001929459150_section2481mcpsimp"></a>

无

