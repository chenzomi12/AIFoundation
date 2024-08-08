# Recv 

## 函数原型<a name="zh-cn_topic_0000001929459146_section2265mcpsimp"></a>

```
HcclResult Recv(void *recvBuf, u32 recvBufLen)
HcclResult Recv(std::string &recvMsg)
```

## 函数功能<a name="zh-cn_topic_0000001929459146_section2268mcpsimp"></a>

Socket recv。

## 参数说明<a name="zh-cn_topic_0000001929459146_section2271mcpsimp"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0000001929459146_table2273mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001929459146_row2280mcpsimp"><th class="cellrowborder" valign="top" width="28.71%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001929459146_p2282mcpsimp"><a name="zh-cn_topic_0000001929459146_p2282mcpsimp"></a><a name="zh-cn_topic_0000001929459146_p2282mcpsimp"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="13.86%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001929459146_p2284mcpsimp"><a name="zh-cn_topic_0000001929459146_p2284mcpsimp"></a><a name="zh-cn_topic_0000001929459146_p2284mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="57.43000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001929459146_p2286mcpsimp"><a name="zh-cn_topic_0000001929459146_p2286mcpsimp"></a><a name="zh-cn_topic_0000001929459146_p2286mcpsimp"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001929459146_row2288mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929459146_p2290mcpsimp"><a name="zh-cn_topic_0000001929459146_p2290mcpsimp"></a><a name="zh-cn_topic_0000001929459146_p2290mcpsimp"></a>void *recvBuf</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929459146_p2292mcpsimp"><a name="zh-cn_topic_0000001929459146_p2292mcpsimp"></a><a name="zh-cn_topic_0000001929459146_p2292mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929459146_p2294mcpsimp"><a name="zh-cn_topic_0000001929459146_p2294mcpsimp"></a><a name="zh-cn_topic_0000001929459146_p2294mcpsimp"></a>接收数据起始地址</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929459146_row2295mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929459146_p2297mcpsimp"><a name="zh-cn_topic_0000001929459146_p2297mcpsimp"></a><a name="zh-cn_topic_0000001929459146_p2297mcpsimp"></a>u32 recvBufLen</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929459146_p2299mcpsimp"><a name="zh-cn_topic_0000001929459146_p2299mcpsimp"></a><a name="zh-cn_topic_0000001929459146_p2299mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929459146_p2301mcpsimp"><a name="zh-cn_topic_0000001929459146_p2301mcpsimp"></a><a name="zh-cn_topic_0000001929459146_p2301mcpsimp"></a>接收数据大小</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929459146_row93951020143216"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929459146_p2371mcpsimp"><a name="zh-cn_topic_0000001929459146_p2371mcpsimp"></a><a name="zh-cn_topic_0000001929459146_p2371mcpsimp"></a>std::string &amp;recvMsg</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929459146_p2373mcpsimp"><a name="zh-cn_topic_0000001929459146_p2373mcpsimp"></a><a name="zh-cn_topic_0000001929459146_p2373mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929459146_p2375mcpsimp"><a name="zh-cn_topic_0000001929459146_p2375mcpsimp"></a><a name="zh-cn_topic_0000001929459146_p2375mcpsimp"></a>接收数据</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001929459146_section2302mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

## 约束说明<a name="zh-cn_topic_0000001929459146_section2305mcpsimp"></a>

无

