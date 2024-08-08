# Accept

## 函数原型<a name="zh-cn_topic_0000001956618417_section2177mcpsimp"></a>

```
HcclResult Accept(const std::string &tag, std::shared_ptr<HcclSocket> &socket)
```

## 函数功能<a name="zh-cn_topic_0000001956618417_section2180mcpsimp"></a>

发起建链请求。

## 参数说明<a name="zh-cn_topic_0000001956618417_section2183mcpsimp"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0000001956618417_table2185mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001956618417_row2192mcpsimp"><th class="cellrowborder" valign="top" width="28.71%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001956618417_p2194mcpsimp"><a name="zh-cn_topic_0000001956618417_p2194mcpsimp"></a><a name="zh-cn_topic_0000001956618417_p2194mcpsimp"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="13.86%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001956618417_p2196mcpsimp"><a name="zh-cn_topic_0000001956618417_p2196mcpsimp"></a><a name="zh-cn_topic_0000001956618417_p2196mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="57.43000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001956618417_p2198mcpsimp"><a name="zh-cn_topic_0000001956618417_p2198mcpsimp"></a><a name="zh-cn_topic_0000001956618417_p2198mcpsimp"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001956618417_row2200mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956618417_p2202mcpsimp"><a name="zh-cn_topic_0000001956618417_p2202mcpsimp"></a><a name="zh-cn_topic_0000001956618417_p2202mcpsimp"></a>const std::string &amp;tag</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956618417_p2204mcpsimp"><a name="zh-cn_topic_0000001956618417_p2204mcpsimp"></a><a name="zh-cn_topic_0000001956618417_p2204mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956618417_p2206mcpsimp"><a name="zh-cn_topic_0000001956618417_p2206mcpsimp"></a><a name="zh-cn_topic_0000001956618417_p2206mcpsimp"></a>Tag标识</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001956618417_row2207mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956618417_p2209mcpsimp"><a name="zh-cn_topic_0000001956618417_p2209mcpsimp"></a><a name="zh-cn_topic_0000001956618417_p2209mcpsimp"></a>std::shared_ptr&lt;HcclSocket&gt; &amp;socket</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956618417_p2211mcpsimp"><a name="zh-cn_topic_0000001956618417_p2211mcpsimp"></a><a name="zh-cn_topic_0000001956618417_p2211mcpsimp"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956618417_p2213mcpsimp"><a name="zh-cn_topic_0000001956618417_p2213mcpsimp"></a><a name="zh-cn_topic_0000001956618417_p2213mcpsimp"></a>建链完成的socket对象</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001956618417_section2214mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

## 约束说明<a name="zh-cn_topic_0000001956618417_section2217mcpsimp"></a>

无

