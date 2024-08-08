# HcclResetQosCfg 

## 函数原型<a name="zh-cn_topic_0000001926464488_section263mcpsimp"></a>

```
HcclResult HcclResetQosCfg(HcclDispatcher dispatcherPtr)
```

## 功能说明<a name="zh-cn_topic_0000001926464488_section265mcpsimp"></a>

重置qos cfg。

## 参数说明<a name="zh-cn_topic_0000001926464488_section267mcpsimp"></a>

<a name="zh-cn_topic_0000001926464488_table268mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001926464488_row274mcpsimp"><th class="cellrowborder" valign="top" width="46%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001926464488_p276mcpsimp"><a name="zh-cn_topic_0000001926464488_p276mcpsimp"></a><a name="zh-cn_topic_0000001926464488_p276mcpsimp"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="22%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001926464488_p278mcpsimp"><a name="zh-cn_topic_0000001926464488_p278mcpsimp"></a><a name="zh-cn_topic_0000001926464488_p278mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="32%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001926464488_p280mcpsimp"><a name="zh-cn_topic_0000001926464488_p280mcpsimp"></a><a name="zh-cn_topic_0000001926464488_p280mcpsimp"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001926464488_row282mcpsimp"><td class="cellrowborder" valign="top" width="46%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001926464488_p284mcpsimp"><a name="zh-cn_topic_0000001926464488_p284mcpsimp"></a><a name="zh-cn_topic_0000001926464488_p284mcpsimp"></a>HcclDispatcher dispatcher</p>
</td>
<td class="cellrowborder" valign="top" width="22%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001926464488_p286mcpsimp"><a name="zh-cn_topic_0000001926464488_p286mcpsimp"></a><a name="zh-cn_topic_0000001926464488_p286mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="32%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001926464488_p288mcpsimp"><a name="zh-cn_topic_0000001926464488_p288mcpsimp"></a><a name="zh-cn_topic_0000001926464488_p288mcpsimp"></a>dispatcher handle</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001926464488_section289mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS；其他失败。

