# Wait 

## 函数原型<a name="zh-cn_topic_0000001929299794_section3023mcpsimp"></a>

```
HcclResult Wait(Stream& stream, HcclDispatcher dispatcher, s32 stage = INVALID_VALUE_STAGE, u32 timeOut = NOTIFY_DEFAULT_WAIT_TIME)
```

## 函数功能<a name="zh-cn_topic_0000001929299794_section3026mcpsimp"></a>

Notify wait任务。

## 参数说明<a name="zh-cn_topic_0000001929299794_section3029mcpsimp"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0000001929299794_table3031mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001929299794_row3038mcpsimp"><th class="cellrowborder" valign="top" width="28.71%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001929299794_p3040mcpsimp"><a name="zh-cn_topic_0000001929299794_p3040mcpsimp"></a><a name="zh-cn_topic_0000001929299794_p3040mcpsimp"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="13.86%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001929299794_p3042mcpsimp"><a name="zh-cn_topic_0000001929299794_p3042mcpsimp"></a><a name="zh-cn_topic_0000001929299794_p3042mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="57.43000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001929299794_p3044mcpsimp"><a name="zh-cn_topic_0000001929299794_p3044mcpsimp"></a><a name="zh-cn_topic_0000001929299794_p3044mcpsimp"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001929299794_row3046mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299794_p3048mcpsimp"><a name="zh-cn_topic_0000001929299794_p3048mcpsimp"></a><a name="zh-cn_topic_0000001929299794_p3048mcpsimp"></a>Stream&amp; stream</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299794_p3050mcpsimp"><a name="zh-cn_topic_0000001929299794_p3050mcpsimp"></a><a name="zh-cn_topic_0000001929299794_p3050mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299794_p3052mcpsimp"><a name="zh-cn_topic_0000001929299794_p3052mcpsimp"></a><a name="zh-cn_topic_0000001929299794_p3052mcpsimp"></a>Stream对象</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929299794_row3053mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299794_p3055mcpsimp"><a name="zh-cn_topic_0000001929299794_p3055mcpsimp"></a><a name="zh-cn_topic_0000001929299794_p3055mcpsimp"></a>HcclDispatcher dispatcher</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299794_p3057mcpsimp"><a name="zh-cn_topic_0000001929299794_p3057mcpsimp"></a><a name="zh-cn_topic_0000001929299794_p3057mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299794_p3059mcpsimp"><a name="zh-cn_topic_0000001929299794_p3059mcpsimp"></a><a name="zh-cn_topic_0000001929299794_p3059mcpsimp"></a>Dispatcher handle</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929299794_row3060mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299794_p3062mcpsimp"><a name="zh-cn_topic_0000001929299794_p3062mcpsimp"></a><a name="zh-cn_topic_0000001929299794_p3062mcpsimp"></a>s32 stage</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299794_p3064mcpsimp"><a name="zh-cn_topic_0000001929299794_p3064mcpsimp"></a><a name="zh-cn_topic_0000001929299794_p3064mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299794_p3066mcpsimp"><a name="zh-cn_topic_0000001929299794_p3066mcpsimp"></a><a name="zh-cn_topic_0000001929299794_p3066mcpsimp"></a>算法stage</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929299794_row3067mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299794_p3069mcpsimp"><a name="zh-cn_topic_0000001929299794_p3069mcpsimp"></a><a name="zh-cn_topic_0000001929299794_p3069mcpsimp"></a>u32 timeOut</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299794_p3071mcpsimp"><a name="zh-cn_topic_0000001929299794_p3071mcpsimp"></a><a name="zh-cn_topic_0000001929299794_p3071mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299794_p3073mcpsimp"><a name="zh-cn_topic_0000001929299794_p3073mcpsimp"></a><a name="zh-cn_topic_0000001929299794_p3073mcpsimp"></a>Notify超时时间</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001929299794_section3074mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

## 约束说明<a name="zh-cn_topic_0000001929299794_section3077mcpsimp"></a>

无

