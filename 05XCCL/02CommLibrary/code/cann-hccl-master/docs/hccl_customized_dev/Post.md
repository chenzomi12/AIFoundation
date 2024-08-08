# Post 

## 函数原型<a name="zh-cn_topic_0000001929459170_section3081mcpsimp"></a>

```
HcclResult Post(Stream& stream, HcclDispatcher dispatcher, s32 stage = INVALID_VALUE_STAGE)
```

## 函数功能<a name="zh-cn_topic_0000001929459170_section3084mcpsimp"></a>

Notify post任务。

## 参数说明<a name="zh-cn_topic_0000001929459170_section3087mcpsimp"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0000001929459170_table3089mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001929459170_row3096mcpsimp"><th class="cellrowborder" valign="top" width="28.71%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001929459170_p3098mcpsimp"><a name="zh-cn_topic_0000001929459170_p3098mcpsimp"></a><a name="zh-cn_topic_0000001929459170_p3098mcpsimp"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="13.86%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001929459170_p3100mcpsimp"><a name="zh-cn_topic_0000001929459170_p3100mcpsimp"></a><a name="zh-cn_topic_0000001929459170_p3100mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="57.43000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001929459170_p3102mcpsimp"><a name="zh-cn_topic_0000001929459170_p3102mcpsimp"></a><a name="zh-cn_topic_0000001929459170_p3102mcpsimp"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001929459170_row3104mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929459170_p3106mcpsimp"><a name="zh-cn_topic_0000001929459170_p3106mcpsimp"></a><a name="zh-cn_topic_0000001929459170_p3106mcpsimp"></a>Stream&amp; stream</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929459170_p3108mcpsimp"><a name="zh-cn_topic_0000001929459170_p3108mcpsimp"></a><a name="zh-cn_topic_0000001929459170_p3108mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929459170_p3110mcpsimp"><a name="zh-cn_topic_0000001929459170_p3110mcpsimp"></a><a name="zh-cn_topic_0000001929459170_p3110mcpsimp"></a>Stream对象</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929459170_row3111mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929459170_p3113mcpsimp"><a name="zh-cn_topic_0000001929459170_p3113mcpsimp"></a><a name="zh-cn_topic_0000001929459170_p3113mcpsimp"></a>HcclDispatcher dispatcher</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929459170_p3115mcpsimp"><a name="zh-cn_topic_0000001929459170_p3115mcpsimp"></a><a name="zh-cn_topic_0000001929459170_p3115mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929459170_p3117mcpsimp"><a name="zh-cn_topic_0000001929459170_p3117mcpsimp"></a><a name="zh-cn_topic_0000001929459170_p3117mcpsimp"></a>Dispatcher handle</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929459170_row3118mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929459170_p3120mcpsimp"><a name="zh-cn_topic_0000001929459170_p3120mcpsimp"></a><a name="zh-cn_topic_0000001929459170_p3120mcpsimp"></a>s32 stage</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929459170_p3122mcpsimp"><a name="zh-cn_topic_0000001929459170_p3122mcpsimp"></a><a name="zh-cn_topic_0000001929459170_p3122mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929459170_p3124mcpsimp"><a name="zh-cn_topic_0000001929459170_p3124mcpsimp"></a><a name="zh-cn_topic_0000001929459170_p3124mcpsimp"></a>算法stage</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001929459170_section3125mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

## 约束说明<a name="zh-cn_topic_0000001929459170_section3128mcpsimp"></a>

无

