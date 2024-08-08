# static Post 

## 函数原型<a name="zh-cn_topic_0000001929459178_section3356mcpsimp"></a>

```
static HcclResult Post(Stream& stream, HcclDispatcher dispatcherPtr, const std::shared_ptr<LocalNotify> &notify, s32 stage = INVALID_VALUE_STAGE)
```

## 函数功能<a name="zh-cn_topic_0000001929459178_section3359mcpsimp"></a>

Notify post任务。

## 参数说明<a name="zh-cn_topic_0000001929459178_section3362mcpsimp"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0000001929459178_table3364mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001929459178_row3371mcpsimp"><th class="cellrowborder" valign="top" width="28.71%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001929459178_p3373mcpsimp"><a name="zh-cn_topic_0000001929459178_p3373mcpsimp"></a><a name="zh-cn_topic_0000001929459178_p3373mcpsimp"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="13.86%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001929459178_p3375mcpsimp"><a name="zh-cn_topic_0000001929459178_p3375mcpsimp"></a><a name="zh-cn_topic_0000001929459178_p3375mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="57.43000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001929459178_p3377mcpsimp"><a name="zh-cn_topic_0000001929459178_p3377mcpsimp"></a><a name="zh-cn_topic_0000001929459178_p3377mcpsimp"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001929459178_row3379mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929459178_p3381mcpsimp"><a name="zh-cn_topic_0000001929459178_p3381mcpsimp"></a><a name="zh-cn_topic_0000001929459178_p3381mcpsimp"></a>Stream&amp; stream</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929459178_p3383mcpsimp"><a name="zh-cn_topic_0000001929459178_p3383mcpsimp"></a><a name="zh-cn_topic_0000001929459178_p3383mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929459178_p3385mcpsimp"><a name="zh-cn_topic_0000001929459178_p3385mcpsimp"></a><a name="zh-cn_topic_0000001929459178_p3385mcpsimp"></a>Stream对象</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929459178_row3386mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929459178_p3388mcpsimp"><a name="zh-cn_topic_0000001929459178_p3388mcpsimp"></a><a name="zh-cn_topic_0000001929459178_p3388mcpsimp"></a>HcclDispatcher dispatcher</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929459178_p3390mcpsimp"><a name="zh-cn_topic_0000001929459178_p3390mcpsimp"></a><a name="zh-cn_topic_0000001929459178_p3390mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929459178_p3392mcpsimp"><a name="zh-cn_topic_0000001929459178_p3392mcpsimp"></a><a name="zh-cn_topic_0000001929459178_p3392mcpsimp"></a>Dispatcher handle</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929459178_row3393mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929459178_p3395mcpsimp"><a name="zh-cn_topic_0000001929459178_p3395mcpsimp"></a><a name="zh-cn_topic_0000001929459178_p3395mcpsimp"></a>const std::shared_ptr&lt;LocalNotify&gt; &amp;notify</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929459178_p3397mcpsimp"><a name="zh-cn_topic_0000001929459178_p3397mcpsimp"></a><a name="zh-cn_topic_0000001929459178_p3397mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929459178_p3399mcpsimp"><a name="zh-cn_topic_0000001929459178_p3399mcpsimp"></a><a name="zh-cn_topic_0000001929459178_p3399mcpsimp"></a>Notify对象指针</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929459178_row3400mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929459178_p3402mcpsimp"><a name="zh-cn_topic_0000001929459178_p3402mcpsimp"></a><a name="zh-cn_topic_0000001929459178_p3402mcpsimp"></a>s32 stage</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929459178_p3404mcpsimp"><a name="zh-cn_topic_0000001929459178_p3404mcpsimp"></a><a name="zh-cn_topic_0000001929459178_p3404mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929459178_p3406mcpsimp"><a name="zh-cn_topic_0000001929459178_p3406mcpsimp"></a><a name="zh-cn_topic_0000001929459178_p3406mcpsimp"></a>算法stage</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001929459178_section3407mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

## 约束说明<a name="zh-cn_topic_0000001929459178_section3410mcpsimp"></a>

无

