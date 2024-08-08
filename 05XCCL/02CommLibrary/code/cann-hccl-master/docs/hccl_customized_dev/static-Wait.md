# static Wait 

## 函数原型<a name="zh-cn_topic_0000001929299802_section3291mcpsimp"></a>

```
static HcclResult Wait(Stream& stream, HcclDispatcher dispatcherPtr, const std::shared_ptr<LocalNotify> &notify, s32 stage = INVALID_VALUE_STAGE, u32 timeOut = NOTIFY_DEFAULT_WAIT_TIME)
```

## 函数功能<a name="zh-cn_topic_0000001929299802_section3294mcpsimp"></a>

Notify wait任务。

## 参数说明<a name="zh-cn_topic_0000001929299802_section3297mcpsimp"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0000001929299802_table3299mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001929299802_row3306mcpsimp"><th class="cellrowborder" valign="top" width="28.71%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001929299802_p3308mcpsimp"><a name="zh-cn_topic_0000001929299802_p3308mcpsimp"></a><a name="zh-cn_topic_0000001929299802_p3308mcpsimp"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="13.86%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001929299802_p3310mcpsimp"><a name="zh-cn_topic_0000001929299802_p3310mcpsimp"></a><a name="zh-cn_topic_0000001929299802_p3310mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="57.43000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001929299802_p3312mcpsimp"><a name="zh-cn_topic_0000001929299802_p3312mcpsimp"></a><a name="zh-cn_topic_0000001929299802_p3312mcpsimp"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001929299802_row3314mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299802_p3316mcpsimp"><a name="zh-cn_topic_0000001929299802_p3316mcpsimp"></a><a name="zh-cn_topic_0000001929299802_p3316mcpsimp"></a>Stream&amp; stream</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299802_p3318mcpsimp"><a name="zh-cn_topic_0000001929299802_p3318mcpsimp"></a><a name="zh-cn_topic_0000001929299802_p3318mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299802_p3320mcpsimp"><a name="zh-cn_topic_0000001929299802_p3320mcpsimp"></a><a name="zh-cn_topic_0000001929299802_p3320mcpsimp"></a>Stream对象</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929299802_row3321mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299802_p3323mcpsimp"><a name="zh-cn_topic_0000001929299802_p3323mcpsimp"></a><a name="zh-cn_topic_0000001929299802_p3323mcpsimp"></a>HcclDispatcher dispatcher</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299802_p3325mcpsimp"><a name="zh-cn_topic_0000001929299802_p3325mcpsimp"></a><a name="zh-cn_topic_0000001929299802_p3325mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299802_p3327mcpsimp"><a name="zh-cn_topic_0000001929299802_p3327mcpsimp"></a><a name="zh-cn_topic_0000001929299802_p3327mcpsimp"></a>Dispatcher handle</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929299802_row3328mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299802_p3330mcpsimp"><a name="zh-cn_topic_0000001929299802_p3330mcpsimp"></a><a name="zh-cn_topic_0000001929299802_p3330mcpsimp"></a>const std::shared_ptr&lt;LocalNotify&gt; &amp;notify</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299802_p3332mcpsimp"><a name="zh-cn_topic_0000001929299802_p3332mcpsimp"></a><a name="zh-cn_topic_0000001929299802_p3332mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299802_p3334mcpsimp"><a name="zh-cn_topic_0000001929299802_p3334mcpsimp"></a><a name="zh-cn_topic_0000001929299802_p3334mcpsimp"></a>Notify对象指针</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929299802_row3335mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299802_p3337mcpsimp"><a name="zh-cn_topic_0000001929299802_p3337mcpsimp"></a><a name="zh-cn_topic_0000001929299802_p3337mcpsimp"></a>s32 stage</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299802_p3339mcpsimp"><a name="zh-cn_topic_0000001929299802_p3339mcpsimp"></a><a name="zh-cn_topic_0000001929299802_p3339mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299802_p3341mcpsimp"><a name="zh-cn_topic_0000001929299802_p3341mcpsimp"></a><a name="zh-cn_topic_0000001929299802_p3341mcpsimp"></a>算法stage</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929299802_row3342mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299802_p3344mcpsimp"><a name="zh-cn_topic_0000001929299802_p3344mcpsimp"></a><a name="zh-cn_topic_0000001929299802_p3344mcpsimp"></a>u32 timeOut</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299802_p3346mcpsimp"><a name="zh-cn_topic_0000001929299802_p3346mcpsimp"></a><a name="zh-cn_topic_0000001929299802_p3346mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299802_p3348mcpsimp"><a name="zh-cn_topic_0000001929299802_p3348mcpsimp"></a><a name="zh-cn_topic_0000001929299802_p3348mcpsimp"></a>Notify超时时间</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001929299802_section3349mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

## 约束说明<a name="zh-cn_topic_0000001929299802_section3352mcpsimp"></a>

无

