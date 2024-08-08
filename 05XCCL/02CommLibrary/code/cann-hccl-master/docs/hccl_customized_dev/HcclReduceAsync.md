# HcclReduceAsync 

## 函数原型<a name="zh-cn_topic_0000001953823221_section889mcpsimp"></a>

```
HcclResult HcclReduceAsync(HcclDispatcher dispatcherPtr, void *src, uint64_t count, const HcclDataType datatype, const HcclReduceOp reduceOp, hccl::Stream &stream, void *dst, const u32 remoteUserRank, const hccl::LinkType linkType, const u64 reduceAttr)
```

## 功能说明<a name="zh-cn_topic_0000001953823221_section891mcpsimp"></a>

异步reduce。

## 参数说明<a name="zh-cn_topic_0000001953823221_section893mcpsimp"></a>

<a name="zh-cn_topic_0000001953823221_table894mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001953823221_row900mcpsimp"><th class="cellrowborder" valign="top" width="47%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001953823221_p902mcpsimp"><a name="zh-cn_topic_0000001953823221_p902mcpsimp"></a><a name="zh-cn_topic_0000001953823221_p902mcpsimp"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="22%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001953823221_p904mcpsimp"><a name="zh-cn_topic_0000001953823221_p904mcpsimp"></a><a name="zh-cn_topic_0000001953823221_p904mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="31%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001953823221_p906mcpsimp"><a name="zh-cn_topic_0000001953823221_p906mcpsimp"></a><a name="zh-cn_topic_0000001953823221_p906mcpsimp"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001953823221_row908mcpsimp"><td class="cellrowborder" valign="top" width="47%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001953823221_p910mcpsimp"><a name="zh-cn_topic_0000001953823221_p910mcpsimp"></a><a name="zh-cn_topic_0000001953823221_p910mcpsimp"></a>HcclDispatcher dispatcher</p>
</td>
<td class="cellrowborder" valign="top" width="22%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001953823221_p912mcpsimp"><a name="zh-cn_topic_0000001953823221_p912mcpsimp"></a><a name="zh-cn_topic_0000001953823221_p912mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="31%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001953823221_p914mcpsimp"><a name="zh-cn_topic_0000001953823221_p914mcpsimp"></a><a name="zh-cn_topic_0000001953823221_p914mcpsimp"></a>dispatcher handle</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001953823221_row915mcpsimp"><td class="cellrowborder" valign="top" width="47%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001953823221_p917mcpsimp"><a name="zh-cn_topic_0000001953823221_p917mcpsimp"></a><a name="zh-cn_topic_0000001953823221_p917mcpsimp"></a>void *src</p>
</td>
<td class="cellrowborder" valign="top" width="22%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001953823221_p919mcpsimp"><a name="zh-cn_topic_0000001953823221_p919mcpsimp"></a><a name="zh-cn_topic_0000001953823221_p919mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="31%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001953823221_p921mcpsimp"><a name="zh-cn_topic_0000001953823221_p921mcpsimp"></a><a name="zh-cn_topic_0000001953823221_p921mcpsimp"></a>src内存地址</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001953823221_row922mcpsimp"><td class="cellrowborder" valign="top" width="47%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001953823221_p924mcpsimp"><a name="zh-cn_topic_0000001953823221_p924mcpsimp"></a><a name="zh-cn_topic_0000001953823221_p924mcpsimp"></a>uint64_t count</p>
</td>
<td class="cellrowborder" valign="top" width="22%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001953823221_p926mcpsimp"><a name="zh-cn_topic_0000001953823221_p926mcpsimp"></a><a name="zh-cn_topic_0000001953823221_p926mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="31%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001953823221_p928mcpsimp"><a name="zh-cn_topic_0000001953823221_p928mcpsimp"></a><a name="zh-cn_topic_0000001953823221_p928mcpsimp"></a>reduce mem大小</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001953823221_row929mcpsimp"><td class="cellrowborder" valign="top" width="47%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001953823221_p931mcpsimp"><a name="zh-cn_topic_0000001953823221_p931mcpsimp"></a><a name="zh-cn_topic_0000001953823221_p931mcpsimp"></a>HcclDataType datatype</p>
</td>
<td class="cellrowborder" valign="top" width="22%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001953823221_p933mcpsimp"><a name="zh-cn_topic_0000001953823221_p933mcpsimp"></a><a name="zh-cn_topic_0000001953823221_p933mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="31%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001953823221_p935mcpsimp"><a name="zh-cn_topic_0000001953823221_p935mcpsimp"></a><a name="zh-cn_topic_0000001953823221_p935mcpsimp"></a>数据类型</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001953823221_row936mcpsimp"><td class="cellrowborder" valign="top" width="47%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001953823221_p938mcpsimp"><a name="zh-cn_topic_0000001953823221_p938mcpsimp"></a><a name="zh-cn_topic_0000001953823221_p938mcpsimp"></a>HcclReduceOp reduceOp</p>
</td>
<td class="cellrowborder" valign="top" width="22%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001953823221_p940mcpsimp"><a name="zh-cn_topic_0000001953823221_p940mcpsimp"></a><a name="zh-cn_topic_0000001953823221_p940mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="31%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001953823221_p942mcpsimp"><a name="zh-cn_topic_0000001953823221_p942mcpsimp"></a><a name="zh-cn_topic_0000001953823221_p942mcpsimp"></a>reduce op类型</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001953823221_row943mcpsimp"><td class="cellrowborder" valign="top" width="47%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001953823221_p945mcpsimp"><a name="zh-cn_topic_0000001953823221_p945mcpsimp"></a><a name="zh-cn_topic_0000001953823221_p945mcpsimp"></a>hccl::Stream &amp;stream</p>
</td>
<td class="cellrowborder" valign="top" width="22%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001953823221_p947mcpsimp"><a name="zh-cn_topic_0000001953823221_p947mcpsimp"></a><a name="zh-cn_topic_0000001953823221_p947mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="31%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001953823221_p949mcpsimp"><a name="zh-cn_topic_0000001953823221_p949mcpsimp"></a><a name="zh-cn_topic_0000001953823221_p949mcpsimp"></a>stream对象</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001953823221_row950mcpsimp"><td class="cellrowborder" valign="top" width="47%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001953823221_p952mcpsimp"><a name="zh-cn_topic_0000001953823221_p952mcpsimp"></a><a name="zh-cn_topic_0000001953823221_p952mcpsimp"></a>void *dst</p>
</td>
<td class="cellrowborder" valign="top" width="22%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001953823221_p954mcpsimp"><a name="zh-cn_topic_0000001953823221_p954mcpsimp"></a><a name="zh-cn_topic_0000001953823221_p954mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="31%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001953823221_p956mcpsimp"><a name="zh-cn_topic_0000001953823221_p956mcpsimp"></a><a name="zh-cn_topic_0000001953823221_p956mcpsimp"></a>dst内存地址</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001953823221_row957mcpsimp"><td class="cellrowborder" valign="top" width="47%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001953823221_p959mcpsimp"><a name="zh-cn_topic_0000001953823221_p959mcpsimp"></a><a name="zh-cn_topic_0000001953823221_p959mcpsimp"></a>u32 remoteUserRank</p>
</td>
<td class="cellrowborder" valign="top" width="22%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001953823221_p961mcpsimp"><a name="zh-cn_topic_0000001953823221_p961mcpsimp"></a><a name="zh-cn_topic_0000001953823221_p961mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="31%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001953823221_p963mcpsimp"><a name="zh-cn_topic_0000001953823221_p963mcpsimp"></a><a name="zh-cn_topic_0000001953823221_p963mcpsimp"></a>对端world rank</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001953823221_row964mcpsimp"><td class="cellrowborder" valign="top" width="47%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001953823221_p966mcpsimp"><a name="zh-cn_topic_0000001953823221_p966mcpsimp"></a><a name="zh-cn_topic_0000001953823221_p966mcpsimp"></a>hccl::LinkType inLinkType</p>
</td>
<td class="cellrowborder" valign="top" width="22%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001953823221_p968mcpsimp"><a name="zh-cn_topic_0000001953823221_p968mcpsimp"></a><a name="zh-cn_topic_0000001953823221_p968mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="31%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001953823221_p970mcpsimp"><a name="zh-cn_topic_0000001953823221_p970mcpsimp"></a><a name="zh-cn_topic_0000001953823221_p970mcpsimp"></a>链路类型</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001953823221_row971mcpsimp"><td class="cellrowborder" valign="top" width="47%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001953823221_p973mcpsimp"><a name="zh-cn_topic_0000001953823221_p973mcpsimp"></a><a name="zh-cn_topic_0000001953823221_p973mcpsimp"></a>const u64 reduceAttr</p>
</td>
<td class="cellrowborder" valign="top" width="22%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001953823221_p975mcpsimp"><a name="zh-cn_topic_0000001953823221_p975mcpsimp"></a><a name="zh-cn_topic_0000001953823221_p975mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="31%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001953823221_p977mcpsimp"><a name="zh-cn_topic_0000001953823221_p977mcpsimp"></a><a name="zh-cn_topic_0000001953823221_p977mcpsimp"></a>reduce类型</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001953823221_section978mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS；其他失败。

