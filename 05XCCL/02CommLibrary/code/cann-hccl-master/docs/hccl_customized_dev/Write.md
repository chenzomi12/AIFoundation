# Write 

## 函数原型<a name="zh-cn_topic_0000001929459322_section8271mcpsimp"></a>

HcclResult Write\(const void \*localAddr, UserMemType remoteMemType, u64 remoteOffset, u64 len, Stream &stream\)

## 函数功能<a name="zh-cn_topic_0000001929459322_section8274mcpsimp"></a>

单边写数据。

## 参数说明<a name="zh-cn_topic_0000001929459322_section8277mcpsimp"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0000001929459322_table8279mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001929459322_row8286mcpsimp"><th class="cellrowborder" valign="top" width="28.71%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001929459322_p8288mcpsimp"><a name="zh-cn_topic_0000001929459322_p8288mcpsimp"></a><a name="zh-cn_topic_0000001929459322_p8288mcpsimp"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="13.86%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001929459322_p8290mcpsimp"><a name="zh-cn_topic_0000001929459322_p8290mcpsimp"></a><a name="zh-cn_topic_0000001929459322_p8290mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="57.43000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001929459322_p8292mcpsimp"><a name="zh-cn_topic_0000001929459322_p8292mcpsimp"></a><a name="zh-cn_topic_0000001929459322_p8292mcpsimp"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001929459322_row8294mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929459322_p8296mcpsimp"><a name="zh-cn_topic_0000001929459322_p8296mcpsimp"></a><a name="zh-cn_topic_0000001929459322_p8296mcpsimp"></a>const void *localAddr</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929459322_p8298mcpsimp"><a name="zh-cn_topic_0000001929459322_p8298mcpsimp"></a><a name="zh-cn_topic_0000001929459322_p8298mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929459322_p8300mcpsimp"><a name="zh-cn_topic_0000001929459322_p8300mcpsimp"></a><a name="zh-cn_topic_0000001929459322_p8300mcpsimp"></a>算法step信息</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929459322_row8301mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929459322_p8303mcpsimp"><a name="zh-cn_topic_0000001929459322_p8303mcpsimp"></a><a name="zh-cn_topic_0000001929459322_p8303mcpsimp"></a>UserMemType remoteMemType</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929459322_p8305mcpsimp"><a name="zh-cn_topic_0000001929459322_p8305mcpsimp"></a><a name="zh-cn_topic_0000001929459322_p8305mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929459322_entry8306mcpsimpp0"><a name="zh-cn_topic_0000001929459322_entry8306mcpsimpp0"></a><a name="zh-cn_topic_0000001929459322_entry8306mcpsimpp0"></a>远端用户内存类型</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929459322_row8307mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929459322_p8309mcpsimp"><a name="zh-cn_topic_0000001929459322_p8309mcpsimp"></a><a name="zh-cn_topic_0000001929459322_p8309mcpsimp"></a>u64 remoteOffset</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929459322_p8311mcpsimp"><a name="zh-cn_topic_0000001929459322_p8311mcpsimp"></a><a name="zh-cn_topic_0000001929459322_p8311mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929459322_entry8312mcpsimpp0"><a name="zh-cn_topic_0000001929459322_entry8312mcpsimpp0"></a><a name="zh-cn_topic_0000001929459322_entry8312mcpsimpp0"></a>远端地址偏移</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929459322_row8313mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929459322_p8315mcpsimp"><a name="zh-cn_topic_0000001929459322_p8315mcpsimp"></a><a name="zh-cn_topic_0000001929459322_p8315mcpsimp"></a>u64 len</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929459322_p8317mcpsimp"><a name="zh-cn_topic_0000001929459322_p8317mcpsimp"></a><a name="zh-cn_topic_0000001929459322_p8317mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929459322_entry8318mcpsimpp0"><a name="zh-cn_topic_0000001929459322_entry8318mcpsimpp0"></a><a name="zh-cn_topic_0000001929459322_entry8318mcpsimpp0"></a>数据长度</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929459322_row8319mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929459322_p8321mcpsimp"><a name="zh-cn_topic_0000001929459322_p8321mcpsimp"></a><a name="zh-cn_topic_0000001929459322_p8321mcpsimp"></a>Stream &amp;stream</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929459322_p8323mcpsimp"><a name="zh-cn_topic_0000001929459322_p8323mcpsimp"></a><a name="zh-cn_topic_0000001929459322_p8323mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929459322_p8325mcpsimp"><a name="zh-cn_topic_0000001929459322_p8325mcpsimp"></a><a name="zh-cn_topic_0000001929459322_p8325mcpsimp"></a>Stream对象</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001929459322_section8326mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

## 约束说明<a name="zh-cn_topic_0000001929459322_section8329mcpsimp"></a>

无

