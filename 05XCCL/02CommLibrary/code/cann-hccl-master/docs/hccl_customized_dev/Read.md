# Read 

## 函数原型<a name="zh-cn_topic_0000001956458821_section8333mcpsimp"></a>

HcclResult Read\(const void \*localAddr, UserMemType remoteMemType, u64 remoteOffset, u64 len, Stream &stream\)

## 函数功能<a name="zh-cn_topic_0000001956458821_section8336mcpsimp"></a>

单边读数据。

## 参数说明<a name="zh-cn_topic_0000001956458821_section8339mcpsimp"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0000001956458821_table8341mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001956458821_row8348mcpsimp"><th class="cellrowborder" valign="top" width="28.71%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001956458821_p8350mcpsimp"><a name="zh-cn_topic_0000001956458821_p8350mcpsimp"></a><a name="zh-cn_topic_0000001956458821_p8350mcpsimp"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="13.86%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001956458821_p8352mcpsimp"><a name="zh-cn_topic_0000001956458821_p8352mcpsimp"></a><a name="zh-cn_topic_0000001956458821_p8352mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="57.43000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001956458821_p8354mcpsimp"><a name="zh-cn_topic_0000001956458821_p8354mcpsimp"></a><a name="zh-cn_topic_0000001956458821_p8354mcpsimp"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001956458821_row8356mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956458821_p8358mcpsimp"><a name="zh-cn_topic_0000001956458821_p8358mcpsimp"></a><a name="zh-cn_topic_0000001956458821_p8358mcpsimp"></a>const void *localAddr</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956458821_p8360mcpsimp"><a name="zh-cn_topic_0000001956458821_p8360mcpsimp"></a><a name="zh-cn_topic_0000001956458821_p8360mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956458821_p8362mcpsimp"><a name="zh-cn_topic_0000001956458821_p8362mcpsimp"></a><a name="zh-cn_topic_0000001956458821_p8362mcpsimp"></a>算法step信息</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001956458821_row8363mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956458821_p8365mcpsimp"><a name="zh-cn_topic_0000001956458821_p8365mcpsimp"></a><a name="zh-cn_topic_0000001956458821_p8365mcpsimp"></a>UserMemType remoteMemType</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956458821_p8367mcpsimp"><a name="zh-cn_topic_0000001956458821_p8367mcpsimp"></a><a name="zh-cn_topic_0000001956458821_p8367mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956458821_entry8306mcpsimpp0"><a name="zh-cn_topic_0000001956458821_entry8306mcpsimpp0"></a><a name="zh-cn_topic_0000001956458821_entry8306mcpsimpp0"></a>远端用户内存类型</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001956458821_row8369mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956458821_p8371mcpsimp"><a name="zh-cn_topic_0000001956458821_p8371mcpsimp"></a><a name="zh-cn_topic_0000001956458821_p8371mcpsimp"></a>u64 remoteOffset</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956458821_p8373mcpsimp"><a name="zh-cn_topic_0000001956458821_p8373mcpsimp"></a><a name="zh-cn_topic_0000001956458821_p8373mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956458821_entry8312mcpsimpp0"><a name="zh-cn_topic_0000001956458821_entry8312mcpsimpp0"></a><a name="zh-cn_topic_0000001956458821_entry8312mcpsimpp0"></a>远端地址偏移</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001956458821_row8375mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956458821_p8377mcpsimp"><a name="zh-cn_topic_0000001956458821_p8377mcpsimp"></a><a name="zh-cn_topic_0000001956458821_p8377mcpsimp"></a>u64 len</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956458821_p8379mcpsimp"><a name="zh-cn_topic_0000001956458821_p8379mcpsimp"></a><a name="zh-cn_topic_0000001956458821_p8379mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956458821_entry8318mcpsimpp0"><a name="zh-cn_topic_0000001956458821_entry8318mcpsimpp0"></a><a name="zh-cn_topic_0000001956458821_entry8318mcpsimpp0"></a>数据长度</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001956458821_row8381mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956458821_p8383mcpsimp"><a name="zh-cn_topic_0000001956458821_p8383mcpsimp"></a><a name="zh-cn_topic_0000001956458821_p8383mcpsimp"></a>Stream &amp;stream</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956458821_p8385mcpsimp"><a name="zh-cn_topic_0000001956458821_p8385mcpsimp"></a><a name="zh-cn_topic_0000001956458821_p8385mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956458821_p8387mcpsimp"><a name="zh-cn_topic_0000001956458821_p8387mcpsimp"></a><a name="zh-cn_topic_0000001956458821_p8387mcpsimp"></a>Stream对象</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001956458821_section8388mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

## 约束说明<a name="zh-cn_topic_0000001956458821_section8391mcpsimp"></a>

无

