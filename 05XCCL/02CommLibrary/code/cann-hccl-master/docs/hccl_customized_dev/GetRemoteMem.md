# GetRemoteMem 

## 函数原型<a name="zh-cn_topic_0000001929299958_section7791mcpsimp"></a>

HcclResult GetRemoteMem\(UserMemType memType, void \*\*remotePtr\)

## 函数功能<a name="zh-cn_topic_0000001929299958_section7794mcpsimp"></a>

获取远端交换的mem。

## 参数说明<a name="zh-cn_topic_0000001929299958_section7797mcpsimp"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0000001929299958_table7799mcpsimp"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001929299958_row7806mcpsimp"><th class="cellrowborder" valign="top" width="28.71%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001929299958_p7808mcpsimp"><a name="zh-cn_topic_0000001929299958_p7808mcpsimp"></a><a name="zh-cn_topic_0000001929299958_p7808mcpsimp"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="13.86%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001929299958_p7810mcpsimp"><a name="zh-cn_topic_0000001929299958_p7810mcpsimp"></a><a name="zh-cn_topic_0000001929299958_p7810mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="57.43000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001929299958_p7812mcpsimp"><a name="zh-cn_topic_0000001929299958_p7812mcpsimp"></a><a name="zh-cn_topic_0000001929299958_p7812mcpsimp"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001929299958_row7814mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299958_p7816mcpsimp"><a name="zh-cn_topic_0000001929299958_p7816mcpsimp"></a><a name="zh-cn_topic_0000001929299958_p7816mcpsimp"></a>UserMemType memType</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299958_p7818mcpsimp"><a name="zh-cn_topic_0000001929299958_p7818mcpsimp"></a><a name="zh-cn_topic_0000001929299958_p7818mcpsimp"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299958_p7820mcpsimp"><a name="zh-cn_topic_0000001929299958_p7820mcpsimp"></a><a name="zh-cn_topic_0000001929299958_p7820mcpsimp"></a>用户内存类型</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001929299958_row7821mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299958_p7823mcpsimp"><a name="zh-cn_topic_0000001929299958_p7823mcpsimp"></a><a name="zh-cn_topic_0000001929299958_p7823mcpsimp"></a>void **remotePtr</p>
</td>
<td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299958_p7825mcpsimp"><a name="zh-cn_topic_0000001929299958_p7825mcpsimp"></a><a name="zh-cn_topic_0000001929299958_p7825mcpsimp"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299958_p7827mcpsimp"><a name="zh-cn_topic_0000001929299958_p7827mcpsimp"></a><a name="zh-cn_topic_0000001929299958_p7827mcpsimp"></a>对端内存地址</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001929299958_section7828mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

## 约束说明<a name="zh-cn_topic_0000001929299958_section7831mcpsimp"></a>

无

