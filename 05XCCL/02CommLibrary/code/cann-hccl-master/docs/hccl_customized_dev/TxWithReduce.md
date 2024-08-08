# TxWithReduce 

## 函数原型<a name="zh-cn_topic_0000001956458781_section6908mcpsimp"></a>

```
// 发送一块内存数据
HcclResult TxWithReduce(UserMemType dstMemType, u64 dstOffset, const void *src, u64 len, const HcclDataType datatype, HcclReduceOp redOp, Stream &stream)

// 发送多块内存数据
HcclResult TxWithReduce(const std::vector<TxMemoryInfo> &txWithReduceMems, const HcclDataType datatype, HcclReduceOp redOp, Stream &stream)
```

## 函数功能<a name="zh-cn_topic_0000001956458781_section6911mcpsimp"></a>

异步发送数据，将本端src地址的数据发送到远端指定类型地址中，并完成reduce操作。

## 参数说明<a name="zh-cn_topic_0000001956458781_section6914mcpsimp"></a>

-   原型1

    **表 1**  参数说明

    <a name="zh-cn_topic_0000001956458781_table6916mcpsimp"></a>
    <table><thead align="left"><tr id="zh-cn_topic_0000001956458781_row6923mcpsimp"><th class="cellrowborder" valign="top" width="28.71%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001956458781_p6925mcpsimp"><a name="zh-cn_topic_0000001956458781_p6925mcpsimp"></a><a name="zh-cn_topic_0000001956458781_p6925mcpsimp"></a>参数</p>
    </th>
    <th class="cellrowborder" valign="top" width="13.86%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001956458781_p6927mcpsimp"><a name="zh-cn_topic_0000001956458781_p6927mcpsimp"></a><a name="zh-cn_topic_0000001956458781_p6927mcpsimp"></a>输入/输出</p>
    </th>
    <th class="cellrowborder" valign="top" width="57.43000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001956458781_p6929mcpsimp"><a name="zh-cn_topic_0000001956458781_p6929mcpsimp"></a><a name="zh-cn_topic_0000001956458781_p6929mcpsimp"></a>说明</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0000001956458781_row6931mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956458781_p6933mcpsimp"><a name="zh-cn_topic_0000001956458781_p6933mcpsimp"></a><a name="zh-cn_topic_0000001956458781_p6933mcpsimp"></a>UserMemType dstMemType</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956458781_p6935mcpsimp"><a name="zh-cn_topic_0000001956458781_p6935mcpsimp"></a><a name="zh-cn_topic_0000001956458781_p6935mcpsimp"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956458781_p6937mcpsimp"><a name="zh-cn_topic_0000001956458781_p6937mcpsimp"></a><a name="zh-cn_topic_0000001956458781_p6937mcpsimp"></a>对端用户内存类型</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000001956458781_row6938mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956458781_p6940mcpsimp"><a name="zh-cn_topic_0000001956458781_p6940mcpsimp"></a><a name="zh-cn_topic_0000001956458781_p6940mcpsimp"></a>u64 dstOffset</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956458781_p6942mcpsimp"><a name="zh-cn_topic_0000001956458781_p6942mcpsimp"></a><a name="zh-cn_topic_0000001956458781_p6942mcpsimp"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956458781_p6944mcpsimp"><a name="zh-cn_topic_0000001956458781_p6944mcpsimp"></a><a name="zh-cn_topic_0000001956458781_p6944mcpsimp"></a>对端内存偏移</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000001956458781_row6945mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956458781_p6947mcpsimp"><a name="zh-cn_topic_0000001956458781_p6947mcpsimp"></a><a name="zh-cn_topic_0000001956458781_p6947mcpsimp"></a>const void *src</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956458781_p6949mcpsimp"><a name="zh-cn_topic_0000001956458781_p6949mcpsimp"></a><a name="zh-cn_topic_0000001956458781_p6949mcpsimp"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956458781_p6951mcpsimp"><a name="zh-cn_topic_0000001956458781_p6951mcpsimp"></a><a name="zh-cn_topic_0000001956458781_p6951mcpsimp"></a>源地址</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000001956458781_row6952mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956458781_p6954mcpsimp"><a name="zh-cn_topic_0000001956458781_p6954mcpsimp"></a><a name="zh-cn_topic_0000001956458781_p6954mcpsimp"></a>u64 len</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956458781_p6956mcpsimp"><a name="zh-cn_topic_0000001956458781_p6956mcpsimp"></a><a name="zh-cn_topic_0000001956458781_p6956mcpsimp"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956458781_p6958mcpsimp"><a name="zh-cn_topic_0000001956458781_p6958mcpsimp"></a><a name="zh-cn_topic_0000001956458781_p6958mcpsimp"></a>发送数据大小</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000001956458781_row6959mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956458781_p6961mcpsimp"><a name="zh-cn_topic_0000001956458781_p6961mcpsimp"></a><a name="zh-cn_topic_0000001956458781_p6961mcpsimp"></a>const HcclDataType datatype</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956458781_p6963mcpsimp"><a name="zh-cn_topic_0000001956458781_p6963mcpsimp"></a><a name="zh-cn_topic_0000001956458781_p6963mcpsimp"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956458781_p6965mcpsimp"><a name="zh-cn_topic_0000001956458781_p6965mcpsimp"></a><a name="zh-cn_topic_0000001956458781_p6965mcpsimp"></a>数据类型</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000001956458781_row6966mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956458781_p6968mcpsimp"><a name="zh-cn_topic_0000001956458781_p6968mcpsimp"></a><a name="zh-cn_topic_0000001956458781_p6968mcpsimp"></a>HcclReduceOp redOp</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956458781_p6970mcpsimp"><a name="zh-cn_topic_0000001956458781_p6970mcpsimp"></a><a name="zh-cn_topic_0000001956458781_p6970mcpsimp"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956458781_p6972mcpsimp"><a name="zh-cn_topic_0000001956458781_p6972mcpsimp"></a><a name="zh-cn_topic_0000001956458781_p6972mcpsimp"></a>Reduce类型</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000001956458781_row6973mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956458781_p6975mcpsimp"><a name="zh-cn_topic_0000001956458781_p6975mcpsimp"></a><a name="zh-cn_topic_0000001956458781_p6975mcpsimp"></a>Stream &amp;stream</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956458781_p6977mcpsimp"><a name="zh-cn_topic_0000001956458781_p6977mcpsimp"></a><a name="zh-cn_topic_0000001956458781_p6977mcpsimp"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956458781_p6979mcpsimp"><a name="zh-cn_topic_0000001956458781_p6979mcpsimp"></a><a name="zh-cn_topic_0000001956458781_p6979mcpsimp"></a>Stream对象</p>
    </td>
    </tr>
    </tbody>
    </table>

-   原型2

    **表 2**  参数说明

    <a name="zh-cn_topic_0000001956458781_table6995mcpsimp"></a>
    <table><thead align="left"><tr id="zh-cn_topic_0000001956458781_row7002mcpsimp"><th class="cellrowborder" valign="top" width="28.71%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001956458781_p7004mcpsimp"><a name="zh-cn_topic_0000001956458781_p7004mcpsimp"></a><a name="zh-cn_topic_0000001956458781_p7004mcpsimp"></a>参数</p>
    </th>
    <th class="cellrowborder" valign="top" width="13.86%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001956458781_p7006mcpsimp"><a name="zh-cn_topic_0000001956458781_p7006mcpsimp"></a><a name="zh-cn_topic_0000001956458781_p7006mcpsimp"></a>输入/输出</p>
    </th>
    <th class="cellrowborder" valign="top" width="57.43000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001956458781_p7008mcpsimp"><a name="zh-cn_topic_0000001956458781_p7008mcpsimp"></a><a name="zh-cn_topic_0000001956458781_p7008mcpsimp"></a>说明</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0000001956458781_row7010mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956458781_p7012mcpsimp"><a name="zh-cn_topic_0000001956458781_p7012mcpsimp"></a><a name="zh-cn_topic_0000001956458781_p7012mcpsimp"></a>const std::vector&lt;TxMemoryInfo&gt; &amp;txWithReduceMems</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956458781_p7014mcpsimp"><a name="zh-cn_topic_0000001956458781_p7014mcpsimp"></a><a name="zh-cn_topic_0000001956458781_p7014mcpsimp"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956458781_p7016mcpsimp"><a name="zh-cn_topic_0000001956458781_p7016mcpsimp"></a><a name="zh-cn_topic_0000001956458781_p7016mcpsimp"></a>发送内存信息</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000001956458781_row7017mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956458781_p7019mcpsimp"><a name="zh-cn_topic_0000001956458781_p7019mcpsimp"></a><a name="zh-cn_topic_0000001956458781_p7019mcpsimp"></a>const HcclDataType datatype</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956458781_p7021mcpsimp"><a name="zh-cn_topic_0000001956458781_p7021mcpsimp"></a><a name="zh-cn_topic_0000001956458781_p7021mcpsimp"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956458781_p7023mcpsimp"><a name="zh-cn_topic_0000001956458781_p7023mcpsimp"></a><a name="zh-cn_topic_0000001956458781_p7023mcpsimp"></a>数据类型</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000001956458781_row7024mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956458781_p7026mcpsimp"><a name="zh-cn_topic_0000001956458781_p7026mcpsimp"></a><a name="zh-cn_topic_0000001956458781_p7026mcpsimp"></a>HcclReduceOp redOp</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956458781_p7028mcpsimp"><a name="zh-cn_topic_0000001956458781_p7028mcpsimp"></a><a name="zh-cn_topic_0000001956458781_p7028mcpsimp"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956458781_p7030mcpsimp"><a name="zh-cn_topic_0000001956458781_p7030mcpsimp"></a><a name="zh-cn_topic_0000001956458781_p7030mcpsimp"></a>Reduce类型</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000001956458781_row7031mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956458781_p7033mcpsimp"><a name="zh-cn_topic_0000001956458781_p7033mcpsimp"></a><a name="zh-cn_topic_0000001956458781_p7033mcpsimp"></a>Stream &amp;stream</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956458781_p7035mcpsimp"><a name="zh-cn_topic_0000001956458781_p7035mcpsimp"></a><a name="zh-cn_topic_0000001956458781_p7035mcpsimp"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956458781_p7037mcpsimp"><a name="zh-cn_topic_0000001956458781_p7037mcpsimp"></a><a name="zh-cn_topic_0000001956458781_p7037mcpsimp"></a>Stream对象</p>
    </td>
    </tr>
    </tbody>
    </table>

## 返回值<a name="zh-cn_topic_0000001956458781_section6980mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

## 约束说明<a name="zh-cn_topic_0000001956458781_section6983mcpsimp"></a>

无

