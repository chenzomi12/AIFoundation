# RxWithReduce 

## 函数原型<a name="zh-cn_topic_0000001929299930_section7045mcpsimp"></a>

```
// 接收并且做reduce操作，单块内存
HcclResult RxWithReduce(UserMemType recvSrcMemType, u64 recvSrcOffset, void *recvDst, u64 recvLen, void *reduceSrc, void *reduceDst, u64 reduceDataCount, HcclDataType reduceDatatype, HcclReduceOp reduceOp, Stream &stream, const u64 reduceAttr)

// 接收并且做reduce操作，多块内存
HcclResult RxWithReduce(const std::vector<RxWithReduceMemoryInfo> &rxWithReduceMems, HcclDataType reduceDatatype, HcclReduceOp reduceOp, Stream &stream, const u64 reduceAttr)
```

## 函数功能<a name="zh-cn_topic_0000001929299930_section7048mcpsimp"></a>

异步接收数据，将远端指定类型地址中的数据接收到本端dst地址中，并完成reduce操作。

## 参数说明<a name="zh-cn_topic_0000001929299930_section7051mcpsimp"></a>

-   原型1

    **表 1**  参数说明

    <a name="zh-cn_topic_0000001929299930_table7053mcpsimp"></a>
    <table><thead align="left"><tr id="zh-cn_topic_0000001929299930_row7060mcpsimp"><th class="cellrowborder" valign="top" width="28.71%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001929299930_p7062mcpsimp"><a name="zh-cn_topic_0000001929299930_p7062mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7062mcpsimp"></a>参数</p>
    </th>
    <th class="cellrowborder" valign="top" width="13.86%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001929299930_p7064mcpsimp"><a name="zh-cn_topic_0000001929299930_p7064mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7064mcpsimp"></a>输入/输出</p>
    </th>
    <th class="cellrowborder" valign="top" width="57.43000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001929299930_p7066mcpsimp"><a name="zh-cn_topic_0000001929299930_p7066mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7066mcpsimp"></a>说明</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0000001929299930_row7068mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299930_p7070mcpsimp"><a name="zh-cn_topic_0000001929299930_p7070mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7070mcpsimp"></a>UserMemType recvSrcMemType</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299930_p7072mcpsimp"><a name="zh-cn_topic_0000001929299930_p7072mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7072mcpsimp"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299930_p7074mcpsimp"><a name="zh-cn_topic_0000001929299930_p7074mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7074mcpsimp"></a>接收用户内存类型</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000001929299930_row7075mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299930_p7077mcpsimp"><a name="zh-cn_topic_0000001929299930_p7077mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7077mcpsimp"></a>u64 recvSrcOffset</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299930_p7079mcpsimp"><a name="zh-cn_topic_0000001929299930_p7079mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7079mcpsimp"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299930_p7081mcpsimp"><a name="zh-cn_topic_0000001929299930_p7081mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7081mcpsimp"></a>接收源地址偏移</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000001929299930_row7082mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299930_p7084mcpsimp"><a name="zh-cn_topic_0000001929299930_p7084mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7084mcpsimp"></a>void *recvDst</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299930_p7086mcpsimp"><a name="zh-cn_topic_0000001929299930_p7086mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7086mcpsimp"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299930_p1174524634412"><a name="zh-cn_topic_0000001929299930_p1174524634412"></a><a name="zh-cn_topic_0000001929299930_p1174524634412"></a>接收目的地址</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000001929299930_row7088mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299930_p7090mcpsimp"><a name="zh-cn_topic_0000001929299930_p7090mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7090mcpsimp"></a>u64 recvLen</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299930_p7092mcpsimp"><a name="zh-cn_topic_0000001929299930_p7092mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7092mcpsimp"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299930_entry7093mcpsimpp0"><a name="zh-cn_topic_0000001929299930_entry7093mcpsimpp0"></a><a name="zh-cn_topic_0000001929299930_entry7093mcpsimpp0"></a>接收长度</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000001929299930_row7094mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299930_p7096mcpsimp"><a name="zh-cn_topic_0000001929299930_p7096mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7096mcpsimp"></a>void *reduceSrc</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299930_p7098mcpsimp"><a name="zh-cn_topic_0000001929299930_p7098mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7098mcpsimp"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299930_entry7099mcpsimpp0"><a name="zh-cn_topic_0000001929299930_entry7099mcpsimpp0"></a><a name="zh-cn_topic_0000001929299930_entry7099mcpsimpp0"></a>reduce源地址</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000001929299930_row7100mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299930_p7102mcpsimp"><a name="zh-cn_topic_0000001929299930_p7102mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7102mcpsimp"></a>void *reduceDst</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299930_p7104mcpsimp"><a name="zh-cn_topic_0000001929299930_p7104mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7104mcpsimp"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299930_entry7105mcpsimpp0"><a name="zh-cn_topic_0000001929299930_entry7105mcpsimpp0"></a><a name="zh-cn_topic_0000001929299930_entry7105mcpsimpp0"></a>reduce目的地址</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000001929299930_row7106mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299930_p7108mcpsimp"><a name="zh-cn_topic_0000001929299930_p7108mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7108mcpsimp"></a>u64 reduceDataCount</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299930_p7110mcpsimp"><a name="zh-cn_topic_0000001929299930_p7110mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7110mcpsimp"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299930_entry7111mcpsimpp0"><a name="zh-cn_topic_0000001929299930_entry7111mcpsimpp0"></a><a name="zh-cn_topic_0000001929299930_entry7111mcpsimpp0"></a>reduce数据量</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000001929299930_row7112mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299930_p7114mcpsimp"><a name="zh-cn_topic_0000001929299930_p7114mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7114mcpsimp"></a>HcclDataType reduceDatatype</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299930_p7116mcpsimp"><a name="zh-cn_topic_0000001929299930_p7116mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7116mcpsimp"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299930_p7118mcpsimp"><a name="zh-cn_topic_0000001929299930_p7118mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7118mcpsimp"></a>数据类型</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000001929299930_row7119mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299930_p7121mcpsimp"><a name="zh-cn_topic_0000001929299930_p7121mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7121mcpsimp"></a>HcclReduceOp reduceOp</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299930_p7123mcpsimp"><a name="zh-cn_topic_0000001929299930_p7123mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7123mcpsimp"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299930_p7125mcpsimp"><a name="zh-cn_topic_0000001929299930_p7125mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7125mcpsimp"></a>Reduce类型</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000001929299930_row7126mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299930_p7128mcpsimp"><a name="zh-cn_topic_0000001929299930_p7128mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7128mcpsimp"></a>Stream &amp;stream</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299930_p7130mcpsimp"><a name="zh-cn_topic_0000001929299930_p7130mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7130mcpsimp"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299930_p7132mcpsimp"><a name="zh-cn_topic_0000001929299930_p7132mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7132mcpsimp"></a>Stream对象</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000001929299930_row7133mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299930_p7135mcpsimp"><a name="zh-cn_topic_0000001929299930_p7135mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7135mcpsimp"></a>const u64 reduceAttr</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299930_p7137mcpsimp"><a name="zh-cn_topic_0000001929299930_p7137mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7137mcpsimp"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299930_p7139mcpsimp"><a name="zh-cn_topic_0000001929299930_p7139mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7139mcpsimp"></a>Reduce属性</p>
    </td>
    </tr>
    </tbody>
    </table>

-   原型2

    **表 2**  参数说明

    <a name="zh-cn_topic_0000001929299930_table7155mcpsimp"></a>
    <table><thead align="left"><tr id="zh-cn_topic_0000001929299930_row7162mcpsimp"><th class="cellrowborder" valign="top" width="28.71%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001929299930_p7164mcpsimp"><a name="zh-cn_topic_0000001929299930_p7164mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7164mcpsimp"></a>参数</p>
    </th>
    <th class="cellrowborder" valign="top" width="13.86%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001929299930_p7166mcpsimp"><a name="zh-cn_topic_0000001929299930_p7166mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7166mcpsimp"></a>输入/输出</p>
    </th>
    <th class="cellrowborder" valign="top" width="57.43000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001929299930_p7168mcpsimp"><a name="zh-cn_topic_0000001929299930_p7168mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7168mcpsimp"></a>说明</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0000001929299930_row7170mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299930_p7172mcpsimp"><a name="zh-cn_topic_0000001929299930_p7172mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7172mcpsimp"></a>const std::vector&lt;RxWithReduceMemoryInfo&gt; &amp;rxWithReduceMems</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299930_p7174mcpsimp"><a name="zh-cn_topic_0000001929299930_p7174mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7174mcpsimp"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299930_p7176mcpsimp"><a name="zh-cn_topic_0000001929299930_p7176mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7176mcpsimp"></a>算法step信息</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000001929299930_row7177mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299930_p7179mcpsimp"><a name="zh-cn_topic_0000001929299930_p7179mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7179mcpsimp"></a>HcclDataType reduceDatatype</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299930_p7181mcpsimp"><a name="zh-cn_topic_0000001929299930_p7181mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7181mcpsimp"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299930_p7183mcpsimp"><a name="zh-cn_topic_0000001929299930_p7183mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7183mcpsimp"></a>数据类型</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000001929299930_row7184mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299930_p7186mcpsimp"><a name="zh-cn_topic_0000001929299930_p7186mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7186mcpsimp"></a>HcclReduceOp reduceOp</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299930_p7188mcpsimp"><a name="zh-cn_topic_0000001929299930_p7188mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7188mcpsimp"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299930_p7190mcpsimp"><a name="zh-cn_topic_0000001929299930_p7190mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7190mcpsimp"></a>Reduce类型</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000001929299930_row7191mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299930_p7193mcpsimp"><a name="zh-cn_topic_0000001929299930_p7193mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7193mcpsimp"></a>Stream &amp;stream</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299930_p7195mcpsimp"><a name="zh-cn_topic_0000001929299930_p7195mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7195mcpsimp"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299930_p7197mcpsimp"><a name="zh-cn_topic_0000001929299930_p7197mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7197mcpsimp"></a>Stream对象</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000001929299930_row7198mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001929299930_p7200mcpsimp"><a name="zh-cn_topic_0000001929299930_p7200mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7200mcpsimp"></a>const u64 reduceAttr</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001929299930_p7202mcpsimp"><a name="zh-cn_topic_0000001929299930_p7202mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7202mcpsimp"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001929299930_p7204mcpsimp"><a name="zh-cn_topic_0000001929299930_p7204mcpsimp"></a><a name="zh-cn_topic_0000001929299930_p7204mcpsimp"></a>Reduce属性</p>
    </td>
    </tr>
    </tbody>
    </table>

## 返回值<a name="zh-cn_topic_0000001929299930_section7140mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

## 约束说明<a name="zh-cn_topic_0000001929299930_section7143mcpsimp"></a>

无

