# RxAsync 

## 函数原型<a name="zh-cn_topic_0000001956618577_section7228mcpsimp"></a>

```
// 异步接收单块内存
HcclResult RxAsync(UserMemType srcMemType, u64 srcOffset, void *dst, u64 len, Stream &stream)

// 异步接收多块内存
HcclResult RxAsync(std::vector<RxMemoryInfo>& rxMems, Stream &stream)
```

## 函数功能<a name="zh-cn_topic_0000001956618577_section7231mcpsimp"></a>

异步接收数据，将远端指定类型地址中的数据接收到本端dst地址中。

## 参数说明<a name="zh-cn_topic_0000001956618577_section7234mcpsimp"></a>

-   原型1

    **表 1**  参数说明

    <a name="zh-cn_topic_0000001956618577_table7236mcpsimp"></a>
    <table><thead align="left"><tr id="zh-cn_topic_0000001956618577_row7243mcpsimp"><th class="cellrowborder" valign="top" width="28.71%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001956618577_p7245mcpsimp"><a name="zh-cn_topic_0000001956618577_p7245mcpsimp"></a><a name="zh-cn_topic_0000001956618577_p7245mcpsimp"></a>参数</p>
    </th>
    <th class="cellrowborder" valign="top" width="13.86%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001956618577_p7247mcpsimp"><a name="zh-cn_topic_0000001956618577_p7247mcpsimp"></a><a name="zh-cn_topic_0000001956618577_p7247mcpsimp"></a>输入/输出</p>
    </th>
    <th class="cellrowborder" valign="top" width="57.43000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001956618577_p7249mcpsimp"><a name="zh-cn_topic_0000001956618577_p7249mcpsimp"></a><a name="zh-cn_topic_0000001956618577_p7249mcpsimp"></a>说明</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0000001956618577_row7251mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956618577_p7253mcpsimp"><a name="zh-cn_topic_0000001956618577_p7253mcpsimp"></a><a name="zh-cn_topic_0000001956618577_p7253mcpsimp"></a>UserMemType srcMemType</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956618577_p7255mcpsimp"><a name="zh-cn_topic_0000001956618577_p7255mcpsimp"></a><a name="zh-cn_topic_0000001956618577_p7255mcpsimp"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956618577_p7257mcpsimp"><a name="zh-cn_topic_0000001956618577_p7257mcpsimp"></a><a name="zh-cn_topic_0000001956618577_p7257mcpsimp"></a>算法step信息</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000001956618577_row7258mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956618577_p7260mcpsimp"><a name="zh-cn_topic_0000001956618577_p7260mcpsimp"></a><a name="zh-cn_topic_0000001956618577_p7260mcpsimp"></a>u64 srcOffset</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956618577_p7262mcpsimp"><a name="zh-cn_topic_0000001956618577_p7262mcpsimp"></a><a name="zh-cn_topic_0000001956618577_p7262mcpsimp"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956618577_entry7263mcpsimpp0"><a name="zh-cn_topic_0000001956618577_entry7263mcpsimpp0"></a><a name="zh-cn_topic_0000001956618577_entry7263mcpsimpp0"></a>源地址偏移</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000001956618577_row7264mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956618577_p7266mcpsimp"><a name="zh-cn_topic_0000001956618577_p7266mcpsimp"></a><a name="zh-cn_topic_0000001956618577_p7266mcpsimp"></a>void *dst</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956618577_p7268mcpsimp"><a name="zh-cn_topic_0000001956618577_p7268mcpsimp"></a><a name="zh-cn_topic_0000001956618577_p7268mcpsimp"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956618577_entry7269mcpsimpp0"><a name="zh-cn_topic_0000001956618577_entry7269mcpsimpp0"></a><a name="zh-cn_topic_0000001956618577_entry7269mcpsimpp0"></a>目的地址</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000001956618577_row7270mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956618577_p7272mcpsimp"><a name="zh-cn_topic_0000001956618577_p7272mcpsimp"></a><a name="zh-cn_topic_0000001956618577_p7272mcpsimp"></a>u64 len</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956618577_p7274mcpsimp"><a name="zh-cn_topic_0000001956618577_p7274mcpsimp"></a><a name="zh-cn_topic_0000001956618577_p7274mcpsimp"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956618577_entry7275mcpsimpp0"><a name="zh-cn_topic_0000001956618577_entry7275mcpsimpp0"></a><a name="zh-cn_topic_0000001956618577_entry7275mcpsimpp0"></a>数据长度</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000001956618577_row7276mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956618577_p7278mcpsimp"><a name="zh-cn_topic_0000001956618577_p7278mcpsimp"></a><a name="zh-cn_topic_0000001956618577_p7278mcpsimp"></a>Stream &amp;stream</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956618577_p7280mcpsimp"><a name="zh-cn_topic_0000001956618577_p7280mcpsimp"></a><a name="zh-cn_topic_0000001956618577_p7280mcpsimp"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956618577_p7282mcpsimp"><a name="zh-cn_topic_0000001956618577_p7282mcpsimp"></a><a name="zh-cn_topic_0000001956618577_p7282mcpsimp"></a>Stream对象</p>
    </td>
    </tr>
    </tbody>
    </table>

-   原型2

    **表 2**  参数说明

    <a name="zh-cn_topic_0000001956618577_table7298mcpsimp"></a>
    <table><thead align="left"><tr id="zh-cn_topic_0000001956618577_row7305mcpsimp"><th class="cellrowborder" valign="top" width="28.71%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001956618577_p7307mcpsimp"><a name="zh-cn_topic_0000001956618577_p7307mcpsimp"></a><a name="zh-cn_topic_0000001956618577_p7307mcpsimp"></a>参数</p>
    </th>
    <th class="cellrowborder" valign="top" width="13.86%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001956618577_p7309mcpsimp"><a name="zh-cn_topic_0000001956618577_p7309mcpsimp"></a><a name="zh-cn_topic_0000001956618577_p7309mcpsimp"></a>输入/输出</p>
    </th>
    <th class="cellrowborder" valign="top" width="57.43000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001956618577_p7311mcpsimp"><a name="zh-cn_topic_0000001956618577_p7311mcpsimp"></a><a name="zh-cn_topic_0000001956618577_p7311mcpsimp"></a>说明</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0000001956618577_row7313mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956618577_p7315mcpsimp"><a name="zh-cn_topic_0000001956618577_p7315mcpsimp"></a><a name="zh-cn_topic_0000001956618577_p7315mcpsimp"></a>std::vector&lt;RxMemoryInfo&gt;&amp; rxMems</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956618577_p7317mcpsimp"><a name="zh-cn_topic_0000001956618577_p7317mcpsimp"></a><a name="zh-cn_topic_0000001956618577_p7317mcpsimp"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956618577_p7319mcpsimp"><a name="zh-cn_topic_0000001956618577_p7319mcpsimp"></a><a name="zh-cn_topic_0000001956618577_p7319mcpsimp"></a>算法step信息</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000001956618577_row7320mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001956618577_p7322mcpsimp"><a name="zh-cn_topic_0000001956618577_p7322mcpsimp"></a><a name="zh-cn_topic_0000001956618577_p7322mcpsimp"></a>Stream &amp;stream</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001956618577_p7324mcpsimp"><a name="zh-cn_topic_0000001956618577_p7324mcpsimp"></a><a name="zh-cn_topic_0000001956618577_p7324mcpsimp"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001956618577_p7326mcpsimp"><a name="zh-cn_topic_0000001956618577_p7326mcpsimp"></a><a name="zh-cn_topic_0000001956618577_p7326mcpsimp"></a>Stream对象</p>
    </td>
    </tr>
    </tbody>
    </table>

## 返回值<a name="zh-cn_topic_0000001956618577_section7283mcpsimp"></a>

HcclResult：接口成功返回HCCL\_SUCCESS。其他失败。

## 约束说明<a name="zh-cn_topic_0000001956618577_section7286mcpsimp"></a>

无

