# DeviceMem

## 函数原型<a name="zh-cn_topic_0000001933105872_section9788mcpsimp"></a>

```
// DeviceMem构造函数
DeviceMem()
DeviceMem(void *ptr, u64 size, bool owner = false)
//DevceMem 拷贝构造函数
DeviceMem(const DeviceMem &that)
// DeviceMem移动构造函数
DeviceMem(DeviceMem &&that)
```

## 函数功能<a name="zh-cn_topic_0000001933105872_section9791mcpsimp"></a>

DeviceMem构造函数

## 参数说明<a name="zh-cn_topic_0000001933105872_section9794mcpsimp"></a>

-   DeviceMem构造函数

    **表 1**  参数说明

    <a name="zh-cn_topic_0000001933105872_table9812mcpsimp"></a>
    <table><thead align="left"><tr id="zh-cn_topic_0000001933105872_row9819mcpsimp"><th class="cellrowborder" valign="top" width="28.71%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001933105872_p9821mcpsimp"><a name="zh-cn_topic_0000001933105872_p9821mcpsimp"></a><a name="zh-cn_topic_0000001933105872_p9821mcpsimp"></a>参数</p>
    </th>
    <th class="cellrowborder" valign="top" width="13.86%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001933105872_p9823mcpsimp"><a name="zh-cn_topic_0000001933105872_p9823mcpsimp"></a><a name="zh-cn_topic_0000001933105872_p9823mcpsimp"></a>输入/输出</p>
    </th>
    <th class="cellrowborder" valign="top" width="57.43000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001933105872_p9825mcpsimp"><a name="zh-cn_topic_0000001933105872_p9825mcpsimp"></a><a name="zh-cn_topic_0000001933105872_p9825mcpsimp"></a>说明</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0000001933105872_row9827mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001933105872_p9829mcpsimp"><a name="zh-cn_topic_0000001933105872_p9829mcpsimp"></a><a name="zh-cn_topic_0000001933105872_p9829mcpsimp"></a>void *ptr</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001933105872_p9831mcpsimp"><a name="zh-cn_topic_0000001933105872_p9831mcpsimp"></a><a name="zh-cn_topic_0000001933105872_p9831mcpsimp"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001933105872_p9833mcpsimp"><a name="zh-cn_topic_0000001933105872_p9833mcpsimp"></a><a name="zh-cn_topic_0000001933105872_p9833mcpsimp"></a>内存地址</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000001933105872_row9834mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001933105872_p9836mcpsimp"><a name="zh-cn_topic_0000001933105872_p9836mcpsimp"></a><a name="zh-cn_topic_0000001933105872_p9836mcpsimp"></a>u64 size</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001933105872_p9838mcpsimp"><a name="zh-cn_topic_0000001933105872_p9838mcpsimp"></a><a name="zh-cn_topic_0000001933105872_p9838mcpsimp"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001933105872_p9840mcpsimp"><a name="zh-cn_topic_0000001933105872_p9840mcpsimp"></a><a name="zh-cn_topic_0000001933105872_p9840mcpsimp"></a>内存大小</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000001933105872_row9841mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001933105872_p9843mcpsimp"><a name="zh-cn_topic_0000001933105872_p9843mcpsimp"></a><a name="zh-cn_topic_0000001933105872_p9843mcpsimp"></a>bool owner</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001933105872_p9845mcpsimp"><a name="zh-cn_topic_0000001933105872_p9845mcpsimp"></a><a name="zh-cn_topic_0000001933105872_p9845mcpsimp"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001933105872_p9847mcpsimp"><a name="zh-cn_topic_0000001933105872_p9847mcpsimp"></a><a name="zh-cn_topic_0000001933105872_p9847mcpsimp"></a>是否是资源拥有者</p>
    </td>
    </tr>
    </tbody>
    </table>

-   DeviceMem拷贝构造函数

    **表 2**  参数说明

    <a name="zh-cn_topic_0000001933105872_table9863mcpsimp"></a>
    <table><thead align="left"><tr id="zh-cn_topic_0000001933105872_row9870mcpsimp"><th class="cellrowborder" valign="top" width="28.71%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001933105872_p9872mcpsimp"><a name="zh-cn_topic_0000001933105872_p9872mcpsimp"></a><a name="zh-cn_topic_0000001933105872_p9872mcpsimp"></a>参数</p>
    </th>
    <th class="cellrowborder" valign="top" width="13.86%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001933105872_p9874mcpsimp"><a name="zh-cn_topic_0000001933105872_p9874mcpsimp"></a><a name="zh-cn_topic_0000001933105872_p9874mcpsimp"></a>输入/输出</p>
    </th>
    <th class="cellrowborder" valign="top" width="57.43000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001933105872_p9876mcpsimp"><a name="zh-cn_topic_0000001933105872_p9876mcpsimp"></a><a name="zh-cn_topic_0000001933105872_p9876mcpsimp"></a>说明</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0000001933105872_row9878mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001933105872_p11355165683114"><a name="zh-cn_topic_0000001933105872_p11355165683114"></a><a name="zh-cn_topic_0000001933105872_p11355165683114"></a>const DeviceMem &amp;that</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001933105872_p9882mcpsimp"><a name="zh-cn_topic_0000001933105872_p9882mcpsimp"></a><a name="zh-cn_topic_0000001933105872_p9882mcpsimp"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001933105872_p9884mcpsimp"><a name="zh-cn_topic_0000001933105872_p9884mcpsimp"></a><a name="zh-cn_topic_0000001933105872_p9884mcpsimp"></a>DeviceMem对象</p>
    </td>
    </tr>
    </tbody>
    </table>

-   DeviceMem移动构造函数

    **表 3**  参数说明

    <a name="zh-cn_topic_0000001933105872_table1265513203315"></a>
    <table><thead align="left"><tr id="zh-cn_topic_0000001933105872_row19655203223311"><th class="cellrowborder" valign="top" width="28.71%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001933105872_p96559322339"><a name="zh-cn_topic_0000001933105872_p96559322339"></a><a name="zh-cn_topic_0000001933105872_p96559322339"></a>参数</p>
    </th>
    <th class="cellrowborder" valign="top" width="13.86%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001933105872_p13655143253312"><a name="zh-cn_topic_0000001933105872_p13655143253312"></a><a name="zh-cn_topic_0000001933105872_p13655143253312"></a>输入/输出</p>
    </th>
    <th class="cellrowborder" valign="top" width="57.43000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001933105872_p16551332133320"><a name="zh-cn_topic_0000001933105872_p16551332133320"></a><a name="zh-cn_topic_0000001933105872_p16551332133320"></a>说明</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0000001933105872_row206550320332"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001933105872_p721320161318"><a name="zh-cn_topic_0000001933105872_p721320161318"></a><a name="zh-cn_topic_0000001933105872_p721320161318"></a>DeviceMem &amp;&amp;that</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001933105872_p1365583212332"><a name="zh-cn_topic_0000001933105872_p1365583212332"></a><a name="zh-cn_topic_0000001933105872_p1365583212332"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001933105872_p15439101843114"><a name="zh-cn_topic_0000001933105872_p15439101843114"></a><a name="zh-cn_topic_0000001933105872_p15439101843114"></a>DeviceMem对象</p>
    </td>
    </tr>
    </tbody>
    </table>

## 返回值<a name="zh-cn_topic_0000001933105872_section9797mcpsimp"></a>

无。

## 约束说明<a name="zh-cn_topic_0000001933105872_section9800mcpsimp"></a>

无

