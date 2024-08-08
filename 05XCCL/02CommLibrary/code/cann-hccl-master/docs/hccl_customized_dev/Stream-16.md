# Stream 

## 函数原型<a name="zh-cn_topic_0000001963694613_section103mcpsimp"></a>

```
// Stream构造函数
Stream()

// 基于类型构造Stream，是stream owner
Stream(const StreamType streamType, bool isMainStream = false)

// 使用rtStream构造Stream，不是stream owner
Stream(const rtStream_t rtStream, bool isMainStream = true)

// 基于HcclComStreamInfo信息构造stream，不是stream owner
Stream(const HcclComStreamInfo &streamInfo, bool isMainStream = false)
```

## 函数功能<a name="zh-cn_topic_0000001963694613_section106mcpsimp"></a>

Stream构造函数。

## 参数说明<a name="zh-cn_topic_0000001963694613_section109mcpsimp"></a>

-   基于基类构造Stream

    **表 1**  参数说明

    <a name="zh-cn_topic_0000001963694613_table127mcpsimp"></a>
    <table><thead align="left"><tr id="zh-cn_topic_0000001963694613_row134mcpsimp"><th class="cellrowborder" valign="top" width="28.71%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001963694613_p136mcpsimp"><a name="zh-cn_topic_0000001963694613_p136mcpsimp"></a><a name="zh-cn_topic_0000001963694613_p136mcpsimp"></a>参数</p>
    </th>
    <th class="cellrowborder" valign="top" width="13.86%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001963694613_p138mcpsimp"><a name="zh-cn_topic_0000001963694613_p138mcpsimp"></a><a name="zh-cn_topic_0000001963694613_p138mcpsimp"></a>输入/输出</p>
    </th>
    <th class="cellrowborder" valign="top" width="57.43000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001963694613_p140mcpsimp"><a name="zh-cn_topic_0000001963694613_p140mcpsimp"></a><a name="zh-cn_topic_0000001963694613_p140mcpsimp"></a>说明</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0000001963694613_row142mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001963694613_p144mcpsimp"><a name="zh-cn_topic_0000001963694613_p144mcpsimp"></a><a name="zh-cn_topic_0000001963694613_p144mcpsimp"></a>const StreamType streamType</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001963694613_p146mcpsimp"><a name="zh-cn_topic_0000001963694613_p146mcpsimp"></a><a name="zh-cn_topic_0000001963694613_p146mcpsimp"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001963694613_p148mcpsimp"><a name="zh-cn_topic_0000001963694613_p148mcpsimp"></a><a name="zh-cn_topic_0000001963694613_p148mcpsimp"></a>Stream类型</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000001963694613_row149mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001963694613_p151mcpsimp"><a name="zh-cn_topic_0000001963694613_p151mcpsimp"></a><a name="zh-cn_topic_0000001963694613_p151mcpsimp"></a>bool isMainStream</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001963694613_p153mcpsimp"><a name="zh-cn_topic_0000001963694613_p153mcpsimp"></a><a name="zh-cn_topic_0000001963694613_p153mcpsimp"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001963694613_p155mcpsimp"><a name="zh-cn_topic_0000001963694613_p155mcpsimp"></a><a name="zh-cn_topic_0000001963694613_p155mcpsimp"></a>是否是主流</p>
    </td>
    </tr>
    </tbody>
    </table>

-   使用rtStream构造Stream

    **表 2**  参数说明

    <a name="zh-cn_topic_0000001963694613_table171mcpsimp"></a>
    <table><thead align="left"><tr id="zh-cn_topic_0000001963694613_row178mcpsimp"><th class="cellrowborder" valign="top" width="28.71%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001963694613_p180mcpsimp"><a name="zh-cn_topic_0000001963694613_p180mcpsimp"></a><a name="zh-cn_topic_0000001963694613_p180mcpsimp"></a>参数</p>
    </th>
    <th class="cellrowborder" valign="top" width="13.86%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001963694613_p182mcpsimp"><a name="zh-cn_topic_0000001963694613_p182mcpsimp"></a><a name="zh-cn_topic_0000001963694613_p182mcpsimp"></a>输入/输出</p>
    </th>
    <th class="cellrowborder" valign="top" width="57.43000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001963694613_p184mcpsimp"><a name="zh-cn_topic_0000001963694613_p184mcpsimp"></a><a name="zh-cn_topic_0000001963694613_p184mcpsimp"></a>说明</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0000001963694613_row186mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001963694613_p188mcpsimp"><a name="zh-cn_topic_0000001963694613_p188mcpsimp"></a><a name="zh-cn_topic_0000001963694613_p188mcpsimp"></a>const rtStream_t rtStream</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001963694613_p190mcpsimp"><a name="zh-cn_topic_0000001963694613_p190mcpsimp"></a><a name="zh-cn_topic_0000001963694613_p190mcpsimp"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001963694613_p192mcpsimp"><a name="zh-cn_topic_0000001963694613_p192mcpsimp"></a><a name="zh-cn_topic_0000001963694613_p192mcpsimp"></a>Rt stream ptr</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000001963694613_row193mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001963694613_p195mcpsimp"><a name="zh-cn_topic_0000001963694613_p195mcpsimp"></a><a name="zh-cn_topic_0000001963694613_p195mcpsimp"></a>bool isMainStream</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001963694613_p197mcpsimp"><a name="zh-cn_topic_0000001963694613_p197mcpsimp"></a><a name="zh-cn_topic_0000001963694613_p197mcpsimp"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001963694613_p199mcpsimp"><a name="zh-cn_topic_0000001963694613_p199mcpsimp"></a><a name="zh-cn_topic_0000001963694613_p199mcpsimp"></a>是否是主流</p>
    </td>
    </tr>
    </tbody>
    </table>

-   基于HcclComStreamInfo信息构造stream

    **表 3**  参数说明

    <a name="zh-cn_topic_0000001963694613_table215mcpsimp"></a>
    <table><thead align="left"><tr id="zh-cn_topic_0000001963694613_row222mcpsimp"><th class="cellrowborder" valign="top" width="28.71%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001963694613_p224mcpsimp"><a name="zh-cn_topic_0000001963694613_p224mcpsimp"></a><a name="zh-cn_topic_0000001963694613_p224mcpsimp"></a>参数</p>
    </th>
    <th class="cellrowborder" valign="top" width="13.86%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001963694613_p226mcpsimp"><a name="zh-cn_topic_0000001963694613_p226mcpsimp"></a><a name="zh-cn_topic_0000001963694613_p226mcpsimp"></a>输入/输出</p>
    </th>
    <th class="cellrowborder" valign="top" width="57.43000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001963694613_p228mcpsimp"><a name="zh-cn_topic_0000001963694613_p228mcpsimp"></a><a name="zh-cn_topic_0000001963694613_p228mcpsimp"></a>说明</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0000001963694613_row230mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001963694613_p232mcpsimp"><a name="zh-cn_topic_0000001963694613_p232mcpsimp"></a><a name="zh-cn_topic_0000001963694613_p232mcpsimp"></a>const HcclComStreamInfo &amp;streamInfo</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001963694613_p234mcpsimp"><a name="zh-cn_topic_0000001963694613_p234mcpsimp"></a><a name="zh-cn_topic_0000001963694613_p234mcpsimp"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001963694613_p236mcpsimp"><a name="zh-cn_topic_0000001963694613_p236mcpsimp"></a><a name="zh-cn_topic_0000001963694613_p236mcpsimp"></a>Stream info</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000001963694613_row237mcpsimp"><td class="cellrowborder" valign="top" width="28.71%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001963694613_p239mcpsimp"><a name="zh-cn_topic_0000001963694613_p239mcpsimp"></a><a name="zh-cn_topic_0000001963694613_p239mcpsimp"></a>bool isMainStream</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.86%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001963694613_p241mcpsimp"><a name="zh-cn_topic_0000001963694613_p241mcpsimp"></a><a name="zh-cn_topic_0000001963694613_p241mcpsimp"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="57.43000000000001%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001963694613_p243mcpsimp"><a name="zh-cn_topic_0000001963694613_p243mcpsimp"></a><a name="zh-cn_topic_0000001963694613_p243mcpsimp"></a>是否是主流</p>
    </td>
    </tr>
    </tbody>
    </table>

## 返回值<a name="zh-cn_topic_0000001963694613_section112mcpsimp"></a>

无。

## 约束说明<a name="zh-cn_topic_0000001963694613_section115mcpsimp"></a>

无。

