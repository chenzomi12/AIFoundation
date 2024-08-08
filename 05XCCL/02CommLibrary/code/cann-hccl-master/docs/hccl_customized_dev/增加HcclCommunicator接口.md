# 增加HcclCommunicator接口 

HcclCommunicator是通信域功能的执行层，在HCCL架构中隶属于框架层。HcclCommunicator与算子类通过三个接口进行交互（参考[增加通信算子Operator](增加通信算子Operator.md)），并进行资源创建（stream、notify、memory、建链等）。

涉及代码文件：

```
src/domain/collective_communication/framework/communicator/impl/hccl_communicator.cc
src/domain/collective_communication/framework/communicator/impl/hccl_communicator.h
```

1. <a name="li1544184665913"></a>在枚举类 HcclCMDType 中为新算子添加一个枚举值。

   HcclCMDType 定义在 hccl\_common.h，每个算子都唯一对应 HcclCMDType 中的一个值。

   枚举值格式：HCCL\_CMD\_XXX

   **注意：** 

    >HCCL\_CMD\_INVALID，HCCL\_CMD\_MAX 和 HCCL\_CMD\_ALL 为特殊值，具有特定作用。
    >HCCL\_CMD\_INVALID 表示无效算子，必须放在第一个，且值等于0；
    >HCCL\_CMD\_MAX 记录了 HcclCMDType 中枚举值的数量，必须放在最后；
    >HCCL\_CMD\_ALL 在某些场景下表示所有算子，建议放在 HCCL\_CMD\_MAX 的前一个位置。

    此外，还需要在 hccl\_impl.h 中的以下 map 成员的默认值中添加新枚举值：

    algType\_

    isAlgoLevel1Default\_

2.  定义新算子的API。

    在 hccl\_communicator.h 中声明新算子的接口。

    ```
    HcclResult MyOperatorOutPlace(args...)
    ```

    其中OutPlace后缀代表单算子模式。

    在 hccl\_communicator.cc 中添加新算子的定义。

    以 ReduceScatter 算子为例：

    ```
    HcclResult ReduceScatterOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op, HcclRtStream stream)
    ```

3. 异常流程处理（可选）

   处理异常流程可以有效避免预期之外的行为，减少错误或提升效率。

   例如检查当前device类型是否支持该算子，检查通信域是否已经初始化等。

   **说明：**源码中的硬件类型体现的是Soc Version，您可以在安装昇腾AI处理器的服务器中执行“npu-smi info”命令查询，查询到的“Chip Name”即为对应的Soc Version。

   以 ReduceScatter 算子为例：

   ```
   HcclResult HcclCommunicator::ReduceScatterOutPlace(const std::string &tag, void *inputPtr, void *outputPtr,
       u64 count, HcclDataType dataType, HcclReduceOp op, HcclRtStream stream)
   {
       ...
   
       // 硬件类型为Atlas 推理系列产品（Ascend 310P处理器）中的加速卡时，不支持ReduceScatter算子
       CHK_PRT_RET(Is310P3Common(), HCCL_ERROR("[HcclCommunicator][ReduceScatterOutPlace]"
           "ReduceScatterOutPlace is not supported"), HCCL_E_NOT_SUPPORT);
   
       // 通信域未初始化，返回报错
       if (!IsAtomicInit()) {
           HCCL_ERROR("[HcclCommunicator][ReduceScatterOutPlace]errNo[0x%016llx] hccl init must be called before"
               " call this function", HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
           return HCCL_E_UNAVAIL;
       }
   
       ...
   }
   ```

4.  添加Debug信息（按需）。

    Hccl提供了若干维测功能，可记录算子运行时的一些信息，用于分析算子行为，有助于问题定位。

    例如：算子统计：在算子执行前后分别调用 StarsCounter 接口，进行头计数和尾计数

    ```
    HcclResult StarsCounter(const HcclDispatcher &dispatcher, Stream &stream, int flag)
    ```

    功能：stars任务计数，用于Debug

    <a name="table10972811151010"></a>
    <table><thead align="left"><tr id="row597216118108"><th class="cellrowborder" valign="top" width="16.31%" id="mcps1.1.5.1.1"><p id="p13972141181016"><a name="p13972141181016"></a><a name="p13972141181016"></a>参数</p>
    </th>
    <th class="cellrowborder" valign="top" width="28.82%" id="mcps1.1.5.1.2"><p id="p79724118103"><a name="p79724118103"></a><a name="p79724118103"></a>类型</p>
    </th>
    <th class="cellrowborder" valign="top" width="13.889999999999999%" id="mcps1.1.5.1.3"><p id="p697281113106"><a name="p697281113106"></a><a name="p697281113106"></a>输入/输出</p>
    </th>
    <th class="cellrowborder" valign="top" width="40.98%" id="mcps1.1.5.1.4"><p id="p1097216118107"><a name="p1097216118107"></a><a name="p1097216118107"></a>说明</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row16972201131014"><td class="cellrowborder" valign="top" width="16.31%" headers="mcps1.1.5.1.1 "><p id="p49722011111016"><a name="p49722011111016"></a><a name="p49722011111016"></a>dispatcher</p>
    </td>
    <td class="cellrowborder" valign="top" width="28.82%" headers="mcps1.1.5.1.2 "><p id="p497281114108"><a name="p497281114108"></a><a name="p497281114108"></a>const HcclDispatcher &amp;</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.889999999999999%" headers="mcps1.1.5.1.3 "><p id="p20972111111109"><a name="p20972111111109"></a><a name="p20972111111109"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="40.98%" headers="mcps1.1.5.1.4 "><p id="p119727115101"><a name="p119727115101"></a><a name="p119727115101"></a>调度器，一般传入成员dispatcher_即可</p>
    </td>
    </tr>
    <tr id="row1497241191012"><td class="cellrowborder" valign="top" width="16.31%" headers="mcps1.1.5.1.1 "><p id="p1097231113109"><a name="p1097231113109"></a><a name="p1097231113109"></a>stream</p>
    </td>
    <td class="cellrowborder" valign="top" width="28.82%" headers="mcps1.1.5.1.2 "><p id="p7943917191114"><a name="p7943917191114"></a><a name="p7943917191114"></a>Stream &amp;</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.889999999999999%" headers="mcps1.1.5.1.3 "><p id="p139722011141017"><a name="p139722011141017"></a><a name="p139722011141017"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="40.98%" headers="mcps1.1.5.1.4 "><p id="p09731011141014"><a name="p09731011141014"></a><a name="p09731011141014"></a>算子的主流</p>
    </td>
    </tr>
    <tr id="row0866183715103"><td class="cellrowborder" valign="top" width="16.31%" headers="mcps1.1.5.1.1 "><p id="p386643721018"><a name="p386643721018"></a><a name="p386643721018"></a>flag</p>
    </td>
    <td class="cellrowborder" valign="top" width="28.82%" headers="mcps1.1.5.1.2 "><p id="p168668372108"><a name="p168668372108"></a><a name="p168668372108"></a>int</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.889999999999999%" headers="mcps1.1.5.1.3 "><p id="p3866173761013"><a name="p3866173761013"></a><a name="p3866173761013"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="40.98%" headers="mcps1.1.5.1.4 "><p id="p1086614376105"><a name="p1086614376105"></a><a name="p1086614376105"></a>0代表头，1代表尾</p>
    </td>
    </tr>
    </tbody>
    </table>

    其中，HcclDispacher 为调度器类，用于封装内存拷贝操作；Stream 为流类。

    返回值：Hccl执行结果，成功时返回HCCL\_SUCCESS，异常时返回相应的错误类型。

    以 ReduceScatter 算子为例：

    ```
    HcclResult HcclCommunicator::ReduceScatterOutPlace(const std::string &tag, void *inputPtr, void *outputPtr,
        u64 count, HcclDataType dataType, HcclReduceOp op, HcclRtStream stream)
    {
        ...
        // 头计数任务
        CHK_RET(StarsCounter(dispatcher_, streamObj, HEAD));
        // 调用算子执行接口
        ...
        // 尾计数任务
        CHK_RET(StarsCounter(dispatcher_, streamObj, TAIL));
        return HCCL_SUCCESS;
    }
    ```

5.  调用算子执行接口。

    通过调用ExecOp接口执行算子流程，包含通过opType获取算子实例，算法选择，根据资源计算结果进行资源创建，和执行算法编排。

    ```
    HcclResult ExecOp(HcclCMDType opType, const OpParam &opParam)
    ```

    参数含义如下表所示。

    **表 1**  ExecOp接口参数说明

    <a name="table827101275518"></a>
    <table><thead align="left"><tr id="row429121265517"><th class="cellrowborder" valign="top" width="13.68%" id="mcps1.2.5.1.1"><p id="p1329121214558"><a name="p1329121214558"></a><a name="p1329121214558"></a>参数</p>
    </th>
    <th class="cellrowborder" valign="top" width="23.080000000000002%" id="mcps1.2.5.1.2"><p id="p146768713238"><a name="p146768713238"></a><a name="p146768713238"></a>类型</p>
    </th>
    <th class="cellrowborder" valign="top" width="12.5%" id="mcps1.2.5.1.3"><p id="p10230141454318"><a name="p10230141454318"></a><a name="p10230141454318"></a>输入/输出</p>
    </th>
    <th class="cellrowborder" valign="top" width="50.739999999999995%" id="mcps1.2.5.1.4"><p id="p83121275519"><a name="p83121275519"></a><a name="p83121275519"></a>说明</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row1131131265511"><td class="cellrowborder" valign="top" width="13.68%" headers="mcps1.2.5.1.1 "><p id="p191061137121320"><a name="p191061137121320"></a><a name="p191061137121320"></a>opType</p>
    </td>
    <td class="cellrowborder" valign="top" width="23.080000000000002%" headers="mcps1.2.5.1.2 "><p id="p46778720237"><a name="p46778720237"></a><a name="p46778720237"></a>HcclCMDType</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.5%" headers="mcps1.2.5.1.3 "><p id="p16105133721316"><a name="p16105133721316"></a><a name="p16105133721316"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="50.739999999999995%" headers="mcps1.2.5.1.4 "><p id="p10105143741311"><a name="p10105143741311"></a><a name="p10105143741311"></a>算子类型</p>
    </td>
    </tr>
    <tr id="row18118485118"><td class="cellrowborder" valign="top" width="13.68%" headers="mcps1.2.5.1.1 "><p id="p11104837101311"><a name="p11104837101311"></a><a name="p11104837101311"></a>opParam</p>
    </td>
    <td class="cellrowborder" valign="top" width="23.080000000000002%" headers="mcps1.2.5.1.2 "><p id="p20677197162311"><a name="p20677197162311"></a><a name="p20677197162311"></a>const OpParam &amp;</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.5%" headers="mcps1.2.5.1.3 "><p id="p8103173701314"><a name="p8103173701314"></a><a name="p8103173701314"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="50.739999999999995%" headers="mcps1.2.5.1.4 "><p id="p151038375137"><a name="p151038375137"></a><a name="p151038375137"></a>算子的入参，包括输入输出指针、数据量等信息。</p>
    </td>
    </tr>
    </tbody>
    </table>

    返回值：Hccl执行结果，成功时返回HCCL\_SUCCESS，异常时返回相应的错误类型。

    其中OpParam类型包含的成员如下表所示，包含了所有算子可能用到的入参，构造OpParam时只需为当前算子实际用到的成员赋值即可。

    **表 2**  OpParam成员说明

    <a name="table15958201412115"></a>
    <table><thead align="left"><tr id="row18958414101111"><th class="cellrowborder" colspan="3" valign="top" id="mcps1.2.6.1.1"><p id="p9958614111110"><a name="p9958614111110"></a><a name="p9958614111110"></a>成员</p>
    </th>
    <th class="cellrowborder" valign="top" id="mcps1.2.6.1.2"><p id="p89581614131117"><a name="p89581614131117"></a><a name="p89581614131117"></a>类型</p>
    </th>
    <th class="cellrowborder" valign="top" id="mcps1.2.6.1.3"><p id="p17958114121116"><a name="p17958114121116"></a><a name="p17958114121116"></a>说明</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row159588144119"><td class="cellrowborder" colspan="3" valign="top" headers="mcps1.2.6.1.1 "><p id="p116754327116"><a name="p116754327116"></a><a name="p116754327116"></a>tag</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p1995961417119"><a name="p1995961417119"></a><a name="p1995961417119"></a>std::string</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p1295918143115"><a name="p1295918143115"></a><a name="p1295918143115"></a>算子在通信域中的标记，用于DFX方面。</p>
    </td>
    </tr>
    <tr id="row11959101412112"><td class="cellrowborder" colspan="3" valign="top" headers="mcps1.2.6.1.1 "><p id="p1227315321312"><a name="p1227315321312"></a><a name="p1227315321312"></a>stream</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p9959121413116"><a name="p9959121413116"></a><a name="p9959121413116"></a>Stream</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p19959191421113"><a name="p19959191421113"></a><a name="p19959191421113"></a>算子执行的主流</p>
    </td>
    </tr>
    <tr id="row17381182171416"><td class="cellrowborder" colspan="3" valign="top" headers="mcps1.2.6.1.1 "><p id="p8613162913142"><a name="p8613162913142"></a><a name="p8613162913142"></a>inputPtr</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p1651112335146"><a name="p1651112335146"></a><a name="p1651112335146"></a>void*</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p1538152171410"><a name="p1538152171410"></a><a name="p1538152171410"></a>输入地址指针</p>
    </td>
    </tr>
    <tr id="row12744122511144"><td class="cellrowborder" colspan="3" valign="top" headers="mcps1.2.6.1.1 "><p id="p6990165012619"><a name="p6990165012619"></a><a name="p6990165012619"></a>inputSize</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p485144551714"><a name="p485144551714"></a><a name="p485144551714"></a>u64</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p4408162132312"><a name="p4408162132312"></a><a name="p4408162132312"></a>输入地址大小</p>
    </td>
    </tr>
    <tr id="row7532371880"><td class="cellrowborder" colspan="3" valign="top" headers="mcps1.2.6.1.1 "><p id="p16667501186"><a name="p16667501186"></a><a name="p16667501186"></a>outputPtr</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p06669501686"><a name="p06669501686"></a><a name="p06669501686"></a>void*</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p15531937283"><a name="p15531937283"></a><a name="p15531937283"></a>输出地址指针</p>
    </td>
    </tr>
    <tr id="row85318373819"><td class="cellrowborder" colspan="3" valign="top" headers="mcps1.2.6.1.1 "><p id="p866618501687"><a name="p866618501687"></a><a name="p866618501687"></a>outputSize</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p1366615501189"><a name="p1366615501189"></a><a name="p1366615501189"></a>u64</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p175418378818"><a name="p175418378818"></a><a name="p175418378818"></a>输出地址大小</p>
    </td>
    </tr>
    <tr id="row1540379810"><td class="cellrowborder" colspan="3" valign="top" headers="mcps1.2.6.1.1 "><p id="p16534113913104"><a name="p16534113913104"></a><a name="p16534113913104"></a>reduceType</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p7541737482"><a name="p7541737482"></a><a name="p7541737482"></a>HcclReduceOp</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p3546374814"><a name="p3546374814"></a><a name="p3546374814"></a>消减运算类型，如求和，乘积，最大值，最小值</p>
    </td>
    </tr>
    <tr id="row155414373819"><td class="cellrowborder" colspan="3" valign="top" headers="mcps1.2.6.1.1 "><p id="p1534133981020"><a name="p1534133981020"></a><a name="p1534133981020"></a>syncMode</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p13548371282"><a name="p13548371282"></a><a name="p13548371282"></a>SyncMode</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p19542378818"><a name="p19542378818"></a><a name="p19542378818"></a>notifywait超时类型</p>
    </td>
    </tr>
    <tr id="row951423531018"><td class="cellrowborder" colspan="3" valign="top" headers="mcps1.2.6.1.1 "><p id="p1553473919104"><a name="p1553473919104"></a><a name="p1553473919104"></a>root</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p1451423521010"><a name="p1451423521010"></a><a name="p1451423521010"></a>RankId</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p65147351102"><a name="p65147351102"></a><a name="p65147351102"></a>根节点rankid</p>
    </td>
    </tr>
    <tr id="row351413541016"><td class="cellrowborder" colspan="3" valign="top" headers="mcps1.2.6.1.1 "><p id="p85341739201017"><a name="p85341739201017"></a><a name="p85341739201017"></a>dstRank</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p1451433517109"><a name="p1451433517109"></a><a name="p1451433517109"></a>RankId</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p185141935171010"><a name="p185141935171010"></a><a name="p185141935171010"></a>目的rankid</p>
    </td>
    </tr>
    <tr id="row175143354101"><td class="cellrowborder" colspan="3" valign="top" headers="mcps1.2.6.1.1 "><p id="p13534133913104"><a name="p13534133913104"></a><a name="p13534133913104"></a>srcRank</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p65151735191014"><a name="p65151735191014"></a><a name="p65151735191014"></a>RankId</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p7515133571017"><a name="p7515133571017"></a><a name="p7515133571017"></a>源rankid</p>
    </td>
    </tr>
    <tr id="row1651563551013"><td class="cellrowborder" colspan="3" valign="top" headers="mcps1.2.6.1.1 "><p id="p9255952201011"><a name="p9255952201011"></a><a name="p9255952201011"></a>opBaseAtraceInfo</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p1515193591015"><a name="p1515193591015"></a><a name="p1515193591015"></a>HcclOpBaseAtraceInfo*</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p251511358102"><a name="p251511358102"></a><a name="p251511358102"></a>用于DFX</p>
    </td>
    </tr>
    <tr id="row115152351106"><td class="cellrowborder" rowspan="11" align="center" valign="top" width="8.85088508850885%" headers="mcps1.2.6.1.1 "><p id="p84471432317"><a name="p84471432317"></a><a name="p84471432317"></a></p>
    <p id="p1166418333118"><a name="p1166418333118"></a><a name="p1166418333118"></a></p>
    <p id="p16815133183117"><a name="p16815133183117"></a><a name="p16815133183117"></a></p>
    <p id="p29760319317"><a name="p29760319317"></a><a name="p29760319317"></a></p>
    <p id="p0129142319"><a name="p0129142319"></a><a name="p0129142319"></a></p>
    <p id="p830615413318"><a name="p830615413318"></a><a name="p830615413318"></a></p>
    <p id="p105385417319"><a name="p105385417319"></a><a name="p105385417319"></a></p>
    <p id="p2312114418413"><a name="p2312114418413"></a><a name="p2312114418413"></a>union</p>
    </td>
    <td class="cellrowborder" rowspan="2" valign="top" width="16.781678167816782%" headers="mcps1.2.6.1.1 "><p id="p1522019072812"><a name="p1522019072812"></a><a name="p1522019072812"></a>DataDes</p>
    </td>
    <td class="cellrowborder" valign="top" width="14.031403140314033%" headers="mcps1.2.6.1.1 "><p id="p21711227161513"><a name="p21711227161513"></a><a name="p21711227161513"></a>count</p>
    </td>
    <td class="cellrowborder" valign="top" width="27.412741274127413%" headers="mcps1.2.6.1.2 "><p id="p17515735111011"><a name="p17515735111011"></a><a name="p17515735111011"></a>u64</p>
    </td>
    <td class="cellrowborder" valign="top" width="32.92329232923292%" headers="mcps1.2.6.1.3 "><p id="p7515335171015"><a name="p7515335171015"></a><a name="p7515335171015"></a>输入数据个数</p>
    </td>
    </tr>
    <tr id="row3651174412147"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p1865264481417"><a name="p1865264481417"></a><a name="p1865264481417"></a>dataType</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p15652204431412"><a name="p15652204431412"></a><a name="p15652204431412"></a>HcclDataType</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p2652344131417"><a name="p2652344131417"></a><a name="p2652344131417"></a>输入数据类型，如int8, in16, in32, float16, fload32等</p>
    </td>
    </tr>
    <tr id="row16515123521017"><td class="cellrowborder" rowspan="7" valign="top" headers="mcps1.2.6.1.1 "><p id="p1722010132810"><a name="p1722010132810"></a><a name="p1722010132810"></a>All2AllDataDes</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p121151620151617"><a name="p121151620151617"></a><a name="p121151620151617"></a>sendType</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p13463926115313"><a name="p13463926115313"></a><a name="p13463926115313"></a>HcclDataType</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p196324110192"><a name="p196324110192"></a><a name="p196324110192"></a>发送数据类型</p>
    </td>
    </tr>
    <tr id="row13737122515192"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p773792521914"><a name="p773792521914"></a><a name="p773792521914"></a>recvType</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p1373772513191"><a name="p1373772513191"></a><a name="p1373772513191"></a>HcclDataType</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p87378258192"><a name="p87378258192"></a><a name="p87378258192"></a>接收数据类型</p>
    </td>
    </tr>
    <tr id="row520163521919"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p142103591917"><a name="p142103591917"></a><a name="p142103591917"></a>sendCounts</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p62103531912"><a name="p62103531912"></a><a name="p62103531912"></a>void*</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p82173541915"><a name="p82173541915"></a><a name="p82173541915"></a>发送数据个数</p>
    </td>
    </tr>
    <tr id="row142163531919"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p1214355198"><a name="p1214355198"></a><a name="p1214355198"></a>recvCounts</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p5211735191914"><a name="p5211735191914"></a><a name="p5211735191914"></a>void*</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p112113353195"><a name="p112113353195"></a><a name="p112113353195"></a>接收数据个数</p>
    </td>
    </tr>
    <tr id="row17830174161917"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p48316416199"><a name="p48316416199"></a><a name="p48316416199"></a>sdispls</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p5831204181917"><a name="p5831204181917"></a><a name="p5831204181917"></a>void*</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p1683104116194"><a name="p1683104116194"></a><a name="p1683104116194"></a>表示发送偏移量的uint64数组</p>
    </td>
    </tr>
    <tr id="row1983194113192"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p78313411196"><a name="p78313411196"></a><a name="p78313411196"></a>rdispls</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p6831241111917"><a name="p6831241111917"></a><a name="p6831241111917"></a>void*</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p48310411196"><a name="p48310411196"></a><a name="p48310411196"></a>表示接收偏移量的uint64数组</p>
    </td>
    </tr>
    <tr id="row1783114111193"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p17831124120194"><a name="p17831124120194"></a><a name="p17831124120194"></a>sendCountMatrix</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p2831341101912"><a name="p2831341101912"></a><a name="p2831341101912"></a>void*</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p183114415194"><a name="p183114415194"></a><a name="p183114415194"></a>代表每张卡要发给别人的count的信息</p>
    </td>
    </tr>
    <tr id="row165151235131018"><td class="cellrowborder" rowspan="2" valign="top" headers="mcps1.2.6.1.1 "><p id="p11579121311510"><a name="p11579121311510"></a><a name="p11579121311510"></a>BatchSendRecvDataDes</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p2220130122818"><a name="p2220130122818"></a><a name="p2220130122818"></a>orderedList</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p19916526155319"><a name="p19916526155319"></a><a name="p19916526155319"></a>HcclSendRecvItem**</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p1747095991811"><a name="p1747095991811"></a><a name="p1747095991811"></a>发送和接收的item列表</p>
    </td>
    </tr>
    <tr id="row28241920181813"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p1382592017184"><a name="p1382592017184"></a><a name="p1382592017184"></a>itemNum</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p8825720111818"><a name="p8825720111818"></a><a name="p8825720111818"></a>u32</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p17825182012189"><a name="p17825182012189"></a><a name="p17825182012189"></a>item数量</p>
    </td>
    </tr>
    <tr id="row19515163591012"><td class="cellrowborder" colspan="3" valign="top" headers="mcps1.2.6.1.1 "><p id="p185151735111015"><a name="p185151735111015"></a><a name="p185151735111015"></a>opType</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p651516356109"><a name="p651516356109"></a><a name="p651516356109"></a>HcclCMDType</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p12515193541010"><a name="p12515193541010"></a><a name="p12515193541010"></a>算子类型</p>
    </td>
    </tr>
    </tbody>
    </table>

    **注意：** 
    >-   对于一个算子，DataDes，All2AllDataDes，BatchSendRecvDataDes只会生效其一，所以为union类型。
    >-   若自定义算子使用了OpParam未包含的入参，需在OpParam的定义中对应增加新的成员。
    >-   调用ExecOp时，opType需要传入步骤[1](#li1544184665913)新增的枚举值，opParam需要用算子入参构造。

    以 ReduceScatter 算子为例：

    ```
    HcclResult HcclCommunicator::ReduceScatterOutPlace(const std::string &tag, void *inputPtr, void *outputPtr,
        u64 count, HcclDataType dataType, HcclReduceOp op, HcclRtStream stream)
    {
        ...
    
        u32 perDataSize = SIZE_TABLE[dataType];
        // 用算子入参构造OpParam
        OpParam opParam;
        opParam.tag = tag;
        opParam.inputPtr = inputPtr;
        opParam.inputSize = userRankSize_ * count * perDataSize;
        opParam.outputPtr = outputPtr;
        opParam.outputSize = count * perDataSize;
        opParam.DataDes.count = count;
        opParam.DataDes.dataType = dataType;
        opParam.reduceType = op;
        opParam.stream = streamObj;
        // 调用算子执行接口
        CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, opParam));
    
        ...
    }
    ```

