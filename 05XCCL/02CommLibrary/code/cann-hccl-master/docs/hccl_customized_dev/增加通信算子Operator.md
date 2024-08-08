# 增加通信算子Operator 

算子类在HCCL架构中隶属于算法层，其通过SelectAlg，CalcResRequest和Orchestrate接口与框架层进行交互，分别实现算法选择，资源计算和算法编排功能。

1.  在“src/domain/collective\_communication/algorithm/impl/operator”路径中添加新算子的头文件（\*.h）与实现文件（\*.cc）。

    新增加文件请遵循如下命名规范：

    xxx\_operator.cc

    xxx\_operator.h

2.  在 xxx\_operator.h 中声明一个新的算子类 xxxOperator，继承自算子基类 CollAlgOperator。
3.  重写算法选择接口。

    ```
    virtual HcclResult SelectAlg(const std::string& tag,
            const OpParam& param, std::string& algName, std::string& newTag);
    ```

    **表 1**  SelectAlg接口参数说明

    <a name="table4669146195918"></a>
    <table><thead align="left"><tr id="row116695619592"><th class="cellrowborder" valign="top" width="20.36%" id="mcps1.2.5.1.1"><p id="p1166986105917"><a name="p1166986105917"></a><a name="p1166986105917"></a>参数</p>
    </th>
    <th class="cellrowborder" valign="top" width="15.68%" id="mcps1.2.5.1.2"><p id="p196694615593"><a name="p196694615593"></a><a name="p196694615593"></a>类型</p>
    </th>
    <th class="cellrowborder" valign="top" width="13.22%" id="mcps1.2.5.1.3"><p id="p1466913612597"><a name="p1466913612597"></a><a name="p1466913612597"></a>输入/输出</p>
    </th>
    <th class="cellrowborder" valign="top" width="50.739999999999995%" id="mcps1.2.5.1.4"><p id="p5669106175920"><a name="p5669106175920"></a><a name="p5669106175920"></a>说明</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row2066919645918"><td class="cellrowborder" valign="top" width="20.36%" headers="mcps1.2.5.1.1 "><p id="p1666913665910"><a name="p1666913665910"></a><a name="p1666913665910"></a>tag</p>
    </td>
    <td class="cellrowborder" valign="top" width="15.68%" headers="mcps1.2.5.1.2 "><p id="p166920655913"><a name="p166920655913"></a><a name="p166920655913"></a>const std::string &amp;</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.22%" headers="mcps1.2.5.1.3 "><p id="p1666917605910"><a name="p1666917605910"></a><a name="p1666917605910"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="50.739999999999995%" headers="mcps1.2.5.1.4 "><p id="p19669196105919"><a name="p19669196105919"></a><a name="p19669196105919"></a>算子在通信域中的标记，用于DFX方面。</p>
    </td>
    </tr>
    <tr id="row667076145918"><td class="cellrowborder" valign="top" width="20.36%" headers="mcps1.2.5.1.1 "><p id="p1167013616596"><a name="p1167013616596"></a><a name="p1167013616596"></a>param</p>
    </td>
    <td class="cellrowborder" valign="top" width="15.68%" headers="mcps1.2.5.1.2 "><p id="p167013616596"><a name="p167013616596"></a><a name="p167013616596"></a>const OpParam &amp;</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.22%" headers="mcps1.2.5.1.3 "><p id="p367012615910"><a name="p367012615910"></a><a name="p367012615910"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="50.739999999999995%" headers="mcps1.2.5.1.4 "><p id="p66702618596"><a name="p66702618596"></a><a name="p66702618596"></a>算子的入参，包括输入输出指针、数据量等信息。</p>
    </td>
    </tr>
    <tr id="row7670136165910"><td class="cellrowborder" valign="top" width="20.36%" headers="mcps1.2.5.1.1 "><p id="p1167016685915"><a name="p1167016685915"></a><a name="p1167016685915"></a>algName</p>
    </td>
    <td class="cellrowborder" valign="top" width="15.68%" headers="mcps1.2.5.1.2 "><p id="p18670116175916"><a name="p18670116175916"></a><a name="p18670116175916"></a>std::string &amp;</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.22%" headers="mcps1.2.5.1.3 "><p id="p1667076155919"><a name="p1667076155919"></a><a name="p1667076155919"></a>输出</p>
    </td>
    <td class="cellrowborder" valign="top" width="50.739999999999995%" headers="mcps1.2.5.1.4 "><p id="p17670196145910"><a name="p17670196145910"></a><a name="p17670196145910"></a>算法选择返回的算法名字，即Executor注册的名字。</p>
    </td>
    </tr>
    <tr id="row15670196195911"><td class="cellrowborder" valign="top" width="20.36%" headers="mcps1.2.5.1.1 "><p id="p1567066135915"><a name="p1567066135915"></a><a name="p1567066135915"></a>newTag</p>
    </td>
    <td class="cellrowborder" valign="top" width="15.68%" headers="mcps1.2.5.1.2 "><p id="p16670186185919"><a name="p16670186185919"></a><a name="p16670186185919"></a>std::string &amp;</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.22%" headers="mcps1.2.5.1.3 "><p id="p16700675917"><a name="p16700675917"></a><a name="p16700675917"></a>输出</p>
    </td>
    <td class="cellrowborder" valign="top" width="50.739999999999995%" headers="mcps1.2.5.1.4 "><p id="p12670156155919"><a name="p12670156155919"></a><a name="p12670156155919"></a>一个标记字符串，用于在框架测缓存Executor资源，避免资源多次申请。</p>
    </td>
    </tr>
    </tbody>
    </table>

    返回值：Hccl执行结果，成功时返回HCCL\_SUCCESS，异常时返回相应的错误类型。

    SelectAlg 的典型流程是：

    1.  针对不同的device类型，分别设计相应的算法选择策略（根据拓扑类型，卡数，工作流模式等）。当然，也可以根据实际情况简化此部分逻辑，比如只有一种算法时，直接返回算法名称（算法名称注册参考[通信算法开发](通信算法开发.md)）。
    2.  为 tag 增加合理的后缀，构造 newTag 返回，框架层会对拥有相同 newTag 的算子执行资源复用。

4.  重写资源计算和算法编排接口（可选）。

     **注意：** 
    >在算子基类 CollAlgOperator 中，CalResRequest 和 Orchestrate 已经被实现为调用 CollExecutorBase 类（见[增加通信算法Executor](增加通信算法Executor.md)）的同名接口，即主要功能是在 CollExecutorBase 中实现。因此在实际开发中，对算子的 CalResRequest 和 Orchestrate 进行重写并不常见，推荐在 CollExecutorBase 的派生类中重写对应的接口，而不是直接在算子类中实现。

    ```
    virtual HcclResult CalcResRequest(const std::string& algName,
            const OpParam& param, AlgResourceRequest& resourceRequest);
    ```

    **表 2**  CalcResRequest接口参数说明

    <a name="table1767221915910"></a>
    <table><thead align="left"><tr id="row1967261912599"><th class="cellrowborder" valign="top" width="16.81%" id="mcps1.2.5.1.1"><p id="p1672019115910"><a name="p1672019115910"></a><a name="p1672019115910"></a>参数</p>
    </th>
    <th class="cellrowborder" valign="top" width="17.21%" id="mcps1.2.5.1.2"><p id="p136724195599"><a name="p136724195599"></a><a name="p136724195599"></a>类型</p>
    </th>
    <th class="cellrowborder" valign="top" width="12.629999999999999%" id="mcps1.2.5.1.3"><p id="p1267213195595"><a name="p1267213195595"></a><a name="p1267213195595"></a>输入/输出</p>
    </th>
    <th class="cellrowborder" valign="top" width="53.349999999999994%" id="mcps1.2.5.1.4"><p id="p15672161945919"><a name="p15672161945919"></a><a name="p15672161945919"></a>说明</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row4672151914594"><td class="cellrowborder" valign="top" width="16.81%" headers="mcps1.2.5.1.1 "><p id="p7672161945912"><a name="p7672161945912"></a><a name="p7672161945912"></a>algName</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.21%" headers="mcps1.2.5.1.2 "><p id="p067216194596"><a name="p067216194596"></a><a name="p067216194596"></a>const std::string &amp;</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.629999999999999%" headers="mcps1.2.5.1.3 "><p id="p13672171918598"><a name="p13672171918598"></a><a name="p13672171918598"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="53.349999999999994%" headers="mcps1.2.5.1.4 "><p id="p8672319195911"><a name="p8672319195911"></a><a name="p8672319195911"></a>算法名称</p>
    </td>
    </tr>
    <tr id="row1167391910594"><td class="cellrowborder" valign="top" width="16.81%" headers="mcps1.2.5.1.1 "><p id="p2673519205920"><a name="p2673519205920"></a><a name="p2673519205920"></a>param</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.21%" headers="mcps1.2.5.1.2 "><p id="p10673141985918"><a name="p10673141985918"></a><a name="p10673141985918"></a>const OpParam &amp;</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.629999999999999%" headers="mcps1.2.5.1.3 "><p id="p1767391912594"><a name="p1767391912594"></a><a name="p1767391912594"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="53.349999999999994%" headers="mcps1.2.5.1.4 "><p id="p2673219175912"><a name="p2673219175912"></a><a name="p2673219175912"></a>算子的入参，包括输入输出指针、数据量等信息。</p>
    </td>
    </tr>
    <tr id="row2673191912590"><td class="cellrowborder" valign="top" width="16.81%" headers="mcps1.2.5.1.1 "><p id="p10673819185913"><a name="p10673819185913"></a><a name="p10673819185913"></a>resourceRequest</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.21%" headers="mcps1.2.5.1.2 "><p id="p136731319195916"><a name="p136731319195916"></a><a name="p136731319195916"></a>AlgResourceRequest &amp;</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.629999999999999%" headers="mcps1.2.5.1.3 "><p id="p1967314196591"><a name="p1967314196591"></a><a name="p1967314196591"></a>输出</p>
    </td>
    <td class="cellrowborder" valign="top" width="53.349999999999994%" headers="mcps1.2.5.1.4 "><p id="p6673319195914"><a name="p6673319195914"></a><a name="p6673319195914"></a>Executor执行需要的资源诉求，包含从流数量、主从流同步需要的notify数量、workspace内存、建链诉求等信息。</p>
    </td>
    </tr>
    </tbody>
    </table>

    返回值：Hccl执行结果，成功时返回HCCL\_SUCCESS，异常时返回相应的错误类型。

    ```
    virtual HcclResult Orchestrate(const std::string& algName,
            const OpParam& param, const AlgResourceResponse& algResource);
    ```

    **表 3**  Orchestrate接口参数说明

    <a name="table867451975913"></a>
    <table><thead align="left"><tr id="row6674201914598"><th class="cellrowborder" valign="top" width="16.3%" id="mcps1.2.5.1.1"><p id="p8674111914591"><a name="p8674111914591"></a><a name="p8674111914591"></a>参数</p>
    </th>
    <th class="cellrowborder" valign="top" width="17.72%" id="mcps1.2.5.1.2"><p id="p1667431919595"><a name="p1667431919595"></a><a name="p1667431919595"></a>类型</p>
    </th>
    <th class="cellrowborder" valign="top" width="12.629999999999999%" id="mcps1.2.5.1.3"><p id="p1467441914590"><a name="p1467441914590"></a><a name="p1467441914590"></a>输入/输出</p>
    </th>
    <th class="cellrowborder" valign="top" width="53.349999999999994%" id="mcps1.2.5.1.4"><p id="p17674171913592"><a name="p17674171913592"></a><a name="p17674171913592"></a>说明</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row1167416195594"><td class="cellrowborder" valign="top" width="16.3%" headers="mcps1.2.5.1.1 "><p id="p167471915911"><a name="p167471915911"></a><a name="p167471915911"></a>algName</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.72%" headers="mcps1.2.5.1.2 "><p id="p967491920599"><a name="p967491920599"></a><a name="p967491920599"></a>const std::string &amp;</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.629999999999999%" headers="mcps1.2.5.1.3 "><p id="p15674119185912"><a name="p15674119185912"></a><a name="p15674119185912"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="53.349999999999994%" headers="mcps1.2.5.1.4 "><p id="p16741819105911"><a name="p16741819105911"></a><a name="p16741819105911"></a>算法名称</p>
    </td>
    </tr>
    <tr id="row567411975914"><td class="cellrowborder" valign="top" width="16.3%" headers="mcps1.2.5.1.1 "><p id="p14674181915915"><a name="p14674181915915"></a><a name="p14674181915915"></a>param</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.72%" headers="mcps1.2.5.1.2 "><p id="p136748199598"><a name="p136748199598"></a><a name="p136748199598"></a>const OpParam &amp;</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.629999999999999%" headers="mcps1.2.5.1.3 "><p id="p76741019115914"><a name="p76741019115914"></a><a name="p76741019115914"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="53.349999999999994%" headers="mcps1.2.5.1.4 "><p id="p11674101910595"><a name="p11674101910595"></a><a name="p11674101910595"></a>算子的入参，包括输入输出指针、数据量等信息。</p>
    </td>
    </tr>
    <tr id="row367431917597"><td class="cellrowborder" valign="top" width="16.3%" headers="mcps1.2.5.1.1 "><p id="p5675519105915"><a name="p5675519105915"></a><a name="p5675519105915"></a>algResource</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.72%" headers="mcps1.2.5.1.2 "><p id="p56751819115917"><a name="p56751819115917"></a><a name="p56751819115917"></a>const AlgResourceResponse &amp;</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.629999999999999%" headers="mcps1.2.5.1.3 "><p id="p10675141945914"><a name="p10675141945914"></a><a name="p10675141945914"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="53.349999999999994%" headers="mcps1.2.5.1.4 "><p id="p3675161917593"><a name="p3675161917593"></a><a name="p3675161917593"></a>算法传给框架的资源。</p>
    </td>
    </tr>
    </tbody>
    </table>

    返回值：Hccl执行结果，成功时返回HCCL\_SUCCESS，异常时返回相应的错误类型。

5.  在xxx\_operator.cc文件的结尾处，通过REGISTER\_OP注册算子，以便框架层能够根据HcclCMDType构造并获取算子实例（通过调用 HcclAlg::GetAlgOperator 接口）。

