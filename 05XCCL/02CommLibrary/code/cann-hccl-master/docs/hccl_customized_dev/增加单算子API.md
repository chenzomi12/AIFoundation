# 增加单算子API 

本节描述如何添加单算子模式下HCCL对外的算子接口，涉及代码文件为：

```
src/domain/collective_communication/framework/op_base/src/op_base.cc
```

1.  定义新算子的API。

    在 op\_base.cc 中添加定义，如下所示：

    ```
    HcclResult HcclMyOperator(args...)
    ```

    _arfgs_为所有可能参数的列表。

    例如，ReduceScatter算子的定义如下：

    ```
    HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream)
    ```

2.  校验入参合法性。

    对于指针类型的入参，可调用 CHK\_PTR\_NUL 宏检查指针是否为空。

    例如，检查 sendBuf 是否为空指针：

    ```
    CHK_PTR_NULL(sendBuf);
    ```

    如果sendBuf是空指针，则程序会报错并返回。

    此外，param\_check\_pub.h 中提供了若干校验函数，可根据需要调用这些接口来校验入参。

3.  定义tag。tag用作下发算子的标识，与资源复用等功能有关。

    例如，ReduceScatter 算子的 tag 定义为算子名+通信域id：

    ```
    const string tag = "ReduceScatter_" + hcclComm->GetIdentifier();
    ```

    在此定义下，同通信域同算子会复用tag。

4.  配置属性（按需）。

    为通信域配置若干属性的默认值。

    例如：

    SetDefaultQosConfig

    SetOverFlowAddr

5.  调用hcclComm层算子接口。

    关于hcclComm层接口的详细实现方法可参见[增加hcclComm层接口](增加hcclComm层接口.md)。

