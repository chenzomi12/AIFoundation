# 增加hcclComm层接口 

hcclComm为通信域层，在HCCL架构中隶属于框架层。

涉及代码文件：

```
src/domain/collective_communication/framework/communicator/hccl_comm.cc
src/domain/collective_communication/framework/inc/hccl_comm_pub.h
```

1.  定义新算子的API。

    ```
    MyOperatorOutPlace(args...)
    ```

    其中OutPlace后缀代表单算子模式。

    例如，ReduceScatter算子的定义如下：

    ```
    HcclResult ReduceScatterOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op, HcclRtStream stream);
    ```

2.  校验入参合法性。

    HcclCommunicator类（见[增加HcclCommunicator接口](增加HcclCommunicator接口.md)）中提供了的若干Check接口，可以按需调用以实现入参校验。

    例如：CheckDataType，CheckReduceDataType，CheckUserRank等。

    例如检查消减运算op是否支持当前的dataType：

    ```
    CHK_RET(communicator_->CheckReduceDataType(dataType, op));
    ```

    其中 CHK\_RET 宏表示检查返回值，若返回值不为 HCCL\_SUCCESS 则 return 这个返回值。

3.  调用HcclCommunicator层算子接口。

    以 ReduceScatter 算子为例：

    ```
    CHK_RET(communicator_->ReduceScatterOutPlace(tag, inputPtr, outputPtr, count, dataType, op, stream));
    ```

    关于HcclCommunicator层接口的详细实现方法可参见[增加HcclCommunicator接口](增加HcclCommunicator接口.md)。

