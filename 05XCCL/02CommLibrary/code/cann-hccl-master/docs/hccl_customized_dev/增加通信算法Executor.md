# 增加通信算法Executor 

Executor实现了算法的主要功能（资源计算和算法编排），由Operator创建并使用。

在以下路径添加新算法Executor的实现文件（包括cc文件和头文件）：

src/domain/collective\_communication/algorithm/impl/coll\_executor/coll\_xxx/coll\_xxx\_executor/

每新增一种算法，都需要新增一个 xxxExecutor 类，继承自算法基类 CollExecutorBase。

通信算法开发具体参考[通信算法开发](通信算法开发.md)。

