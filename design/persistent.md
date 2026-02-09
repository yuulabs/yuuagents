## 持久化设计

yuuagents 有两套持久化系统：

1. 任务持久化系统。对于下发的任务，它可能由于各种原因挂起（比如说网络，停机，请求用户输入，etc），在下次加载时需要能够重启。这主要是储存agent对话历史。目前的代码对于AgentState只使用了in-memory储存，这导致停机会发生数据丢失。
2. Ytrace. 这是目前代码里面已经完成的系统，它是可观测性。

## 持久化任务

### 储存

yuuagents使用任何数据库储存；默认使用sqlite. 使用异步ORM编写储存层。测试时使用sqlite的in-memory模式。

任务唯一地由task id标定。基本照抄当前的AgentState即可。

yuuagents有一个任务批量收集器，它会定期地积攒任务，然后批量写入数据库，以避免频繁写入数据库。yuuagents不保证eagerly写入数据库（通常会导致性能问题），但是会在程序退出时写入。

### 加载

yuuagents启动时，会从数据库加载未完成任务。特别地，指处于running或者block_on_input状态的任务。有趣的是，请求工具调用的任务被视作block on input（这与当前代码有出入）。

### 挂起

yuuagents调用工具时，将切换至block on input状态。当一个agent长时间处于该状态时，yuuagents会将其挂起，flush到数据库中