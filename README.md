# 模型分割仿真器

该项目将语言模型切割成片，将这些分片部署在各个边缘节点上运行，形成一个模型链，让多个设备能够一起运行一个模型，从而解决内存容量问题，实现边缘协作计算

## 架构简介

整体结构由一个设备担任主节点 `master_node`，其他设备作为普通节点；  
调度算法在主节点上运行，并调用 `ConfigSender` 发送配置给其他节点；  
每个节点都运行一个 `NodeController`，它承担一些通信功能（接收主节点配置信息，接收用户请求），并驱使 `NodeWorker`
工作（接收上一个节点传入的 hidden state ，处理后发送给下一个节点）

配置信息包含：

1. 分片起止序号
2. 是否可以接收用户请求（为保护用户隐私，用户请求必须经过嵌入层才能发送给模型的起点节点，因此节点加载了嵌入层才允许接收用户请求）
3. 模型链下一个节点ip（传递 hidden state 用）
4. 模型链第一个节点的地址（将经过嵌入层处理的用户请求传递给模型链第一个节点）

## 部署

### 安装环境

项目中的 `requirements.txt` 针对 Windows Python3.12 环境，对边缘节点部署时请勿使用，仅作安装参考

### 使用

1. 下载模型到 `weights` 文件夹（暂时只支持 llama2 7b 模型），并使用 `utils/shard_loader.py` 进行切割（需要能够完整加载模型的设备），保存到
   `shards` 文件夹中；
2. 将分片放到各个节点的 `shards` 文件夹中；
3. 各节点运行 `start_node.py`，主节点直接发送配置给所有节点，或按照算法的调度来发送配置

### 运行 `start_node.py` 前，检查配置项

1. `NodeController` 对象中：切片文件的保存位置；模型运行时使用的设备；模型精度
2. `ConfigSender` 的 `.build_config` 方法（若使用算法调度，则忽略本项）

## 注意事项

### 默认端口

* 所有节点接收主节点配置信息的端口：40700
* 所有节点接收模型链上其他节点传入数据的端口：40800（传入数据有多种类型，见 `node_worker.py` 中 `NodeController` 的
  `run_worker_loop` 方法中的多个 `if-elif` 分支；更具体的数据内容详见 `node_worker.py` 中 `NodeWorker` 的
  `receive_user_request`、 `pass_through_shard` 和 `receive_next_token` 方法）
