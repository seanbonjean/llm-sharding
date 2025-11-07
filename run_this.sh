#!/bin/bash
# 启动多个 start_node.py 实例
# 用法: ./run_this.sh [数量]
# 默认启动 4 个节点，端口从 40700 开始

NUM_NODES=${1:-4}      # 默认4个节点
BASE_PORT=40700        # 起始端口

for ((i=0; i<NUM_NODES; i++)); do
    PORT=$((BASE_PORT + i))
    echo "Starting node $i on port $PORT..."
    # 启动 Python 进程并在后台运行
    python3 start_node.py --port "$PORT" > "node_${PORT}.log" 2>&1 &
    sleep 1   # 稍微延迟，避免瞬间并发
done

echo "All $NUM_NODES nodes started (ports $BASE_PORT to $((BASE_PORT + NUM_NODES - 1)))."
echo 'Use "cat node_<port>.log" to view logs, or "tail -fn 30 node_<port>.log" to follow the last 30 lines of logs.'
echo 'Use "ps aux | grep start_node.py" to check which nodes are running.'
echo 'Use "killall -9 python3" to stop all nodes.'
