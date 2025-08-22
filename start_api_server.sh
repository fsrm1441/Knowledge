#!/bin/bash

# 确保脚本有执行权限
# chmod +x start_api_server.sh

# 设置环境变量（可选，默认值已在.env中设置）
# export API_SERVER_PORT=8000

# 检查是否有虚拟环境
if [ -f .venv/bin/activate ]; then
    echo "激活虚拟环境..."
    source .venv/bin/activate
else
    echo "未找到虚拟环境，请先创建虚拟环境"
    read -p "按Enter键继续..."
    exit 1
fi

# 安装依赖
pip install -r requirements.txt

# 启动API服务器
export PYTHONUNBUFFERED=1
python api_server.py

read -p "按Enter键继续..."