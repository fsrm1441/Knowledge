#!/bin/bash

# 确保脚本有执行权限
# chmod +x start_rag_api.sh

# 设置环境变量（如果需要）
# export MODEL_TYPE=deepseek
# export WORD_DOC_PATH=path/to/your/document.docx
# 设置端口环境变量（可选，默认值已在.env中设置）
# export RAG_API_PORT=8001

# 启动RAG知识库API服务
export PYTHONUNBUFFERED=1
python api_rag_knowledge.py

read -p "按Enter键继续..."