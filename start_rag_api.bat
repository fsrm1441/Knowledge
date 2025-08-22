@echo off

:: 设置环境变量（如果需要）
:: set MODEL_TYPE=deepseek
:: set WORD_DOC_PATH=path\to\your\document.docx
:: 设置端口环境变量（可选，默认值已在.env中设置）
:: set RAG_API_PORT=8001

rem 启动RAG知识库API服务
python api_rag_knowledge.py

pause