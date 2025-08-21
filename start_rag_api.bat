@echo off

rem 设置环境变量（如果需要）
rem set MODEL_TYPE=deepseek
rem set WORD_DOC_PATH=path\to\your\document.docx

rem 启动RAG知识库API服务
python api_rag_knowledge.py

pause