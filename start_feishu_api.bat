@echo off

:: 确保中文显示正常
chcp 65001

:: 启动飞书知识库API服务
echo 正在启动飞书知识库API服务...
echo ====================================================
echo 请确保已正确配置.env文件中的飞书相关信息：
echo - FEISHU_APP_ID：飞书应用ID
- FEISHU_APP_SECRET：飞书应用密钥
- FEISHU_DOCUMENT_ID：飞书云文档ID（可选）
- FEISHU_KNOWLEDGE_BASE_ID：飞书直属库ID（可选）
- FEISHU_API_PORT：API服务端口（默认为8002）
echo ====================================================

:: 检查是否安装了必要的依赖
echo 正在检查Python依赖...
pip install fastapi uvicorn python-dotenv

:: 启动API服务
python api_feishu_knowledge.py

:: 检查命令执行状态
if %errorlevel% neq 0 (
    echo 飞书知识库API服务启动失败！
    pause
    exit /b %errorlevel%
)

:: 保持窗口打开
pause