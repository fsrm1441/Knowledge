@echo off

:: 设置中文显示
chcp 65001

:: 检查是否有虚拟环境
if exist .venv\Scripts\activate ( 
    echo 激活虚拟环境...
    call .venv\Scripts\activate
) else (
    echo 未找到虚拟环境，请先创建虚拟环境
    pause
    exit /b 1
)

:: 安装依赖
pip install -r requirements.txt

:: 启动API服务器
python api_server.py

pause