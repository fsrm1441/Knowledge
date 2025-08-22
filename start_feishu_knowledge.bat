@echo off

:: 启动飞书知识库处理程序
:: 使用前请确保已在.env文件中配置好飞书相关参数

echo 正在启动飞书知识库处理程序...
echo 请确保已在.env文件中配置好飞书APP ID、APP Secret等参数

python process_feishu_knowledge.py

if %ERRORLEVEL% EQU 0 (
    echo 飞书知识库处理程序执行完成
) else (
    echo 飞书知识库处理程序执行失败
)

pause