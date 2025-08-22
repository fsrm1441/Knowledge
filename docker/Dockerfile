# 使用官方Python基础镜像
FROM python:3.13-slim

# 设置工作目录
WORKDIR /app

# 配置apt国内镜像源（阿里云）以加速系统依赖安装
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources

# 安装系统依赖，包括git等必要工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    gcc \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements.txt文件到容器
COPY requirements.txt .

# 安装uv包管理器
RUN pip install uv -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装Python依赖，使用uv加速并直接指定镜像源
RUN uv pip install --system -r requirements.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 复制项目文件到容器
COPY . .

# 创建.env文件，使用.env.temp作为模板
RUN cp .env.temp .env && \
    echo "# Docker环境配置" >> .env

# 暴露API端口
EXPOSE 8000
EXPOSE 8001
EXPOSE 8002

# 设置环境变量
ENV PYTHONUNBUFFERED=1

# 运行三个API服务的脚本
CMD ["/bin/bash", "-c", "python api_server.py & python api_rag_knowledge.py & python api_feishu_knowledge.py"]
