# LangChain Docker 部署指南

本目录包含用于将LangChain项目部署到Docker容器的配置文件。

## 前提条件

- 已安装 [Docker](https://www.docker.com/get-started) 和 [Docker Compose](https://docs.docker.com/compose/install/)

## 快速开始

### 1. 准备环境变量

在项目根目录创建 `.env` 文件，并根据 `.env.temp` 文件中的模板填写必要的配置信息，特别是API密钥等敏感信息。

可以在 `.env` 文件中配置以下端口参数：
- `API_SERVER_PORT`: 主要API服务器端口（默认8000）
- `RAG_API_PORT`: RAG知识库问答API端口（默认8001）

### 2. 使用Docker Compose运行（推荐）

在项目根目录执行以下命令启动所有服务：

```bash
cd docker
 docker-compose up -d --build
```

### 3. 直接使用Docker构建和运行

也可以直接使用Docker命令构建和运行镜像：

```bash
# 构建镜像
 docker build -t langchain-api -f docker/Dockerfile .

# 运行容器
 docker run -d -p 8000:8000 -p 8001:8001 --name langchain_api_container langchain-api
```

## 服务访问

启动后，可以通过以下地址访问服务：

- API服务器 Swagger 文档: http://localhost:8000/docs
- RAG知识库问答API Swagger 文档: http://localhost:8001/docs

## 管理命令

```bash
# 查看容器日志
 docker-compose logs -f

# 停止服务
 docker-compose down

# 重启服务
 docker-compose restart

# 进入容器内部
 docker exec -it langchain_api_container /bin/bash
```

## 注意事项

1. 首次构建镜像可能需要较长时间，因为需要下载和安装所有依赖。
2. 确保 `.env` 文件包含了正确的配置信息，特别是API密钥。
3. 如需修改服务端口，请同时修改 `docker-compose.yml` 中的端口映射和项目中的API服务配置。
4. 在开发环境中，代码会通过卷挂载实时同步到容器中，修改代码后只需重启容器或让服务自动重载即可生效。
5. Docker镜像中使用了uv包管理器（替代传统的pip）来加速Python依赖的安装过程，镜像源直接通过命令行参数配置。