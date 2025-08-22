# LangChain + 多模型知识库问答系统

一个基于LangChain和多种大语言模型的知识库问答系统，支持文档加载、文本分割、向量存储和检索增强生成(RAG)功能。系统支持DeepSeek、千问、豆包和Ollama本地模型等多种大语言模型，并支持飞书云文档和直属库的集成。

## 功能特性

- 📚 支持多种文档格式加载
- 🔍 文本智能分割与向量化存储
- 🤖 集成多种大语言模型进行智能问答（支持DeepSeek、千问、豆包、Ollama本地模型）
- 💾 知识库的保存与加载功能
- 💬 交互式命令行问答界面
- 🔄 灵活的模型切换机制，通过配置文件即可切换不同大模型
- 📝 集成飞书云文档和直属库，支持从飞书直接导入内容
- 🌐 提供专用的飞书知识库API服务

## 技术栈

- Python 3.13+
- LangChain - 大语言模型应用开发框架
- DeepSeek/千问/豆包/Ollama - 支持多种大语言模型
- FAISS - 高效的向量相似度搜索库
- Sentence Transformers - 文本嵌入模型
- feishu-sdk-python - 飞书官方Python SDK，用于与飞书API交互

## 安装指南

### 1. 克隆项目

```bash
git clone <项目仓库地址>
cd langchain-study
```

### 2. 创建虚拟环境

```bash
# 使用uv创建虚拟环境
uv venv -p 3.13 .venv

# 激活虚拟环境
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

### 3. 安装依赖

```bash
# 使用uv安装依赖
uv pip install -e .

# 或使用pip
pip install -r requirements.txt
```

### 4. 配置环境变量

创建`.env`文件并根据你选择的模型添加相应的配置：

```
# 模型类型选择 (deepseek, qianwen, doubao, ollama) 默认deepseek
MODEL_TYPE=deepseek
# 模型温度参数，控制生成文本的随机性
TEMPERATURE=0.7

# DeepSeek模型配置
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_MODEL=deepseek-chat

# 千问模型配置（如果选择千问模型，取消下面的注释并填入你的API密钥）
# QIANWEN_API_KEY=your_qianwen_api_key_here
# QIANWEN_MODEL=qwen-turbo

# 豆包模型配置（如果选择豆包模型，取消下面的注释并填入你的API密钥）
# DOBAO_API_KEY=your_doubao_api_key_here
# DOBAO_MODEL=ERNIE-Bot

# Ollama本地模型配置（如果选择Ollama模型，取消下面的注释）
# OLLAMA_MODEL=llama3
# OLLAMA_BASE_URL=http://localhost:11434

# HuggingFace配置（使用镜像源加速下载）
# HF_ENDPOINT=https://hf-mirror.com

# 飞书云文档和直属库配置
FEISHU_APP_ID=your_feishu_app_id_here
FEISHU_APP_SECRET=your_feishu_app_secret_here
# 可选：飞书云文档ID
# FEISHU_DOCUMENT_ID=your_document_id_here
# 可选：飞书直属库ID
# FEISHU_KNOWLEDGE_BASE_ID=your_knowledge_base_id_here
# 飞书API基础URL（默认即可）
FEISHU_API_BASE_URL=https://open.feishu.cn/open-apis
```

## 使用方法

### 基本使用

运行知识库问答系统：

```bash
python langchain_knowledge.py
```

### 完整API服务使用

运行完整的FastAPI服务：

```bash
# 方法1：直接运行Python文件
python api_server.py

# 方法2：使用批处理脚本（Windows环境）
start_api_server.bat

# 方法3：使用shell脚本（Linux/Mac环境）
chmod +x start_api_server.sh
./start_api_server.sh
```

API服务启动后，可以访问以下地址查看API文档：

```
http://localhost:8000/docs
```

> 注意：API服务的端口可以在`.env`文件中通过`API_SERVER_PORT`配置，默认为8000。

### API端点说明

API服务提供了以下主要端点：

- **GET /** - 基础接口，检查服务是否运行
- **GET /status** - 获取API服务状态和模型配置信息
- **POST /knowledge/create** - 创建知识库
- **POST /knowledge/query** - 查询知识库
- **POST /knowledge/save** - 保存知识库
- **POST /knowledge/load** - 加载知识库
- **POST /knowledge/create_and_query** - 一站式创建知识库并查询
- **POST /knowledge/upload_and_query** - 上传文件并查询知识库
- **POST /knowledge/process_word** - 处理Word文档并创建知识库

详细的请求和响应格式可以在API文档中查看。

### 自定义文档

将你的文档放在`sample_docs`目录下，系统会自动加载目录中的文档创建知识库。

### 飞书知识库使用

运行飞书知识库处理程序，创建基于飞书云文档和直属库的知识库：

```bash
# 方法1：直接运行Python文件
python process_feishu_knowledge.py

# 方法2：使用批处理脚本（Windows环境）
start_feishu_knowledge.bat
```

运行后，程序会自动从配置的飞书云文档和直属库中获取内容，创建知识库，并进入交互式问答模式。

> 注意：使用前请确保已在`.env`文件中正确配置了飞书相关参数。

### RAG知识库问答API使用

如果您只需要一个简单的接口让用户上传问题并获取答案，可以使用RAG知识库问答API：

```bash
# 方法1：直接运行Python文件
python api_rag_knowledge.py

# 方法2：使用批处理脚本（Windows环境）
start_rag_api.bat

# 方法3：使用shell脚本（Linux/Mac环境）
chmod +x start_rag_api.sh
./start_rag_api.sh
```

RAG API服务启动后，可以访问以下地址查看API文档：

```
http://localhost:8001/docs
```

> 注意：RAG API服务的端口可以在`.env`文件中通过`RAG_API_PORT`配置，默认为8001。

### 飞书知识库API使用

如果您需要专门针对飞书云文档和直属库的API服务，可以使用飞书知识库API：

```bash
# 方法1：直接运行Python文件
python api_feishu_knowledge.py

# 方法2：使用批处理脚本（Windows环境）
start_feishu_api.bat
```

飞书API服务启动后，可以访问以下地址查看API文档：

```
http://localhost:8002/docs
```

> 注意：飞书API服务的端口可以在`.env`文件中通过`FEISHU_API_PORT`配置，默认为8002。

RAG API提供了以下主要端点：

- **GET /** - 基础接口，检查服务是否运行
- **GET /status** - 获取API服务状态和模型配置信息
- **POST /query** - 用户上传问题，获取知识库回答
- **POST /reload_knowledge_base** - 重新加载知识库

飞书知识库API提供了以下主要端点：

- **GET /** - 基础接口，检查服务是否运行
- **GET /status** - 获取API服务状态和模型配置信息
- **POST /query** - 用户上传问题，获取飞书知识库回答
- **POST /reload_knowledge_base** - 重新加载飞书知识库

### API使用示例

```python
import requests
import json

# 使用RAG API的示例
# 用户只需上传问题即可获取答案
def query_rag_api(question, use_fallback=True):
    url = "http://localhost:8001/query"
    payload = {
        "question": question,
        "use_fallback": use_fallback
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        result = response.json()
        return result.get("answer", "未找到答案")
    else:
        print(f"请求失败: {response.status_code}")
        return None

# 示例使用
if __name__ == "__main__":
    question = "请解释文档中的主要概念"
    answer = query_rag_api(question)
    print(f"问题: {question}")
    print(f"答案: {answer}")

# 使用完整API服务的示例
from langchain_knowledge import DeepSeekKnowledgeBase

# 初始化知识库
kb = DeepSeekKnowledgeBase()

# 创建知识库
kb.create_knowledge_base(['path/to/your/document.txt'])

# 保存知识库
kb.save_knowledge_base('your_knowledge_base')

# 加载知识库
kb.load_knowledge_base('your_knowledge_base')

# 查询知识库
result = kb.query_knowledge_base('你的问题是什么？')
print(result['answer'])
```

## 项目结构

```
├── .env                # 环境变量配置文件
├── .env.temp           # 环境变量模板文件
├── .gitignore          # Git忽略文件
├── LICENSE             # 许可证文件
├── LangChain_Study.py  # LangChain基础学习示例
├── README.md           # 项目说明文档
├── api_feishu_knowledge.py # 飞书知识库API服务
├── api_rag_knowledge.py # RAG知识库问答API
├── api_server.py       # 完整的API服务器
├── docker/             # Docker相关配置
│   ├── Dockerfile
│   ├── README.md
│   └── docker-compose.yml
├── langchain_knowledge.py # 知识库问答系统主文件
├── process_feishu_knowledge.py # 飞书文档处理工具
├── process_word_knowledge.py # Word文档处理工具
├── pyproject.toml      # 项目配置文件
├── requirements.txt    # 依赖列表
├── sample_docs/        # 示例文档目录
├── faiss_knowledge_base/  # 默认FAISS知识库存储目录
├── word_knowledge_base/   # Word文档知识库存储目录
├── feishu_knowledge_base/ # 飞书知识库存储目录
├── start_api_server.bat # 启动API服务器的批处理脚本
├── start_api_server.sh # 启动API服务器的shell脚本（Linux/Mac）
├── start_feishu_api.bat # 启动飞书API服务的批处理脚本
├── start_feishu_knowledge.bat # 启动飞书知识库处理的批处理脚本
├── start_rag_api.bat   # 启动RAG API的批处理脚本
└── start_rag_api.sh    # 启动RAG API的shell脚本（Linux/Mac）
```

## 许可证

本项目采用MIT许可证 - 详见[LICENSE](LICENSE)文件

## Docker部署指南

本项目支持使用Docker进行容器化部署，简化环境配置和服务管理流程。

### 前提条件

- 已安装 [Docker](https://www.docker.com/get-started) 和 [Docker Compose](https://docs.docker.com/compose/install/)

### 准备环境变量

在项目根目录创建 `.env` 文件，并根据 `.env.temp` 文件中的模板填写必要的配置信息，特别是API密钥等敏感信息。

可以在 `.env` 文件中配置以下端口参数：
- `API_SERVER_PORT`: 主要API服务器端口（默认8000）
- `RAG_API_PORT`: RAG知识库问答API端口（默认8001）
- `FEISHU_API_PORT`: 飞书知识库API端口（默认8002）

### 使用Docker Compose运行（推荐）

在项目根目录执行以下命令启动所有服务：

```bash
cd docker
docker-compose up -d --build
```

### 直接使用Docker构建和运行

也可以直接使用Docker命令构建和运行镜像：

```bash
# 构建镜像
docker build -t langchain-api -f docker/Dockerfile .

# 运行容器
docker run -d -p 8000:8000 -p 8001:8001 -p 8002:8002 --name langchain_api_container langchain-api
```

### 服务访问

启动后，可以通过以下地址访问服务：

- API服务器 Swagger 文档: http://localhost:8000/docs
- RAG知识库问答API Swagger 文档: http://localhost:8001/docs
- 飞书知识库API Swagger 文档: http://localhost:8002/docs

### 管理命令

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

### 注意事项

1. 首次构建镜像可能需要较长时间，因为需要下载和安装所有依赖。
2. 确保 `.env` 文件包含了正确的配置信息，特别是API密钥。
3. 如需修改服务端口，请同时修改 `docker-compose.yml` 中的端口映射和项目中的API服务配置。
4. 在开发环境中，代码会通过卷挂载实时同步到容器中，修改代码后只需重启容器或让服务自动重载即可生效。
5. Docker镜像中使用了uv包管理器（替代传统的pip）来加速Python依赖的安装过程，镜像源直接通过命令行参数配置。

## 鸣谢

- [LangChain](https://www.langchain.com/) - 大语言模型应用开发框架
- [DeepSeek](https://www.deepseek.com/) - 大语言模型提供商
- [阿里通义千问](https://qianwen.aliyun.com/) - 大语言模型提供商
- [百度豆包](https://www.doubao.com/) - 大语言模型提供商
- [Ollama](https://ollama.com/) - 本地大语言模型运行平台
- [Hugging Face](https://huggingface.co/) - 提供优秀的开源模型
- [飞书开放平台](https://open.feishu.cn/) - 提供飞书云文档和直属库API
- [Docker](https://www.docker.com/) - 容器化平台，简化部署流程