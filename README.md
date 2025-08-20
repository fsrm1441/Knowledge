# LangChain + DeepSeek 知识库问答系统

一个基于LangChain和DeepSeek大语言模型的知识库问答系统，支持文档加载、文本分割、向量存储和检索增强生成(RAG)功能。

## 功能特性

- 📚 支持多种文档格式加载
- 🔍 文本智能分割与向量化存储
- 🤖 集成DeepSeek大语言模型进行智能问答
- 💾 知识库的保存与加载功能
- 💬 交互式命令行问答界面

## 技术栈

- Python 3.13+
- LangChain - 大语言模型应用开发框架
- DeepSeek - 强大的国产大语言模型
- FAISS - 高效的向量相似度搜索库
- Sentence Transformers - 文本嵌入模型

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

创建`.env`文件并添加DeepSeek API密钥：

```
DEEPSEEK_API_KEY=your_api_key_here
```

## 使用方法

### 基本使用

运行知识库问答系统：

```bash
python langchain_knowledge.py
```

### 自定义文档

将你的文档放在`sample_docs`目录下，系统会自动加载目录中的文档创建知识库。

### API使用示例

```python
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
├── LangChain_Study.py  # LangChain基础学习示例
├── langchain_knowledge.py # 知识库问答系统主文件
├── pyproject.toml      # 项目配置文件
├── requirements.txt    # 依赖列表
├── README.md           # 项目说明文档
└── sample_docs/        # 示例文档目录
```

## 许可证

本项目采用MIT许可证 - 详见[LICENSE](LICENSE)文件

## 鸣谢

- [LangChain](https://www.langchain.com/) - 大语言模型应用开发框架
- [DeepSeek](https://www.deepseek.com/) - 大语言模型提供商
- [Hugging Face](https://huggingface.co/) - 提供优秀的开源模型