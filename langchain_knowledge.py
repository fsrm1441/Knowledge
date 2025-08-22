# LangChain + 多模型知识库示例
# 导入必要的库
import os
from dotenv import load_dotenv
# 移除无效的导入语句
# from huggingface_hub import set_proxy  # 修改导入
import requests.adapters
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
# 添加Word文档加载器
try:
    from langchain_community.document_loaders import Docx2txtLoader
except ImportError:
    print("警告: 未安装docx2txt库，将无法加载Word文档")
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
# 在文件顶部添加必要的导入
from langchain_core.prompts import PromptTemplate

# 导入不同模型的支持库
# 首先获取并标准化模型类型
model_type = os.getenv('MODEL_TYPE', 'deepseek').lower()

# 根据选定的模型类型导入对应的库
if model_type == 'deepseek':
    try:
        # DeepSeek模型支持
        from langchain_deepseek import ChatDeepSeek
    except ImportError:
        print("警告: 未安装langchain_deepseek库，DeepSeek模型不可用")
elif model_type == 'qianwen':
    try:
        # 千问模型支持
        from langchain_community.chat_models import QianwenChat
    except ImportError:
        print("警告: 未安装千问模型相关依赖，千问模型不可用")
elif model_type == 'doubao':
    try:
        # 豆包模型支持
        from langchain_community.chat_models import DoubaoChat
    except ImportError:
        print("警告: 未安装豆包模型相关依赖，豆包模型不可用")
elif model_type == 'ollama':
    try:
        # Ollama本地模型支持
        from langchain_community.chat_models import ChatOllama
    except ImportError:
        print("警告: 未安装ChatOllama库，Ollama本地模型不可用")
else:
    print("警告: 未配置有效模型类型，默认使用DeepSeek模型")
    # 默认尝试导入DeepSeek模型
    try:
        from langchain_deepseek import ChatDeepSeek
    except ImportError:
        print("警告: 未安装langchain_deepseek库，DeepSeek模型不可用")


# 从.env文件加载环境变量
load_dotenv()

# 从环境变量加载HF_ENDPOINT配置，已在.env文件中设置
# 不需要在代码中硬编码设置，dotenv会自动加载所有环境变量

class DeepSeekKnowledgeBase:
    """基于LangChain的多模型知识库类"""
    def __init__(self):
        # 从环境变量加载模型配置
        model_type = os.getenv('MODEL_TYPE', 'deepseek').lower()
        temperature = float(os.getenv('TEMPERATURE', '0.7'))
        
        # 根据模型类型初始化不同的模型
        self.llm = self._initialize_model(model_type, temperature)
        
        # 初始化嵌入模型 (使用开源的HuggingFace嵌入模型)
        # 通过清华镜像源下载模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            # 重要：首次下载时不要使用local_files_only，这样才能从镜像源下载
            # model_kwargs={'local_files_only': True}  # 下载完成后可以取消注释这行
        )
        
        # 知识库向量存储
        self.vector_store = None
        
        # 检索问答链
        self.qa_chain = None
        
    def _initialize_model(self, model_type, temperature):
        """根据模型类型初始化相应的大语言模型
        
        Args:
            model_type: 模型类型，如'deepseek', 'qianwen', 'doubao', 'ollama'
            temperature: 模型温度参数
            
        Returns:
            初始化好的大语言模型实例
        """
        try:
            if model_type == 'deepseek':
                # DeepSeek模型
                model_name = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')
                return ChatDeepSeek(model=model_name, temperature=temperature)
            
            elif model_type == 'qianwen':
                # 千问模型
                model_name = os.getenv('QIANWEN_MODEL', 'qwen-turbo')
                api_key = os.getenv('QIANWEN_API_KEY')
                if not api_key:
                    raise ValueError("千问模型需要配置QIANWEN_API_KEY环境变量")
                return QianwenChat(model_name=model_name, api_key=api_key, temperature=temperature)
            
            elif model_type == 'doubao':
                # 豆包模型
                model_name = os.getenv('DOUBAO_MODEL', 'ERNIE-Bot')
                api_key = os.getenv('DOUBAO_API_KEY')
                if not api_key:
                    raise ValueError("豆包模型需要配置DOUBAO_API_KEY环境变量")
                return DoubaoChat(model=model_name, api_key=api_key, temperature=temperature)
            
            elif model_type == 'ollama':
                # Ollama本地模型
                model_name = os.getenv('OLLAMA_MODEL', 'llama3')
                base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
                return ChatOllama(model=model_name, base_url=base_url, temperature=temperature)
            
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
                
        except Exception as e:
            print(f"初始化模型失败: {str(e)}")
            print("将使用默认的DeepSeek模型作为备选")
            # 默认使用DeepSeek模型作为备选
            return ChatDeepSeek(model=os.getenv('DEEPSEEK_MODEL', 'deepseek-chat'), 
                               temperature=temperature)
    
    def _load_prompt_template(self):
        """从环境变量加载提示词模板
        
        Returns:
            str: 提示词模板字符串
        """
        # 从环境变量获取提示词模板
        template = os.getenv('RAG_PROMPT_TEMPLATE')
        
        # 如果环境变量中没有，使用默认模板作为备用
        if not template:
            print("警告: 未在环境变量中找到RAG_PROMPT_TEMPLATE，使用默认模板")
            template = """
<instruction>

你是一个RAG专家，专注于回答名词解释。请按照以下步骤完成任务：
1. 使用提供的名词 {question} 生成准确、简洁的名词解释。
2. 解释应包含该名词的基本定义、常见用途或相关背景（如适用）。
3. 确保输出为纯文本，不包含任何XML标签或格式符号。
4. 若名词无法识别或解释，请明确回复"无法提供该名词的解释"。
</instruction>

<input>
需要解释的名词：{question}
</input>

上下文信息:
{context}
"""
        
        return template
    
    def create_knowledge_base(self, file_paths):
        """创建知识库
        Args:
            file_paths: 文档文件路径列表
        """
        documents = []
        
        # 加载每个文档
        for file_path in file_paths:
            if os.path.exists(file_path):
                # 根据文件扩展名选择不同的加载器
                if file_path.endswith('.txt'):
                    loader = TextLoader(file_path, encoding="utf-8")
                elif file_path.endswith('.docx'):
                    try:
                        loader = Docx2txtLoader(file_path)
                    except Exception as e:
                        print(f"加载Word文档 {file_path} 出错: {str(e)}")
                        continue
                else:
                    print(f"警告: 不支持的文件格式: {file_path}")
                    continue
                
                try:
                    document = loader.load()
                    documents.extend(document)
                except Exception as e:
                    print(f"加载文档 {file_path} 出错: {str(e)}")
            else:
                print(f"警告: 文件 {file_path} 不存在")
        
        if not documents:
            print("错误: 没有找到任何文档，请检查文件路径")
            return False
        
        # 分割文档
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        
        # 创建向量存储
        self.vector_store = FAISS.from_documents(texts, self.embeddings)
        
        # 从环境变量加载提示词模板
        template = self._load_prompt_template()
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        # 创建检索问答链并使用自定义提示词
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        print(f"成功创建知识库，共加载 {len(documents)} 个文档，分割为 {len(texts)} 个片段")
        return True
    
    def query_knowledge_base(self, question):
        """查询知识库
        Args:
            question: 查询问题
        Returns:
            回答和相关文档
        """
        if not self.qa_chain:
            print("错误: 知识库尚未创建，请先调用create_knowledge_base方法")
            return None
        
        try:
            # 修改为与模板一致的变量名
            result = self.qa_chain.invoke({
                "query": question
            })
            
            # 格式化回答
            answer = result["result"]
            source_documents = result["source_documents"]
            
            return {
                "answer": answer,
                "sources": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    } for doc in source_documents
                ]
            }
        except Exception as e:
            print(f"查询出错: {str(e)}")
            return None
    
    def get_knowledge_answer(self, term_to_explain, use_fallback=False):
        """获取知识库中关于特定术语的解释（封装增强版查询方法）
        
        Args:
            term_to_explain: 需要解释的术语
            use_fallback: 当查询失败时是否使用默认回复
            
        Returns:
            dict: 包含回答和来源的字典，格式为{"answer": str, "sources": list}
        """
        # 参数验证
        if not term_to_explain or not isinstance(term_to_explain, str):
            return {"answer": "请提供有效的查询术语", "sources": []}
        
        # 尝试查询知识库
        result = self.query_knowledge_base(term_to_explain)
        
        if result:
            # 添加查询成功的信息
            result["status"] = "success"
            return result
        elif use_fallback:
            # 使用默认回复
            return {
                "answer": f"无法提供 '{term_to_explain}' 的解释",
                "sources": [],
                "status": "fallback"
            }
        else:
            return {
                "answer": "查询失败，请检查知识库是否已正确创建",
                "sources": [],
                "status": "error"
            }
    
    def create_and_query_knowledge_base(self, file_paths, query, save_path=None):
        """一站式创建知识库并查询（便捷封装方法）
        
        Args:
            file_paths: 文档文件路径列表
            query: 查询问题
            save_path: 可选，保存知识库的路径
            
        Returns:
            dict: 包含回答和来源的字典
        """
        # 创建知识库
        success = self.create_knowledge_base(file_paths)
        if not success:
            return {"answer": "知识库创建失败", "sources": [], "status": "error"}
        
        # 保存知识库（如果提供了路径）
        if save_path:
            self.save_knowledge_base(save_path)
        
        # 查询知识库
        return self.get_knowledge_answer(query)
    
    def save_knowledge_base(self, file_path):
        """保存知识库
        Args:
            file_path: 保存路径
        """
        if not self.vector_store:
            print("错误: 没有知识库可保存")
            return False
        
        try:
            # 提取目录路径
            dir_path = os.path.dirname(file_path)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path)
            
            # 保存向量存储
            self.vector_store.save_local(file_path)
            print(f"知识库已保存到 {file_path}")
            return True
        except Exception as e:
            print(f"保存知识库出错: {str(e)}")
            return False
    
    def load_knowledge_base(self, file_path):
        """加载已保存的知识库
        Args:
            file_path: 知识库文件路径
        """
        try:
            # 加载向量存储
            self.vector_store = FAISS.load_local(file_path, self.embeddings, allow_dangerous_deserialization=True)
            
            # 从环境变量加载提示词模板
            template = self._load_prompt_template()
            
            prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )
            
            # 创建检索问答链并使用自定义提示词
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                # 保持与上面相同的k值调整
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt}  # 添加这一行以使用自定义提示词
            )
            
            print(f"成功加载知识库: {file_path}")
            return True
        except Exception as e:
            print(f"加载知识库出错: {str(e)}")
            return False
    
    def load_and_query_knowledge_base(self, file_path, query):
        """一站式加载知识库并查询（便捷封装方法）
        
        Args:
            file_path: 知识库文件路径
            query: 查询问题
            
        Returns:
            dict: 包含回答和来源的字典
        """
        # 加载知识库
        success = self.load_knowledge_base(file_path)
        if not success:
            return {"answer": f"知识库加载失败: {file_path}", "sources": [], "status": "error"}
        
        # 查询知识库
        return self.get_knowledge_answer(query)


# 示例用法
if __name__ == "__main__":
    # 创建示例文档目录
    sample_docs_dir = "sample_docs"
    os.makedirs(sample_docs_dir, exist_ok=True)
    
    # 创建示例文档
    sample_content = """LangChain是一个用于构建大语言模型应用程序的框架。
它提供了一套工具、组件和接口，使开发者能够更容易地构建端到端的应用程序。

LangChain的主要特性包括：
1. 模型集成：支持多种大语言模型，如OpenAI、DeepSeek等
2. 检索增强：允许模型访问外部知识库
3. 代理系统：支持构建基于LLM的代理
4. 链式调用：简化复杂工作流的构建

LangChain的核心组件包括：
- 模型：各种LLM和聊天模型
- 提示：提示模板和管理
- 链：将多个组件链接在一起
- 文档加载器：加载各种格式的文档
- 向量存储：存储和检索嵌入向量
- 代理：使用LLM做出决策的代理系统
"""
    
    # 写入示例文档
    sample_doc_path = os.path.join(sample_docs_dir, "langchain_intro.txt")
    with open(sample_doc_path, "w", encoding="utf-8") as f:
        f.write(sample_content)
    
    print(f"已创建示例文档: {sample_doc_path}")
    
    # 初始化知识库
    kb = DeepSeekKnowledgeBase()
    
    # 创建知识库
    print("正在创建知识库...")
    kb.create_knowledge_base([sample_doc_path])
    
    # 保存知识库
    kb.save_knowledge_base("faiss_knowledge_base")
    
    # 交互式查询
    print("\n===== 知识库问答系统 =====")
    print("输入'退出'结束程序")
    
    while True:
        question = input("\n请输入您的问题: ")
        if question.lower() in ["退出", "exit", "quit"]:
            break
        
        result = kb.query_knowledge_base(question)
        if result:
            print(f"\n回答: {result['answer']}")
            
            # 显示相关来源
            print("\n相关来源:")
            for i, source in enumerate(result['sources'], 1):
                print(f"{i}. {source['metadata'].get('source', '未知来源')}")
                # 显示部分内容
                content_preview = source['content'][:100] + ("..." if len(source['content']) > 100 else "")
                print(f"   内容摘要: {content_preview}")

    print("程序已退出")
