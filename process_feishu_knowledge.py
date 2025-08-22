# 导入必要的库
import os
import requests
from dotenv import load_dotenv
from langchain_knowledge import DeepSeekKnowledgeBase
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# 从.env文件加载环境变量
load_dotenv()

class FeishuKnowledgeProcessor:
    """飞书云文档和直属库处理类"""
    def __init__(self):
        # 从环境变量加载飞书配置
        self.app_id = os.getenv('FEISHU_APP_ID')
        self.app_secret = os.getenv('FEISHU_APP_SECRET')
        self.document_id = os.getenv('FEISHU_DOCUMENT_ID')
        self.knowledge_base_id = os.getenv('FEISHU_KNOWLEDGE_BASE_ID')
        self.api_base_url = os.getenv('FEISHU_API_BASE_URL', 'https://open.feishu.cn/open-apis')

        # 初始化访问令牌
        self.tenant_access_token = None
        
        # 获取访问令牌
        self._get_tenant_access_token()

    def _get_tenant_access_token(self):
        """获取飞书应用的tenant_access_token"""
        if not self.app_id or not self.app_secret:
            print("错误: 未配置飞书APP ID或APP Secret")
            return

        url = f"{self.api_base_url}/auth/v3/tenant_access_token/internal"
        headers = {
            'Content-Type': 'application/json; charset=utf-8'
        }
        data = {
            "app_id": self.app_id,
            "app_secret": self.app_secret
        }

        try:
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                result = response.json()
                if result.get('code') == 0:
                    self.tenant_access_token = result.get('tenant_access_token')
                    print("成功获取飞书访问令牌")
                else:
                    print(f"获取飞书访问令牌失败: {result.get('msg')}")
            else:
                print(f"获取飞书访问令牌请求失败: {response.status_code}")
        except Exception as e:
            print(f"获取飞书访问令牌发生异常: {str(e)}")

    def get_document_content(self, document_id=None):
        """获取飞书云文档内容

        Args:
            document_id: 文档ID，默认为None（使用环境变量配置的值）

        Returns:
            str: 文档内容，若失败则返回None
        """
        if not self.tenant_access_token:
            print("错误: 飞书访问令牌未初始化")
            return None

        doc_id = document_id if document_id else self.document_id
        if not doc_id:
            print("错误: 未提供文档ID")
            return None

        url = f"{self.api_base_url}/doc/v2/{doc_id}/content"
        headers = {
            'Authorization': f'Bearer {self.tenant_access_token}',
            'Content-Type': 'application/json; charset=utf-8'
        }

        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                result = response.json()
                if result.get('code') == 0:
                    # 提取文档内容，具体结构可能需要根据飞书API返回调整
                    content = result.get('data', {}).get('content', '')
                    return content
                else:
                    print(f"获取文档内容失败: {result.get('msg')}")
            else:
                print(f"获取文档内容请求失败: {response.status_code}")
        except Exception as e:
            print(f"获取文档内容发生异常: {str(e)}")

        return None

    def get_knowledge_base_content(self, knowledge_base_id=None):
        """获取飞书直属库内容

        Args:
            knowledge_base_id: 直属库ID，默认为None（使用环境变量配置的值）

        Returns:
            list: 文档内容列表，若失败则返回None
        """
        if not self.tenant_access_token:
            print("错误: 飞书访问令牌未初始化")
            return None

        kb_id = knowledge_base_id if knowledge_base_id else self.knowledge_base_id
        if not kb_id:
            print("错误: 未提供直属库ID")
            return None

        # 这里需要根据飞书直属库API的实际接口进行实现
        # 以下为示例代码框架，实际实现可能需要调整
        url = f"{self.api_base_url}/knowledge/v1/bases/{kb_id}/documents"
        headers = {
            'Authorization': f'Bearer {self.tenant_access_token}',
            'Content-Type': 'application/json; charset=utf-8'
        }

        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                result = response.json()
                if result.get('code') == 0:
                    # 提取文档列表
                    documents = result.get('data', {}).get('documents', [])
                    document_contents = []

                    # 遍历文档并获取内容
                    for doc in documents:
                        doc_id = doc.get('document_id')
                        doc_title = doc.get('title')
                        print(f"正在获取文档: {doc_title}")
                        doc_content = self.get_document_content(doc_id)
                        if doc_content:
                            document_contents.append({
                                'title': doc_title,
                                'content': doc_content
                            })

                    return document_contents
                else:
                    print(f"获取直属库文档列表失败: {result.get('msg')}")
            else:
                print(f"获取直属库文档列表请求失败: {response.status_code}")
        except Exception as e:
            print(f"获取直属库内容发生异常: {str(e)}")

        return None

    def process_feishu_documents(self, save_path="feishu_knowledge_base"):
        """处理飞书文档并创建知识库

        Args:
            save_path: 知识库保存路径

        Returns:
            DeepSeekKnowledgeBase: 初始化并创建好的知识库实例，若失败则返回None
        """
        try:
            # 初始化知识库
            kb = DeepSeekKnowledgeBase()

            # 处理飞书云文档
            if self.document_id:
                print("正在处理飞书云文档...")
                doc_content = self.get_document_content()
                if doc_content:
                    # 将文档内容写入临时文件
                    temp_file = "temp_feishu_doc.txt"
                    with open(temp_file, 'w', encoding='utf-8') as f:
                        f.write(doc_content)

                    # 加载文档
                    loader = TextLoader(temp_file, encoding="utf-8")
                    document = loader.load()

                    # 分割文档
                    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                    docs = text_splitter.split_documents(document)

                    # 添加到知识库
                    kb.add_documents(docs)

                    # 删除临时文件
                    os.remove(temp_file)

            # 处理飞书直属库
            if self.knowledge_base_id:
                print("正在处理飞书直属库...")
                kb_documents = self.get_knowledge_base_content()
                if kb_documents:
                    for doc in kb_documents:
                        # 将文档内容写入临时文件
                        temp_file = f"temp_feishu_kb_{doc['title'].replace(' ', '_')}.txt"
                        with open(temp_file, 'w', encoding='utf-8') as f:
                            f.write(doc['content'])

                        # 加载文档
                        loader = TextLoader(temp_file, encoding="utf-8")
                        document = loader.load()

                        # 分割文档
                        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                        docs = text_splitter.split_documents(document)

                        # 添加到知识库
                        kb.add_documents(docs)

                        # 删除临时文件
                        os.remove(temp_file)

            # 保存知识库
            kb.save_knowledge_base(save_path)

            return kb
        except Exception as e:
            print(f"处理飞书文档时发生错误: {str(e)}")
            return None

    @staticmethod
    def start_interactive_query(kb):
        """启动交互式查询会话
        
        Args:
            kb: 已初始化的知识库实例
        """
        if not kb:
            print("错误: 知识库未初始化")
            return
            
        # 交互式查询
        # 从环境变量获取欢迎信息，如未设置则使用默认值
        welcome_message = os.getenv('WELCOME_MESSAGE', "欢迎！我是RAG专家，为您提供专业名词解释服务。")
        print(f"\n===== {welcome_message} =====")
        print("输入'退出'、'exit'、'quit'、'Q!'或'q!'结束程序")
        
        while True:
            question = input("\n请输入您的问题: ")
            # 检查退出命令
            if question.lower() in ["退出", "exit", "quit"] or question in ["Q!", "q!"]:
                break
            
            result = kb.query_knowledge_base(question)
            if result:
                print(f"\n回答: {result['answer']}")

# 主函数
if __name__ == "__main__":
    # 创建飞书知识处理器
    feishu_processor = FeishuKnowledgeProcessor()

    # 处理飞书文档并创建知识库
    kb = feishu_processor.process_feishu_documents()

    if kb:
        print("飞书知识库创建成功！")
        # 启动交互式查询
        FeishuKnowledgeProcessor.start_interactive_query(kb)
    else:
        print("飞书知识库创建失败。")