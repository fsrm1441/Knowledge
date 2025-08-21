# 导入必要的库
import os
from dotenv import load_dotenv
from langchain_knowledge import DeepSeekKnowledgeBase

# 从.env文件加载环境变量
load_dotenv()

# 从环境变量加载HF_ENDPOINT配置，已在.env文件中设置
# load_dotenv()会自动加载所有环境变量，包括模型配置

class WordKnowledgeProcessor:
    """支持多模型的Word文档知识库处理类"""
    
    @staticmethod
    def process_word_document(doc_path=None, save_path="word_knowledge_base"):
        """处理Word文档并创建知识库
        
        Args:
            doc_path: Word文档路径，默认为None（将从环境变量获取）
            save_path: 知识库保存路径
            
        Returns:
            DeepSeekKnowledgeBase: 初始化并创建好的知识库实例，若失败则返回None
        """
        # 获取文档路径
        word_doc_path = doc_path if doc_path else os.getenv('WORD_DOC_PATH')
        
        # 验证文档路径
        if not word_doc_path:
            print("错误: 未提供文档路径且未配置WORD_DOC_PATH环境变量")
            return None
        
        if not os.path.exists(word_doc_path):
            print(f"错误: 文件 {word_doc_path} 不存在")
            return None
        
        try:
            # 初始化知识库
            kb = DeepSeekKnowledgeBase()
            
            # 创建知识库
            success = kb.create_knowledge_base([word_doc_path])
            
            if not success:
                print("创建知识库失败")
                return None
            
            # 保存知识库
            kb.save_knowledge_base(save_path)
            
            return kb
        except Exception as e:
            print(f"处理文档时发生错误: {str(e)}")
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
def main():
    # 输出当前使用的模型信息
    model_type = os.getenv('MODEL_TYPE', 'deepseek').lower()
    print(f"当前使用的模型类型: {model_type}")
    
    # 处理Word文档并创建知识库
    kb = WordKnowledgeProcessor.process_word_document()
    
    if kb:
        # 启动交互式查询
        WordKnowledgeProcessor.start_interactive_query(kb)

if __name__ == "__main__":
    main()