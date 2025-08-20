# LangChain 示例：自定义链式调用
# 导入必要的库
import os
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek

# 从.env文件加载环境变量
load_dotenv()

# 注意：运行前需要在.env文件中设置DEEPSEEK_API_KEY
# 格式：DEEPSEEK_API_KEY=你的API密钥

# 创建LLM实例
# API密钥会自动从环境变量中获取
llm = ChatDeepSeek(model="deepseek-chat", temperature=0.7)

# 创建提示模板
prompt_template = PromptTemplate(
    input_variables=["question"],
    template="请简要回答以下问题：{question}\n"
)

# 创建LLMChain
chain = LLMChain(llm=llm, prompt=prompt_template)

# 从控制台获取用户输入的问题
user_question = input("请输入您的问题：")

# 使用用户输入的问题运行链式调用
result = chain.run(question=user_question)

# 打印结果
print(f"\n问题：{user_question}")
print(f"回答：{result}")
