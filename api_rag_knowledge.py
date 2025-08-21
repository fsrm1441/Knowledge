# 导入必要的库
import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import time

# 导入项目中的Word文档处理类
from process_word_knowledge import WordKnowledgeProcessor

# 从.env文件加载环境变量
from dotenv import load_dotenv
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(title="RAG知识库问答API", 
              description="一个简单的RAG知识库问答接口，用户只需上传问题即可获取答案",
              version="1.0.0")

# 配置CORS，允许所有来源
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局知识库实例
knowledge_base = None

# 请求和响应模型
class QueryRequest(BaseModel):
    question: str = Field(..., description="查询问题")
    use_fallback: bool = Field(False, description="查询失败时是否使用默认回复")
    knowledge_base_path: Optional[str] = Field("word_knowledge_base", description="知识库路径")

class KnowledgeResponse(BaseModel):
    success: bool = Field(..., description="操作是否成功")
    message: str = Field(..., description="操作结果消息")
    answer: Optional[str] = Field(None, description="问题答案")
    processing_time: Optional[float] = Field(None, description="处理时间(秒)")

# 初始化知识库
@app.on_event("startup")
async def startup_event():
    global knowledge_base
    try:
        # 输出当前使用的模型信息
        model_type = os.getenv('MODEL_TYPE', 'deepseek').lower()
        logger.info(f"当前使用的模型类型: {model_type}")
        
        # 初始化知识库
        knowledge_base = WordKnowledgeProcessor.process_word_document()
        if knowledge_base:
            logger.info("知识库实例已成功初始化")
        else:
            logger.warning("知识库初始化失败，将在首次请求时尝试重新初始化")
    except Exception as e:
        logger.error(f"初始化知识库失败: {str(e)}")
        # 即使初始化失败，应用仍能启动，后续请求会返回错误

# API端点
@app.get("/", tags=["基础接口"])
async def root():
    return {"message": "欢迎使用RAG知识库问答API", "status": "running"}

@app.get("/status", tags=["基础接口"])
async def get_status():
    """获取API服务状态"""
    model_type = os.getenv('MODEL_TYPE', 'deepseek').lower()
    kb_status = "已初始化" if knowledge_base else "未初始化"
    return {
        "status": "running",
        "model_type": model_type,
        "knowledge_base_status": kb_status
    }

@app.post("/query", tags=["问答接口"], response_model=KnowledgeResponse)
async def query_knowledge(request: QueryRequest):
    """用户上传问题，获取知识库回答"""
    start_time = time.time()
    try:
        global knowledge_base
        
        # 检查知识库是否初始化，如果没有初始化则尝试初始化
        if not knowledge_base:
            logger.info("知识库未初始化，正在尝试初始化...")
            knowledge_base = WordKnowledgeProcessor.process_word_document()
            
            if not knowledge_base:
                raise HTTPException(status_code=503, detail="知识库初始化失败，请检查Word文档路径和模型配置")
            
            logger.info("知识库初始化成功")
        
        # 处理用户问题
        result = knowledge_base.query_knowledge_base(request.question)
        
        if not result or not result.get('answer'):
            if request.use_fallback:
                return KnowledgeResponse(
                    success=True,
                    message="查询成功，但知识库中没有找到相关信息",
                    answer="抱歉，我无法从现有知识库中找到相关信息。",
                    processing_time=round(time.time() - start_time, 2)
                )
            else:
                raise HTTPException(status_code=404, detail="在知识库中未找到相关信息")
        
        return KnowledgeResponse(
            success=True,
            message="查询成功",
            answer=result['answer'],
            processing_time=round(time.time() - start_time, 2)
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"查询知识库出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"查询知识库出错: {str(e)}")

@app.post("/reload_knowledge_base", tags=["知识库操作"], response_model=KnowledgeResponse)
async def reload_knowledge_base(knowledge_base_path: str = "word_knowledge_base"):
    """重新加载知识库"""
    try:
        global knowledge_base
        logger.info(f"正在重新加载知识库: {knowledge_base_path}")
        
        # 重新加载知识库
        knowledge_base = WordKnowledgeProcessor.process_word_document(save_path=knowledge_base_path)
        
        if not knowledge_base:
            raise HTTPException(status_code=500, detail="知识库重新加载失败")
        
        return KnowledgeResponse(
            success=True,
            message="知识库重新加载成功",
            answer=None
        )
    except Exception as e:
        logger.error(f"重新加载知识库出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"重新加载知识库出错: {str(e)}")

# 运行服务器
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")