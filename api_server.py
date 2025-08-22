# 导入必要的库
import os
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import List, Optional
import tempfile
import shutil

# 导入项目中的知识库类
from langchain_knowledge import DeepSeekKnowledgeBase

# 从.env文件加载环境变量
from dotenv import load_dotenv
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global knowledge_base
    # 启动时初始化
    try:
        # 创建知识库实例
        knowledge_base = DeepSeekKnowledgeBase()
        logger.info("知识库实例已成功初始化")
    except Exception as e:
        logger.error(f"初始化知识库失败: {str(e)}")
        # 即使初始化失败，应用仍能启动，后续请求会返回错误
    
    yield
    
    # 关闭时清理
    logger.info("应用正在关闭...")

# 创建FastAPI应用
app = FastAPI(title="LangChain多模型知识库API", 
              description="基于FastAPI的知识库问答系统接口，支持多种大语言模型",
              version="1.0.0",
              lifespan=lifespan)

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
class CreateKnowledgeBaseRequest(BaseModel):
    file_paths: List[str] = Field(..., description="文档文件路径列表")

class ProcessWordDocumentRequest(BaseModel):
    doc_path: Optional[str] = Field(None, description="Word文档路径，未提供时从环境变量获取")
    save_path: str = Field("word_knowledge_base", description="知识库保存路径")

class QueryRequest(BaseModel):
    question: str = Field(..., description="查询问题")
    use_fallback: bool = Field(False, description="查询失败时是否使用默认回复")

class SaveKnowledgeBaseRequest(BaseModel):
    save_path: str = Field(..., description="知识库保存路径")

class LoadKnowledgeBaseRequest(BaseModel):
    file_path: str = Field(..., description="知识库文件路径")

class CreateAndQueryRequest(BaseModel):
    file_paths: List[str] = Field(..., description="文档文件路径列表")
    query: str = Field(..., description="查询问题")
    save_path: Optional[str] = Field(None, description="可选的知识库保存路径")

class KnowledgeBaseResponse(BaseModel):
    success: bool = Field(..., description="操作是否成功")
    message: str = Field(..., description="操作结果消息")
    data: Optional[dict] = Field(None, description="操作结果数据")


# API端点
@app.get("/", tags=["基础接口"])
async def root():
    return {"message": "欢迎使用LangChain多模型知识库API", "status": "running"}

@app.get("/status", tags=["基础接口"])
async def get_status():
    """获取API服务状态"""
    model_type = os.getenv('MODEL_TYPE', 'deepseek').lower()
    vector_store_status = "已初始化" if (knowledge_base and knowledge_base.vector_store) else "未初始化"
    return {
        "status": "running",
        "model_type": model_type,
        "vector_store_status": vector_store_status
    }

@app.post("/knowledge/create", tags=["知识库操作"], response_model=KnowledgeBaseResponse)
async def create_knowledge_base(request: CreateKnowledgeBaseRequest):
    """创建知识库"""
    try:
        if not knowledge_base:
            raise HTTPException(status_code=500, detail="知识库未初始化")
            
        success = knowledge_base.create_knowledge_base(request.file_paths)
        if not success:
            return KnowledgeBaseResponse(
                success=False,
                message="知识库创建失败，请检查文件路径或文件格式",
                data=None
            )
        
        return KnowledgeBaseResponse(
            success=True,
            message="知识库创建成功",
            data={"file_count": len(request.file_paths)}
        )
    except Exception as e:
        logger.error(f"创建知识库出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"创建知识库出错: {str(e)}")

@app.post("/knowledge/query", tags=["知识库操作"], response_model=KnowledgeBaseResponse)
async def query_knowledge_base(request: QueryRequest):
    """查询知识库"""
    try:
        if not knowledge_base:
            raise HTTPException(status_code=500, detail="知识库未初始化")
        
        result = knowledge_base.get_knowledge_answer(request.question, request.use_fallback)
        if not result:
            return KnowledgeBaseResponse(
                success=False,
                message="查询失败，请检查知识库是否已创建",
                data=None
            )
        
        return KnowledgeBaseResponse(
            success=True,
            message="查询成功",
            data=result
        )
    except Exception as e:
        logger.error(f"查询知识库出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"查询知识库出错: {str(e)}")

@app.post("/knowledge/save", tags=["知识库操作"], response_model=KnowledgeBaseResponse)
async def save_knowledge_base(request: SaveKnowledgeBaseRequest):
    """保存知识库"""
    try:
        if not knowledge_base:
            raise HTTPException(status_code=500, detail="知识库未初始化")
        
        success = knowledge_base.save_knowledge_base(request.save_path)
        if not success:
            return KnowledgeBaseResponse(
                success=False,
                message="知识库保存失败",
                data=None
            )
        
        return KnowledgeBaseResponse(
            success=True,
            message="知识库保存成功",
            data={"save_path": request.save_path}
        )
    except Exception as e:
        logger.error(f"保存知识库出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"保存知识库出错: {str(e)}")

@app.post("/knowledge/load", tags=["知识库操作"], response_model=KnowledgeBaseResponse)
async def load_knowledge_base(request: LoadKnowledgeBaseRequest):
    """加载知识库"""
    try:
        if not knowledge_base:
            raise HTTPException(status_code=500, detail="知识库未初始化")
        
        success = knowledge_base.load_knowledge_base(request.file_path)
        if not success:
            return KnowledgeBaseResponse(
                success=False,
                message="知识库加载失败，请检查文件路径",
                data=None
            )
        
        return KnowledgeBaseResponse(
            success=True,
            message="知识库加载成功",
            data={"file_path": request.file_path}
        )
    except Exception as e:
        logger.error(f"加载知识库出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"加载知识库出错: {str(e)}")

@app.post("/knowledge/create_and_query", tags=["知识库操作"], response_model=KnowledgeBaseResponse)
async def create_and_query_knowledge_base(request: CreateAndQueryRequest):
    """一站式创建知识库并查询"""
    try:
        if not knowledge_base:
            raise HTTPException(status_code=500, detail="知识库未初始化")
        
        result = knowledge_base.create_and_query_knowledge_base(
            request.file_paths, 
            request.query, 
            request.save_path
        )
        
        if result.get("status") == "error":
            return KnowledgeBaseResponse(
                success=False,
                message="创建并查询知识库失败",
                data=result
            )
        
        return KnowledgeBaseResponse(
            success=True,
            message="创建并查询知识库成功",
            data=result
        )
    except Exception as e:
        logger.error(f"创建并查询知识库出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"创建并查询知识库出错: {str(e)}")

@app.post("/knowledge/process_word", tags=["Word文档处理"])
async def process_word_document(request: ProcessWordDocumentRequest):
    """处理Word文档并创建知识库"""
    try:
        if not knowledge_base:
            raise HTTPException(status_code=500, detail="知识库未初始化")
        
        # 获取文档路径
        word_doc_path = request.doc_path if request.doc_path else os.getenv('WORD_DOC_PATH')
        
        # 验证文档路径
        if not word_doc_path:
            raise HTTPException(status_code=400, detail="未提供文档路径且未配置WORD_DOC_PATH环境变量")
        
        if not os.path.exists(word_doc_path):
            raise HTTPException(status_code=404, detail=f"文件 {word_doc_path} 不存在")
        
        # 确保是Word文档
        if not word_doc_path.endswith('.docx'):
            raise HTTPException(status_code=400, detail="仅支持.docx格式的Word文档")
        
        # 创建知识库
        success = knowledge_base.create_knowledge_base([word_doc_path])
        
        if not success:
            return KnowledgeBaseResponse(
                success=False,
                message="创建知识库失败",
                data=None
            )
        
        # 保存知识库
        save_success = knowledge_base.save_knowledge_base(request.save_path)
        
        return KnowledgeBaseResponse(
            success=True,
            message="Word文档处理成功并创建知识库",
            data={
                "doc_path": word_doc_path,
                "save_path": request.save_path,
                "save_success": save_success
            }
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"处理Word文档时发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理Word文档时发生错误: {str(e)}")

@app.post("/knowledge/upload_and_query", tags=["文件上传"])
async def upload_and_query_knowledge_base(files: List[UploadFile] = File(...), query: str = "请解释文档中的主要内容"):
    """上传文件并查询知识库"""
    try:
        if not knowledge_base:
            raise HTTPException(status_code=500, detail="知识库未初始化")
        
        # 创建临时目录保存上传的文件
        with tempfile.TemporaryDirectory() as temp_dir:
            file_paths = []
            
            # 保存上传的文件
            for file in files:
                file_path = os.path.join(temp_dir, file.filename)
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                file_paths.append(file_path)
                
            # 创建并查询知识库
            result = knowledge_base.create_and_query_knowledge_base(file_paths, query)
            
            if result.get("status") == "error":
                return KnowledgeBaseResponse(
                    success=False,
                    message="上传文件并查询知识库失败",
                    data=result
                )
            
            return KnowledgeBaseResponse(
                success=True,
                message="上传文件并查询知识库成功",
                data=result
            )
    except Exception as e:
        logger.error(f"上传文件并查询知识库出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"上传文件并查询知识库出错: {str(e)}")

# 运行服务器
if __name__ == "__main__":
    import uvicorn
    # 从环境变量获取端口，默认为8000，处理空字符串的情况
    port_str = os.getenv("API_SERVER_PORT", "8000")
    try:
        port = int(port_str) if port_str else 8000
    except ValueError:
        port = 8000
        logger.warning(f"无效的端口配置 '{port_str}'，使用默认端口 8000")
    
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
