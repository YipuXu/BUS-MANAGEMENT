"""
FastAPI 后端服务
提供 /api/chat 接口，集成 ChromaDB 检索和 DeepSeek 大模型问答
"""

import os
import json
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel

import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# 加载环境变量
load_dotenv()

# ==================== 配置区 ====================
PROJECT_ROOT = Path(__file__).parent.parent
CHROMA_DIR = PROJECT_ROOT / "backend" / "chroma_db"

# DeepSeek API 配置（从 .env 读取）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# ChromaDB 集合名称
MANUAL_COLLECTION = "maintenance_manual"
CIRCUIT_COLLECTION = "circuit_diagram"

# 检索参数
TOP_K_RESULTS = 5  # 每个集合返回的结果数
# ================================================

# 初始化 FastAPI
app = FastAPI(
    title="工业设备远程诊断 AI 知识库",
    description="多格式工程资料知识库平台后端 API",
    version="1.0.0"
)

# 配置 CORS（允许前端跨域请求）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 数据模型 ====================
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    session_id: str


# ==================== 初始化服务 ====================
chroma_client = None
manual_collection = None
circuit_collection = None
deepseek_client = None


def init_services():
    """初始化 ChromaDB 和 DeepSeek 客户端"""
    global chroma_client, manual_collection, circuit_collection, deepseek_client
    
    # 初始化 ChromaDB
    try:
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        manual_collection = chroma_client.get_collection(MANUAL_COLLECTION)
        circuit_collection = chroma_client.get_collection(CIRCUIT_COLLECTION)
        print(f"[ChromaDB] 连接成功")
        print(f"  - {MANUAL_COLLECTION}: {manual_collection.count()} 条记录")
        print(f"  - {CIRCUIT_COLLECTION}: {circuit_collection.count()} 条记录")
    except Exception as e:
        print(f"[ChromaDB] 连接失败: {e}")
    
    # 初始化 DeepSeek
    if DEEPSEEK_API_KEY:
        deepseek_client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL
        )
        print(f"[DeepSeek] 客户端初始化成功")
    else:
        print("[DeepSeek] 未配置 API Key，将使用 Mock 回复")


# ==================== 核心逻辑 ====================
def retrieve_context(query: str) -> tuple:
    """
    从 ChromaDB 检索相关上下文
    
    Args:
        query: 用户问题
        
    Returns:
        (context_text, sources) 元组
    """
    context_parts = []
    sources = []
    
    # 从维修手册集合检索
    if manual_collection:
        try:
            results = manual_collection.query(
                query_texts=[query],
                n_results=TOP_K_RESULTS
            )
            
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                context_parts.append(f"[资料 {i+1}] {doc}")
                sources.append({
                    'type': 'maintenance_manual',
                    'page': metadata.get('page', '未知'),
                    'table_index': metadata.get('table_index', ''),
                    'chunk_type': metadata.get('chunk_type', ''),
                    'relevance': round(1 - distance, 3)  # 距离越小越相关
                })
        except Exception as e:
            print(f"[检索] 维修手册集合查询失败: {e}")
    
    # 从电路图集合检索
    if circuit_collection and circuit_collection.count() > 0:
        try:
            results = circuit_collection.query(
                query_texts=[query],
                n_results=TOP_K_RESULTS
            )
            
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                context_parts.append(f"[资料 {len(sources)+1}] {doc}")
                sources.append({
                    'type': 'circuit_diagram',
                    'page': metadata.get('page', '未知'),
                    'chunk_type': metadata.get('chunk_type', ''),
                    'relevance': round(1 - distance, 3)
                })
        except Exception as e:
            print(f"[检索] 电路图集合查询失败: {e}")
    
    context_text = "\n\n".join(context_parts)
    return context_text, sources


def build_prompt(query: str, context: str) -> str:
    """
    构建提示词（Prompt）
    将检索到的资料和用户问题结合，确保大模型基于资料回答
    """
    prompt = f"""你是一个工业设备远程诊断专家助手。请根据以下参考资料回答用户的问题。

【参考资料】
{context}

【回答要求】
1. 仅基于上述参考资料回答问题，不要编造信息
2. 如果参考资料中没有相关信息，请明确告知用户
3. 回答要结构化、清晰，适合运维人员阅读
4. 如果涉及具体参数或步骤，请准确引用资料中的数据
5. 回答语言与用户提问语言保持一致

【用户问题】
{query}

请根据参考资料给出专业、准确的回答："""
    
    return prompt


def call_deepseek(prompt: str) -> str:
    """
    调用 DeepSeek API 获取回答
    
    Args:
        prompt: 构建好的提示词
        
    Returns:
        大模型的回答
    """
    if not deepseek_client:
        return "（Mock 回复）DeepSeek API Key 未配置，无法连接真实大模型。请在 .env 文件中配置 DEEPSEEK_API_KEY。"
    
    try:
        response = deepseek_client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": "你是工业设备远程诊断专家，请基于提供的资料准确回答问题。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # 较低的温度确保回答更准确、稳定
            max_tokens=2000
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"（调用 DeepSeek API 失败）{str(e)}"


# ==================== API 接口 ====================
@app.on_event("startup")
def startup_event():
    """服务启动时初始化"""
    print("=" * 60)
    print("工业设备远程诊断 AI 知识库 - 后端服务启动")
    print("=" * 60)
    init_services()


@app.get("/api/health")
def health_check():
    """健康检查接口"""
    return {
        "status": "ok",
        "chroma_db": chroma_client is not None,
        "deepseek": DEEPSEEK_API_KEY != "",
        "manual_records": manual_collection.count() if manual_collection else 0,
        "circuit_records": circuit_collection.count() if circuit_collection else 0
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    聊天问答接口
    
    流程：
    1. 接收用户问题
    2. 从 ChromaDB 检索相关资料
    3. 构建提示词
    4. 调用 DeepSeek 生成回答
    5. 返回回答和来源
    """
    query = request.message.strip()
    
    if not query:
        raise HTTPException(status_code=400, detail="问题不能为空")
    
    # 步骤 1: 检索相关上下文
    context, sources = retrieve_context(query)
    
    if not context:
        return ChatResponse(
            answer="抱歉，知识库中暂时没有相关资料可以回答您的问题。",
            sources=[],
            session_id=request.session_id or "session_1"
        )
    
    # 步骤 2: 构建提示词
    prompt = build_prompt(query, context)
    
    # 步骤 3: 调用大模型
    answer = call_deepseek(prompt)
    
    return ChatResponse(
        answer=answer,
        sources=sources,
        session_id=request.session_id or "session_1"
    )


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    流式聊天接口（支持打字机效果）
    """
    query = request.message.strip()
    
    if not query:
        raise HTTPException(status_code=400, detail="问题不能为空")
    
    # 检索上下文
    context, sources = retrieve_context(query)
    
    if not context:
        yield json.dumps({
            "answer": "抱歉，知识库中暂时没有相关资料可以回答您的问题。",
            "sources": []
        }, ensure_ascii=False)
        return
    
    # 构建提示词
    prompt = build_prompt(query, context)
    
    # 流式调用 DeepSeek
    if deepseek_client:
        try:
            response = deepseek_client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": "你是工业设备远程诊断专家，请基于提供的资料准确回答问题。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000,
                stream=True  # 启用流式输出
            )
            
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        
        except Exception as e:
            yield f"（调用失败）{str(e)}"
    else:
        yield "（Mock 回复）DeepSeek API Key 未配置"


# ==================== 启动入口 ====================
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # 开发模式：代码修改后自动重启
    )
