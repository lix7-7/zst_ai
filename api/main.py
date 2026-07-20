"""
智扫通 FastAPI 应用入口
启动: uvicorn api.main:app --reload
文档: http://localhost:8000/docs
"""
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from api.routers import chat, knowledge
from api.schemas.chat import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    from utils.logger_handler import logger

    logger.info("[FastAPI] 智扫通 Agent API 启动中...")

    # 初始化记忆系统（Redis 不可用时降级运行）
    try:
        from memory.manager import MemoryManager
        mgr = MemoryManager.get_instance()
        await mgr.initialize()
        app.state.memory_manager = mgr
        logger.info("[FastAPI] 记忆系统初始化完成")
    except Exception as e:
        logger.warning(f"[FastAPI] 记忆系统初始化失败（不影响核心功能）: {str(e)}")
        app.state.memory_manager = None

    # 预加载 Agent 实例（确认各项依赖正常）
    from agent.react_agent import ReactAgent
    app.state.agent = ReactAgent()
    logger.info("[FastAPI] Agent 实例初始化完成")

    yield

    # 关闭时清理
    if app.state.memory_manager:
        try:
            await app.state.memory_manager.close()
        except Exception:
            pass
    logger.info("[FastAPI] 智扫通 Agent API 关闭")


app = FastAPI(
    title="智扫通 Agent API",
    description="基于 LangChain ReAct Agent + RAG 的扫地机器人智能客服 API",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# 注册 API 路由
app.include_router(chat.router, prefix="/api/v1")
app.include_router(knowledge.router, prefix="/api/v1")


@app.get("/")
async def index():
    """聊天界面首页"""
    return FileResponse(static_dir / "index.html")


@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    return HealthResponse(status="ok", version="2.0.0")
