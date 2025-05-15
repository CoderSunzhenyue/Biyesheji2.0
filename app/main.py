# app/main.py
import os
import logging
from pathlib import Path # 导入 Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
import uvicorn
import sys # 用于可能的应用退出

# --- 导入核心配置 ---
from app.core.config import settings # 假设你的配置在 app.core.config 中

# --- 导入数据库初始化和会话管理 ---
from app.db.session import init_db # 数据库初始化函数
# get_db 函数通过依赖注入使用，不需要在这里直接导入

# --- 导入应用启动时需要加载的资源初始化函数 ---
# 从 utils 层导入模型加载函数
from app.utils.classification import load_classification_model, bert_loading_error # 导入分类模型加载函数及其状态

# --- 导入各个模块的 API 路由 ---
# 路由文件现在放在 app/api/endpoints/ 目录下
from app.api.endpoints import detect # 导入图片检测路由
from app.api.endpoints import records # 导入记录管理路由 (假设你已将原 records 文件移动到这里)
from app.api.endpoints import user # 导入用户管理路由 (假设你已将原 user 文件移动到这里)
from app.api.endpoints import auth # 导入认证管理路由 (假设你已将原 auth.py 文件移动到这里)

# --- 导入其他可能的路由 (根据你实际项目需要) ---
# from app.api.endpoints import classification as document_classify_router # 导入文档分类路由 (假设已移动并重命名)
# from app.api.endpoints import audio_classify as audio_router # 导入音频分类路由 (假设已移动并重命名)


# ====================== 日志配置 (放在应用创建之前，确保尽早配置) ======================
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO if settings.ENVIRONMENT == "production" else logging.DEBUG # 根据环境设置日志级别
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO if settings.ENVIRONMENT == "production" else logging.DEBUG)


# ====================== FastAPI 应用创建 ======================
app = FastAPI(
    title=settings.PROJECT_NAME, # 从配置中读取项目名称
    version=settings.API_V1_STR, # 从配置中读取API版本号 (如果settings中有定义的话)
    # description="提供图像上传、敏感信息检测、历史记录等API服务", # 应用描述
    openapi_url=f"{settings.API_V1_STR}/openapi.json" # 如果你想为 API v1 设置特定的 openapi 路径
)

# ====================== 中间件配置 ======================

# ✅ CORS 设置
app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(origin) for origin in settings.ALLOWED_ORIGINS], # 确保 origins 是字符串列表
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================== 静态文件挂载 ======================
# ✅ 确保目录存在并在应用启动时挂载
# 获取当前文件 (main.py) 所在的目录
# current_dir = os.path.dirname(os.path.abspath(__file__)) # 或者直接使用 Path(__file__).parent
# 项目根目录 (假定 main.py 在 app 目录下)
# project_root = Path(__file__).parent.parent # 回退两级到项目根目录
# 假设 static 目录就在项目根目录
# static_dir = project_root / "static"

# 或者，更简洁的方式：直接根据 app 目录找到 static 目录
BASE_DIR = Path(__file__).resolve().parent.parent # app 目录的上级目录 (项目根目录)
STATIC_DIR = BASE_DIR / "static" # 项目根目录下的 static 目录

if not STATIC_DIR.exists():
    STATIC_DIR.mkdir(parents=True, exist_ok=True) # 确保目录存在

# 挂载静态文件目录，使其可以通过 /static 路径访问
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
logger.info(f"静态文件目录已挂载到 /static，指向: {STATIC_DIR}")


# ====================== 路由注册 ======================
# 将各个文件中的 APIRouter 实例包含到主应用中
# prefix="/api" 或 prefix="/api/v1"取决于你的API版本策略

# tags 用于在 API 文档中对接口进行分组

app.include_router(auth.router, prefix="/api",) # 认证路由通常有独立前缀
app.include_router(user.router, prefix="/api") # 用户路由通常有独立前缀
app.include_router(detect.router, prefix="/api") # 图片检测路由
app.include_router(records.router, prefix="/api") # 记录管理路由



# ====================== 应用启动事件 ======================
# 在应用启动时执行初始化任务，如数据库连接、模型加载等
@app.on_event("startup")
async def on_startup():
    logger.info("应用启动事件触发...")
    # 1. 初始化数据库连接
    await init_db()
    logger.info("数据库初始化完成。")

    # 2. 加载分类模型等核心资源
    logger.info("正在加载分类模型...")
    await load_classification_model()
    # 检查模型加载状态，如果失败，可以记录致命日志或终止应用启动
    if bert_loading_error:
         logger.critical(f"分类模块资源加载失败，应用无法正常工作: {bert_loading_error}")
         # 如果模型是核心功能且加载失败不可接受，可以考虑强制退出
         # logger.critical("强制退出应用由于关键资源加载失败。")
         # sys.exit(1) # 强制退出
    else:
         logger.info("分类模块资源加载完成。")



    logger.info("应用启动事件完成。")


# ====================== 应用关闭事件 (可选) ======================
@app.on_event("shutdown")
async def on_shutdown():
    logger.info("应用关闭事件触发...")
    # 在这里可以进行资源清理，如关闭数据库连接、模型卸载等
    logger.info("应用关闭事件完成。")


# ====================== 运行入口 (用于直接运行 main.py) ======================
# 注意：在生产环境中通常使用 gunicorn 或 uvicorn 直接通过命令行启动应用
# 例如：uvicorn app.main:app --host 0.0.0.0 --port 8000
# 这里的 if __name__ == "__main__": 块主要用于开发阶段方便地直接运行文件进行调试。
if __name__ == "__main__":
    # uvicorn.run() 的参数需要根据你的实际情况和需求进行调整
    # reload=True 方便开发调试，但生产环境不使用
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1", # 监听地址
        port=8000,       # 监听端口
        reload=True,     # 代码变更时自动重启 (仅开发环境使用)
        # log_level="info" # 可以通过这里或basicConfig设置日志级别
    )