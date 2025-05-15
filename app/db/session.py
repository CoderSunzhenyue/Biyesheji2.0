# app/db/session.py

import asyncio
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import logging # 导入日志

from app.core.config import settings # 导入配置，需要你确保 app/core/config.py 存在且包含 settings 对象
from app.db.base import Base # 导入 Base 基类

logger = logging.getLogger(__name__) # 获取 logger 实例

# 数据库引擎创建
# settings.DATABASE_URL 应该是一个异步数据库连接字符串，例如 "mysql+aiomysql://user:password@host/db"
try:
    engine = create_async_engine(
        settings.DATABASE_URL,
        # echo=True, # 设置为 True 可以打印所有 SQL 语句，方便调试
        pool_pre_ping=True, # 检查连接是否仍然有效
        pool_recycle=3600 # 设置连接的回收时间（秒），避免连接长时间闲置后失效
    )
    logger.info("数据库引擎创建成功。")
except Exception as e:
    logger.error(f"数据库引擎创建失败: {e}", exc_info=True)
    # 引擎创建失败通常是致命错误，可以考虑在这里记录更严重的日志或在应用启动时检查

# 异步 session 工厂
async_session = sessionmaker(
    engine,
    class_=AsyncSession, # 指定 session 类为 AsyncSession
    expire_on_commit=False # 设置为 False 可以让对象在 commit 后不会过期，避免二次访问数据库
)
logger.info("数据库异步会话工厂创建成功。")

# 初始化数据库（创建表结构）
async def init_db():
    """
    初始化数据库，创建所有通过 Base 定义的表。
    这个函数通常在应用启动时调用一次。
    """
    logger.info("正在初始化数据库...")
    try:
        async with engine.begin() as conn:
            # 使用 run_sync 在异步连接中运行同步的 create_all
            await conn.run_sync(Base.metadata.create_all)
        logger.info("数据库初始化完成（表结构已创建或已存在）。")
    except Exception as e:
         logger.critical(f"数据库初始化失败: {e}", exc_info=True)
         # 数据库初始化失败是严重错误，可能需要阻止应用启动
         # raise RuntimeError(f"数据库初始化失败: {e}")


# 用于 FastAPI 依赖注入的函数
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    提供一个异步数据库会话给 FastAPI 依赖注入。
    确保在请求处理完成后会话被关闭。
    """
    async with async_session() as session:
        try:
            # yield 会话给请求处理函数使用
            yield session
        except Exception as e:
             # 可以捕获会话使用过程中的异常并处理
             logger.error(f"数据库会话使用过程中发生错误: {e}", exc_info=True)
             # 在这里可以根据需要进行会话回滚等操作
             await session.rollback() # 如果有未提交的事务，进行回滚
             raise # 重新抛出异常
        finally:
            # 确保会话最终被关闭，释放连接
            await session.close()
            # logger.debug("数据库会话已关闭。") # debug级别日志