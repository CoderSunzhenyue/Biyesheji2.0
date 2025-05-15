# app/services/auth.py

import logging # 导入日志
from typing import List, Optional # 导入类型提示
from fastapi import HTTPException, status # 导入 HTTPException 和 status
from sqlalchemy.ext.asyncio import AsyncSession # 导入异步数据库会话
from sqlalchemy import select # 导入 select 语句

# 导入数据库模型
from app.db.models.user import User # 导入 User 模型

# 导入 schemas 定义的数据结构 (用于函数参数类型提示)
from app.schemas.user import UserCreate # 导入用户创建 Schema

# 导入核心安全相关的函数 (密码哈希、验证等)
from app.core.security import verify_password, get_password_hash # 导入安全函数


logger = logging.getLogger(__name__) # 获取 logger 实例

# ====================== 用户注册服务函数 ======================
async def register_user(db: AsyncSession, user_in: UserCreate):
    """
    用户注册服务函数。
    Args:
        db: 数据库异步会话。
        user_in: 用户注册请求 Schema (包含用户名和密码)。
    Raises:
        HTTPException: 如果用户名已被注册。
    """
    logger.info(f"服务层：正在注册用户 {user_in.username}")
    # 1. 检查用户名是否已存在
    result = await db.execute(select(User).where(User.username == user_in.username))
    existing_user = result.scalars().first()
    if existing_user:
        logger.warning(f"服务层：用户名 {user_in.username} 已存在，注册失败。")
        # 抛出 HTTPException，API 层会捕获并返回错误响应
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="用户名已注册")

    # 2. 对密码进行哈希
    hashed_password = get_password_hash(user_in.password)

    # 3. 创建用户模型实例
    user = User(username=user_in.username, hashed_password=hashed_password)

    # 4. 添加到数据库会话并提交
    try:
        db.add(user)
        await db.commit() # 提交事务
        # await db.refresh(user) # 如果需要获取新用户的 ID 或其他默认值

        logger.info(f"服务层：用户 {user_in.username} 注册成功。用户ID: {user.id}")
    except Exception as e:
        logger.error(f"服务层：保存用户到数据库失败: {user_in.username}, 错误: {e}", exc_info=True)
        await db.rollback() # 回滚事务
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="注册失败，请稍后再试")


# ====================== 用户认证服务函数 ======================
async def authenticate_user(db: AsyncSession, username: str, password: str) -> Optional[User]:
    """
    用户认证服务函数。
    验证用户名和密码是否匹配数据库中的用户。
    Args:
        db: 数据库异步会话。
        username: 待认证的用户名。
        password: 待认证的密码。
    Returns:
        如果认证成功，返回用户模型实例；否则返回 None。
    Raises:
        HTTPException: 如果用户名或密码不正确（或者直接返回 None 让调用者判断）。
                       这里根据你原代码逻辑，选择抛出 HTTPException。
    """
    logger.info(f"服务层：正在认证用户 {username}")
    # 1. 根据用户名查询用户
    result = await db.execute(select(User).where(User.username == username))
    user = result.scalars().first()

    # 2. 验证用户是否存在且密码是否正确
    # 调用 core.security 中的函数验证密码
    if not user or not verify_password(password, user.hashed_password):
        logger.warning(f"服务层：用户 {username} 认证失败，用户名不存在或密码错误。")
        # 根据你原代码逻辑，这里抛出 HTTPException
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码不正确") # 401 Unauthorized

    logger.info(f"服务层：用户 {username} 认证成功。")

    # 3. 认证成功，返回用户模型实例
    return user

# --- 其他可能的认证相关的服务函数 (根据你实际项目需要) ---
# 例如：通过邮箱认证用户
# async def authenticate_user_by_email(db: AsyncSession, email: str, password: str) -> Optional[User]:
#     # ... 类似 authenticate_user 的逻辑，但按邮箱查询
#     pass