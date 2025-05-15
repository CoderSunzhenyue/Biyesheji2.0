# app/api/deps.py

from fastapi import Depends, HTTPException, status # 导入 status
from fastapi.security import OAuth2PasswordBearer # 用于从请求头获取 token
from jose import JWTError, jwt # 用于处理 JWT
from sqlalchemy.ext.asyncio import AsyncSession # 导入异步数据库会话类型
from sqlalchemy import select # 用于数据库查询
import logging # 导入 logging
# 导入应用核心配置，用于获取 SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES 等
from app.core.config import settings

# 导入数据库会话获取函数
from app.db.session import get_db # 假设 get_db 函数在 app.db.session 中

# 导入用户模型，用于根据 token payload 查询用户
from app.db.models.user import User # 假设 User 模型在 app.db.models.user 中

# 导入安全相关的通用函数 (可选，如果需要在 deps.py 中使用，如 verify_token)
# from app.core.security import verify_token

# ====================== 获取 logger 实例 ======================
logger = logging.getLogger(__name__) # <-- 添加这行代码，获取 logger 实例

# ====================== 数据库会话依赖 (已在 app/db/session.py 中定义并导入) ======================
# 这里的 get_db 函数是从 app/db/session.py 中导入的，用于在 API 路由或服务层函数中注入数据库会话
# async def get_db() -> AsyncGenerator[AsyncSession, None]:
#     # 实现已经在 app/db/session.py 中
#     pass


# ====================== 认证相关的依赖 ======================

# 定义 OAuth2PasswordBearer 实例，告诉 FastAPI 如何从请求中提取 token (通常从 Authorization: Bearer 头部)
# tokenUrl 指向用于获取 token 的登录接口的 URL
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth.py/login") # <-- 请根据你的实际登录接口 URL 修改 tokenUrl


# 获取当前认证用户的依赖函数
async def get_current_user(
    # Depends(oauth2_scheme) 会尝试从请求头获取 token，如果失败（如没有 token），返回 401
    token: str = Depends(oauth2_scheme), # 从请求中获取 token
    db: AsyncSession = Depends(get_db) # 获取数据库会话依赖
) -> User:
    """
    FastAPI 依赖，用于获取当前认证用户的模型实例。
    从 JWT Token 中提取信息，查询数据库验证用户是否存在。
    """
    # 定义认证失败时抛出的 HTTPException
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, # 401 Unauthorized
        detail="无法验证身份凭证", # 认证失败的详细信息
        headers={"WWW-Authenticate": "Bearer"}, # 在响应头中添加认证方式提示
    )

    try:
        # 使用配置中的密钥和算法解码 JWT Token
        # 也可以调用 app.core.security 中的 verify_token 函数来处理解码和验证逻辑
        # payload = verify_token(token)
        # if payload is None:
        #     raise credentials_exception

        # 直接在这里解码 JWT
        # audience 和 issuer 等高级验证可根据需要添加
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
            # audience="your_api_audience" # 如果有 audience 验证
            # issuer="your_api_issuer" # 如果有 issuer 验证
        )

        # 从 payload 中提取用户唯一标识，通常是 'sub' (subject)
        # 在你的 JWT 创建时，payload 中应该包含了能唯一标识用户的信息，例如 username 或 user_id
        # 假设你在创建 token 时将 username 放到了 'sub' 字段
        username: str = payload.get("sub")
        # 假设你在创建 token 时将 user_id 放到了 'user_id' 字段
        # user_id: int = payload.get("user_id")

        if username is None:
            # 如果 token 解码成功但 payload 中没有预期的用户标识字段
            raise credentials_exception
        # 如果使用 user_id 作为标识，可以这样：
        # if user_id is None:
        #     raise credentials_exception

    except JWTError:
        # 如果 token 无效（签名错误、过期等），jwt.decode 会抛出 JWTError
        raise credentials_exception # 抛出认证失败异常
    except Exception as e:
        # 捕获解码或处理 payload 时的其他异常
        logger.error(f"解码或处理 JWT payload 时发生意外错误: {e}", exc_info=True)
        raise credentials_exception # 抛出认证失败异常


    # 根据从 token payload 中获取的用户标识查询数据库，验证用户是否存在
    # 如果你从 token 中获取的是 username
    result = await db.execute(select(User).where(User.username == username))
    # 如果你从 token 中获取的是 user_id
    # result = await db.execute(select(User).where(User.id == user_id))

    # result.scalars().first() 获取查询结果的第一个（也是唯一的）用户模型实例
    user = result.scalars().first()

    if user is None:
        # 如果根据 token 中的标识在数据库中找不到用户
        raise credentials_exception # 抛出认证失败异常

    # 如果用户存在，返回用户模型实例
    return user


# 获取当前认证用户 ID 的依赖函数
async def get_current_user_id(
    # 这个依赖函数直接依赖于 get_current_user 函数
    # get_current_user 会先被执行，如果成功，返回 User 实例，然后传递给这里的 current_user 参数
    current_user: User = Depends(get_current_user)
) -> int:
    """
    FastAPI 依赖，用于获取当前认证用户的 ID。
    依赖于 get_current_user 依赖函数。
    """
    # 从 get_current_user 返回的 User 模型实例中获取 ID 并返回
    return current_user.id

# --- 其他可能的依赖函数 (根据你实际项目需要) ---
# 例如：检查用户是否是管理员的依赖
# async def is_admin(current_user: User = Depends(get_current_user)) -> bool:
#     # return current_user.is_admin # 假设 User 模型有 is_admin 属性
#     pass

# 例如：获取系统配置的依赖 (如果不想直接导入 settings)
# from app.core.config import Settings
# async def get_settings() -> Settings:
#    return settings