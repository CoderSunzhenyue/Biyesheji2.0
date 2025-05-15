# app/api/endpoints/auth.py.py

import logging # 导入日志
from fastapi import APIRouter, Depends, HTTPException, Form, status # 导入 status
from sqlalchemy.ext.asyncio import AsyncSession

# 导入 schemas 定义的请求和响应模型
from app.schemas.user import UserCreate, TokenOut # 导入用户相关的 Schema

# 导入服务层编写的业务逻辑函数
from app.services.auth import register_user, authenticate_user # 导入认证相关的服务函数

# 导入核心安全相关的函数 (创建 token)
from app.core.security import create_access_token # 导入创建 token 函数

# 导入数据库会话获取函数
from app.db.session import get_db # 导入 get_db 依赖函数

# 导入用户模型 (如果服务层函数或依赖需要类型提示)
# from app.db.models.user import User


logger = logging.getLogger(__name__) # 获取 logger 实例

# 创建 APIRouter 实例，用于定义认证相关的路由
# prefix="/auth.py" 将使这个文件中定义的所有路由都以 /auth.py 开头
router = APIRouter(prefix="/auth", tags=["Authentication"]) # tags 用于 API 文档分组

# ====================== API 端点：用户注册接口 ======================
@router.post(
    "/register", # 完整的 URL 将是 /auth.py/register
    # response_model=UserInfo, # 注册成功后可以选择返回用户信息 Schema
    status_code=status.HTTP_201_CREATED, # 注册成功通常返回 201 Created
    summary="用户注册", # API 文档摘要
    description="接收用户名和密码，创建新用户。", # API 文档描述
)
async def api_register(
    # 通过 Body(...) 获取请求体，自动验证是否符合 UserCreate Schema
    user_in: UserCreate,
    # 通过 Depends(get_db) 依赖注入获取异步数据库会话
    db: AsyncSession = Depends(get_db)
):
    """
    API 层处理用户注册请求：调用服务层注册用户。
    """
    logger.info(f"API层接收到用户注册请求，用户名: {user_in.username}")
    try:
        # 调用服务层编写的注册业务逻辑函数
        await register_user(db, user_in)
        logger.info(f"API层成功处理用户 {user_in.username} 注册请求。")
        return {"message": "用户注册成功"} # 返回成功信息

    # 捕获服务层可能抛出的业务异常，如用户名已存在
    except HTTPException as e:
         logger.warning(f"API层注册用户失败: {user_in.username}, 错误: {e.detail}")
         raise e # 向上抛出 HTTPException
    except Exception as e: # 捕获其他未知异常
         logger.error(f"API层处理用户注册请求时发生未知错误: {user_in.username}, 错误: {e}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"用户注册失败: {e}")


# ====================== API 端点：用户登录接口 ======================
@router.post(
    "/login", # 完整的 URL 将是 /auth.py/login
    response_model=TokenOut, # 指定响应数据的 Pydantic 模型 (TokenOut)
    summary="用户登录", # API 文档摘要
    description="接收用户名和密码，验证身份并返回 Access Token。", # API 文档描述
)
async def api_login(
    # 通过 Form(...) 获取表单数据，用于接收用户名和密码
    username: str = Form(..., description="用户名"),
    password: str = Form(..., description="密码"),
    # 通过 Depends(get_db) 依赖注入获取异步数据库会话
    db: AsyncSession = Depends(get_db)
):
    """
    API 层处理用户登录请求：验证用户身份并生成 JWT Token。
    """
    logger.info(f"API层接收到用户登录请求，用户名: {username}")
    try:
        # 调用服务层编写的身份验证业务逻辑函数
        # authenticate_user 函数会验证用户名和密码，并返回用户模型实例或 None
        user = await authenticate_user(db, username, password)
        if not user:
            # 如果用户验证失败（用户名不存在或密码错误），服务层应返回 None 或抛出异常
            # 这里抛出 401 Unauthorized 错误
            logger.warning(f"API层用户登录失败：用户名 {username} 验证失败。")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="用户名或密码不正确",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # 调用 core.security 中的函数创建 JWT Access Token
        # 将用户的唯一标识（如 username 或 id）放入 token 的 payload 中 (通常用 'sub' 字段)
        # 例如: {"sub": user.username} 或 {"sub": str(user.id)}
        token_payload = {"sub": user.username} # 示例：使用 username 作为标识
        token = create_access_token(token_payload)

        logger.info(f"API层用户 {username} 登录成功，生成并返回 Token。")
        # 返回 TokenOut Schema 实例
        return {"access_token": token, "token_type": "bearer"}

    except Exception as e: # 捕获其他未知异常
         logger.error(f"API层处理用户登录请求时发生未知错误: {username}, 错误: {e}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"用户登录失败: {e}")