# app/api/endpoints/user.py

import logging # 导入日志
from typing import List # 导入 List
from fastapi import APIRouter, Depends, HTTPException, Query, status, Body  # 导入 FastAPI 核心组件和 Query, status
from sqlalchemy.ext.asyncio import AsyncSession # 导入异步数据库会话类型

# 导入服务层编写的业务逻辑函数
from app.services.user import ( # <-- 从 services 导入业务逻辑函数
    get_user_info,
    change_user_password,
    update_user_info,
    list_user_records, # list_user_records 返回的是 SQLAlchemy 模型列表
    # get_record_detail_by_id # 如果有获取详情服务函数
)

# 导入依赖函数 (获取DB会话, 获取当前认证用户)
# 假设这些依赖函数在 app/api/deps.py 中
from app.api.deps import get_db, get_current_user # <-- 从 api.deps 导入依赖函数

# 导入 schemas 定义的请求和响应模型
from app.schemas.user import ( # <-- 从 schemas 导入 Pydantic 模型
    UserInfo, # 用于 /me 接口的响应模型
    ChangePasswordRequest, # 用于 /change-password 接口的请求模型
    UserUpdate, # 用于 /update-info 接口的请求模型
    RecordDetail, # <-- 将这里的 RecordOut 改为 RecordDetail
    # UserCreate # 如果有注册接口需要 UserCreate schema
    # TokenOut # 如果有登录接口需要 TokenOut schema
)

# 导入数据库模型 (API 层可能需要 User 模型的类型提示)
from app.db.models.user import User # <-- 导入 User 模型

logger = logging.getLogger(__name__) # 获取 logger 实例

# 创建 APIRouter 实例，用于定义用户相关的路由
router = APIRouter(prefix="/user", tags=["User Management"]) # tags 用于 API 文档分组


# ====================== API 端点：获取当前用户信息 ======================
@router.get(
    "/me",
    response_model=UserInfo, # 指定响应数据的 Pydantic 模型
    summary="获取当前认证用户信息",
    description="需要有效的 Access Token。返回当前登录用户的基本信息。",
)
async def api_get_user_info(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    API 层处理获取当前用户信息请求：调用服务层获取信息并返回。
    """
    logger.info(f"API层接收到获取用户 {current_user.id} 信息请求。")
    try:
        user_info = await get_user_info(db, current_user)
        logger.info(f"API层成功获取用户 {current_user.id} 信息。")
        return user_info

    except Exception as e:
         logger.error(f"API层处理获取用户 {current_user.id} 信息请求时出错: {e}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"获取用户信息失败: {e}")


# ====================== API 端点：修改用户密码 ======================
@router.post(
    "/change-password",
    summary="修改当前用户密码",
    description="需要有效的 Access Token。接收旧密码和新密码，修改当前用户的密码。",
)
async def api_change_password(
    request: ChangePasswordRequest = Body(..., description="旧密码、新密码和确认密码"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    API 层处理修改密码请求：调用服务层修改密码。
    """
    logger.info(f"API层接收到用户 {current_user.id} 修改密码请求。")
    try:
        await change_user_password(db, current_user, request)
        logger.info(f"API层成功处理用户 {current_user.id} 修改密码请求。")
        return {"message": "密码修改成功"}

    except HTTPException as e:
         logger.warning(f"API层修改用户 {current_user.id} 密码失败: {e.detail}")
         raise e
    except Exception as e:
         logger.error(f"API层处理修改用户 {current_user.id} 密码请求时发生未知错误: {e}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"修改密码失败: {e}")


# ====================== API 端点：修改用户信息 ======================
@router.put(
    "/update-info",
    response_model=UserInfo, # 返回更新后的用户信息 Schema
    summary="修改当前用户信息",
    description="需要有效的 Access Token。接收要更新的用户信息（用户名、邮箱、手机号）。",
)
async def api_update_user_info(
    user_update: UserUpdate = Body(..., description="要更新的用户信息"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    API 层处理修改用户信息请求：调用服务层更新信息并返回更新后的用户。
    """
    logger.info(f"API层接收到用户 {current_user.id} 修改信息请求。")
    try:
        updated_user = await update_user_info(db, current_user, user_update)
        logger.info(f"API层成功处理用户 {current_user.id} 修改信息请求。")
        # 服务层返回的是 SQLAlchemy 模型实例，这里 FastAPI 会自动通过 response_model 序列化为 UserInfo
        return updated_user

    except Exception as e:
         logger.error(f"API层处理用户 {current_user.id} 修改信息请求时出错: {e}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"修改用户信息失败: {e}")


# ====================== API 端点：获取用户历史检测记录列表 ======================
@router.get(
    "/records",
    response_model=List[RecordDetail], # <-- 将这里的 RecordOut 改为 RecordDetail
    summary="获取当前用户历史检测记录列表",
    description="需要有效的 Access Token。返回当前用户的所有检测历史记录列表。",
)
async def api_list_user_records(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    API 层处理获取历史记录请求：调用服务层获取记录列表并返回。
    """
    logger.info(f"API层接收到获取用户 {current_user.id} 历史记录请求。")
    try:
        # list_user_records 服务层返回的是 DetectionRecord 模型实例列表
        records_list = await list_user_records(db, current_user)
        logger.info(f"API层成功获取用户 {current_user.id} 的 {len(records_list)} 条历史记录。")
        # FastAPI 会自动将 SQLAlchemy 模型实例列表序列化为 List[RecordDetail] Schema
        return records_list

    except Exception as e:
         logger.error(f"API层处理获取用户 {current_user.id} 历史记录请求时出错: {e}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"获取历史记录失败: {e}")


# --- 其他可能的 API 路由 (根据你实际项目需要) ---
# 例如：用户注册接口 (如果在 auth.py 中没有)
# @router.post("/register", ...)
# async def api_register(...):
#    # 调用服务层注册用户
#    # ...

# 例如：用户登录接口 (如果在 auth.py 中没有)
# @router.post("/login", ...)
# async def api_login(...):
#    # 调用服务层验证用户和密码
#    # ...