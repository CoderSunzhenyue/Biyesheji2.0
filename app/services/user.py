# app/services/user.py

import logging # 导入日志
from typing import List, Optional, Dict, Any # 导入 Python 标准库类型提示
from sqlalchemy.ext.asyncio import AsyncSession # 导入异步数据库会话
from sqlalchemy import select # 导入 select 语句
from sqlalchemy.orm import joinedload # 用于关系加载

from fastapi import HTTPException, status # 导入 HTTPException 和 status

# 导入数据库模型
from app.db.models.user import User # 导入 User 模型
from app.db.models.ocr_record import DetectionRecord # 导入 DetectionRecord 模型

# 导入 schemas 定义的数据结构 (用于函数参数类型提示或返回值类型提示)
from app.schemas.user import ( # <-- 从 schemas 导入需要的 Schema
    UserInfo,
    ChangePasswordRequest,
    UserUpdate,
    # RecordOut # 这个 Schema 现在改名为 RecordDetail 并主要在 API 层使用
    # UserCreate # 用于注册服务函数，如果在这个文件实现注册，需要导入
    # RecordDetail # 如果服务函数需要返回这个 Schema 实例，需要导入
)

# 导入核心安全相关的函数 (密码哈希、验证等)
from app.core.security import verify_password, get_password_hash # 导入安全函数

# 导入其他可能的依赖或工具函数
# import datetime # 如果函数中需要使用 datetime


logger = logging.getLogger(__name__) # 获取 logger 实例


# ====================== 获取当前用户信息服务函数 ======================
# 这个函数由 app/api/endpoints/user.py 中的 /me 接口调用
async def get_user_info(db: AsyncSession, current_user: User) -> UserInfo:
    """
    获取当前用户的详细信息。
    Args:
        db: 数据库异步会话。
        current_user: 通过依赖注入获取的当前认证用户模型实例。
    Returns:
        用户信息 Schema (UserInfo)。
    Raises:
        HTTPException: 如果用户未找到（理论上不应发生，因为依赖已验证）。
    """
    logger.info(f"服务层：正在获取用户 {current_user.id} 的详细信息。")
    # 使用 select 语句查询用户
    # 如果 UserInfo Schema 只需要 username, email, phone 这些字段，不需要加载 records
    # result = await db.execute(select(User).where(User.id == current_user.id))

    # 如果 UserInfo Schema 包含需要从关联关系中获取的信息 (例如你之前的 create_time 可能来自 records)，
    # 并且 UserInfo 已经调整为可以直接从 User 模型创建 (from_attributes=True)，
    # 那么在这里加载关联关系是必要的。
    # 如果 UserInfo 只需要 User 模型自身的字段，则无需 joinedload。
    result = await db.execute(
        select(User)
        .where(User.id == current_user.id)
        # .options(joinedload(User.records)) # 只有当 UserInfo 需要访问 User.records 中的字段时才需要加载
    )
    user = result.scalars().first()

    if user is None:
        logger.error(f"服务层：通过依赖注入获取的用户ID {current_user.id} 在数据库中未找到。")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="用户未找到") # 抛出 404 错误

    # 构建 UserInfo Schema 返回
    # UserInfo 应该设置了 orm_mode=True 或 from_attributes=True，
    # 这样可以直接返回 SQLAlchemy 模型实例，FastAPI 会自动序列化。
    # 如果 UserInfo 不需要从 records 获取信息，而 User 模型本身有这些字段，直接返回 user 模型即可。
    logger.info(f"服务层：成功获取用户 {current_user.id} 的信息。")
    return UserInfo.model_validate(user) # Pydantic v2 推荐用法
    # return UserInfo.from_orm(user) # Pydantic v1 用法
    # 或者如果直接返回 user 模型，FastAPI 也会尝试根据 response_model 序列化


# ====================== 用户修改密码服务函数 ======================
# 这个函数由 app/api/endpoints/user.py 中的 /change-password 接口调用
async def change_user_password(
    db: AsyncSession,
    current_user: User, # 接收通过依赖注入获取的当前用户模型实例
    req: ChangePasswordRequest # 接收修改密码请求 Schema
):
    """
    修改当前用户的密码。
    Args:
        db: 数据库异步会话。
        current_user: 当前认证用户模型实例。
        req: 修改密码请求 Schema。
    Raises:
        HTTPException: 如果旧密码错误或新密码不一致。
        RuntimeError: 如果数据库操作或哈希密码失败。
    """
    logger.info(f"服务层：用户 {current_user.id} 正在尝试修改密码。")
    # 1. 验证旧密码
    # 调用 core.security 中的函数验证旧密码
    if not verify_password(req.old_password, current_user.hashed_password):
        logger.warning(f"服务层：用户 {current_user.id} 修改密码时旧密码错误。")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="旧密码错误") # 抛出业务错误

    # 2. 检查新密码与确认密码是否一致
    if req.new_password != req.confirm_password:
        logger.warning(f"服务层：用户 {current_user.id} 修改密码时新密码与确认密码不一致。")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="新密码和确认密码不一致") # 抛出业务错误

    # 3. 对新密码进行哈希
    try:
        hashed_password = get_password_hash(req.new_password)
    except ValueError as e: # 捕获哈希失败的异常
         logger.error(f"服务层：用户 {current_user.id} 哈希新密码失败: {e}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="处理密码时出错")


    # 4. 更新数据库中的用户记录
    current_user.hashed_password = hashed_password
    # current_user 已经被添加到会话中（因为它来自 Depends），SQLAlchemy 会跟踪其修改
    # db.add(current_user) # 这一行通常不是必需的，因为对象已经在会话中且被修改
    try:
        await db.commit() # 提交事务到数据库
        # await db.refresh(current_user) # 如果需要刷新对象状态（通常修改密码不需要）

        logger.info(f"服务层：用户 {current_user.id} 密码修改成功。")
    except Exception as e: # 捕获数据库操作异常
        logger.error(f"服务层：用户 {current_user.id} 保存新密码到数据库失败: {e}", exc_info=True)
        await db.rollback() # 回滚事务
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="保存新密码失败")


# ====================== 用户修改信息（用户名 / 邮箱 / 手机号）服务函数 ======================
# 这个函数由 app/api/endpoints/user.py 中的 /update-info 接口调用
async def update_user_info(
    db: AsyncSession,
    current_user: User, # 接收通过依赖注入获取的当前用户模型实例
    update_data: UserUpdate # 接收用户更新信息请求 Schema
) -> User: # 返回更新后的用户模型实例
    """
    更新当前用户的个人信息。
    Args:
        db: 数据库异步会话。
        current_user: 当前认证用户模型实例。
        update_data: 用户更新信息请求 Schema (包含要更新的字段)。
    Returns:
        更新后的用户模型实例。
    Raises:
        HTTPException: 如果更新的用户名、邮箱或手机号已存在。
        RuntimeError: 如果数据库操作失败。
    """
    logger.info(f"服务层：用户 {current_user.id} 正在尝试更新个人信息。")
    # 1. 检查是否有需要更新的字段
    # Pydantic v2: model_dump(exclude_unset=True)
    # Pydantic v1: update_data.dict(exclude_unset=True)
    update_dict = update_data.model_dump(exclude_unset=True) # Pydantic v2 示例

    if not update_dict:
         # 如果 UserUpdate 中的所有字段都是 None，表示没有提供更新信息
         logger.info(f"服务层：用户 {current_user.id} 尝试更新信息但未提供任何有效字段。")
         return current_user # 直接返回当前用户，不进行数据库操作


    # 2. 检查唯一性冲突 (用户名、邮箱、手机号)
    # 如果更新了用户名
    if 'username' in update_dict and update_dict['username'] != current_user.username:
        result = await db.execute(select(User).where(User.username == update_dict['username']).where(User.id != current_user.id))
        if result.scalars().first():
            logger.warning(f"服务层：用户 {current_user.id} 更新用户名失败，用户名 '{update_dict['username']}' 已存在。")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="用户名已存在")
    # 如果更新了邮箱
    if 'email' in update_dict and update_dict['email'] is not None and update_dict['email'] != current_user.email:
         result = await db.execute(select(User).where(User.email == update_dict['email']).where(User.id != current_user.id))
         if result.scalars().first():
             logger.warning(f"服务层：用户 {current_user.id} 更新邮箱失败，邮箱 '{update_dict['email']}' 已存在。")
             raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="邮箱已存在")
    # 如果更新了手机号
    if 'phone' in update_dict and update_dict['phone'] is not None and update_dict['phone'] != current_user.phone:
         result = await db.execute(select(User).where(User.phone == update_dict['phone']).where(User.id != current_user.id))
         if result.scalars().first():
             logger.warning(f"服务层：用户 {current_user.id} 更新手机号失败，手机号 '{update_dict['phone']}' 已存在。")
             raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="手机号已存在")


    # 3. 更新用户模型字段
    for field, value in update_dict.items():
        setattr(current_user, field, value)

    # 4. 保存到数据库
    # db.add(current_user) # 对象已在会话中，无需再次 add
    try:
        await db.commit()
        await db.refresh(current_user) # 刷新对象，获取数据库中的最新状态

        logger.info(f"服务层：用户 {current_user.id} 信息修改成功，更新字段: {list(update_dict.keys())}")
    except Exception as e: # 捕获数据库操作异常
        logger.error(f"服务层：用户 {current_user.id} 保存更新信息到数据库失败: {e}", exc_info=True)
        await db.rollback() # 回滚事务
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="保存用户信息失败")


    return current_user # 返回更新后的用户模型实例


# ====================== 获取用户历史检测记录服务函数 ======================
# 这个函数由 app/api/endpoints/user.py 中的 /records 接口调用
async def list_user_records(db: AsyncSession, current_user: User) -> List[DetectionRecord]:
    """
    获取当前用户的所有历史检测记录。
    Args:
        db: 数据库异步会话。
        current_user: 当前认证用户模型实例。
    Returns:
        DetectionRecord 模型实例列表。
    """
    logger.info(f"服务层：正在获取用户 {current_user.id} 的历史检测记录。")
    # 使用 select 语句查询 DetectionRecord 表
    # 直接通过 User 模型的 records 关系加载是更 SQLAlchemy 的方式，但需要确保 User 模型已加载且 records 关系已加载
    # 或者直接查询 DetectionRecord 表，按 user_id 过滤
    result = await db.execute(
        select(DetectionRecord)
        .where(DetectionRecord.user_id == current_user.id) # 按用户ID过滤
        .order_by(DetectionRecord.created_at.desc()) # 按创建时间倒序
        # .options(joinedload(DetectionRecord.user)) # 如果需要加载关联的用户信息
    )
    records = result.scalars().all() # 获取所有结果行的 DetectionRecord 实例列表

    logger.info(f"服务层：用户 {current_user.id} 共找到 {len(records)} 条历史记录。")

    return records # 返回 DetectionRecord 模型实例列表


# --- 其他可能的业务逻辑函数 (根据你实际项目需要) ---
# 例如：删除指定记录的服务函数 (也可以放在 services/records.py 中)
# async def delete_user_record(db: AsyncSession, user_id: int, record_id: int):
#     """删除用户指定的记录"""
#     # ... 调用 db 操作删除记录 ...
#     pass

# 例如：用户注册的服务函数 (如果不在 auth.py 中)
# async def register_user_service(db: AsyncSession, user_create: UserCreate):
#    # ... 注册逻辑 ...
#    pass

# 例如：用户认证的服务函数 (如果不在 auth.py 中)
# async def authenticate_user_service(db: AsyncSession, username: str, password: str) -> Optional[User]:
#    # ... 认证逻辑 ...
#    pass