# app/services/records.py

import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from sqlalchemy.orm import joinedload

from fastapi import HTTPException, status

from app.db.models.user import User
from app.db.models.ocr_record import DetectionRecord

from app.schemas.user import RecordDetail # 导入 RecordDetail Schema

# 导入 utils 层的文件系统操作函数
# 需要确保 app/utils/storage.py 存在并包含 delete_static_file 函数
from app.utils.storage import delete_static_file


logger = logging.getLogger(__name__)


# ====================== 获取用户历史检测记录服务函数 ======================
# 这个函数由 app/api/endpoints/records.py 中的 GET / 接口调用
async def list_user_records(db: AsyncSession, user_id: int) -> List[DetectionRecord]: # 返回 SQLAlchemy 模型列表
    """
    获取指定用户ID的所有历史检测记录。
    """
    logger.info(f"服务层：正在获取用户 {user_id} 的历史检测记录。")
    result = await db.execute(
        select(DetectionRecord)
        .where(DetectionRecord.user_id == user_id)
        .order_by(DetectionRecord.created_at.desc())
        # .options(joinedload(DetectionRecord.user)) # 如果需要加载关联的用户信息
    )
    records = result.scalars().all()

    logger.info(f"服务层：用户 {user_id} 共找到 {len(records)} 条历史记录。")

    return records # 返回 DetectionRecord 模型实例列表


# ====================== 删除指定记录服务函数 ======================
# 这个函数由 app/api/endpoints/records.py 中的 DELETE /{record_id} 接口调用
async def delete_record_by_id(db: AsyncSession, user_id: int, record_id: int) -> bool:
    """
    删除指定用户ID下，ID为 record_id 的检测记录。
    Returns:
        bool: 如果记录存在且属于该用户并被删除返回 True，否则返回 False。
    """
    logger.info(f"服务层：用户 {user_id} 正在尝试删除记录ID: {record_id}")
    # 1. 查询记录是否存在且属于当前用户
    result = await db.execute(
        select(DetectionRecord)
        .where(DetectionRecord.id == record_id, DetectionRecord.user_id == user_id)
    )
    record = result.scalars().first()

    if not record:
        logger.warning(f"服务层：用户 {user_id} 未找到记录ID {record_id} 或其不属于该用户。")
        return False

    # 2. 删除数据库中的记录
    try:
        await db.delete(record)
        await db.commit()
        logger.info(f"服务层：用户 {user_id} 已删除记录ID: {record_id}。")
        return True
    except Exception as e:
        logger.error(f"服务层：删除记录ID {record_id} 失败: {e}", exc_info=True)
        await db.rollback()
        raise RuntimeError(f"删除记录失败: {record_id}")


# ====================== 删除指定文件名所有记录服务函数 ======================
# 这个函数由 app/api/endpoints/records.py 中的 DELETE / 接口调用
async def delete_records_by_filename(db: AsyncSession, user_id: int, filename: str):
    """
    删除指定用户ID下，文件名为 filename 的所有检测记录，并删除对应的图片文件。
    Returns:
        dict: 包含删除结果（删除数量，文件是否删除）的字典。
    """
    logger.info(f"服务层：用户 {user_id} 正在尝试删除文件名 '{filename}' 的所有记录。")

    # 1. 获取要删除记录的图片相对 URL (假设 DetectionRecord 模型的 saved_path 字段存储的是 URL)
    # 需要先查询一条记录来获取图片的 saved_path，假设同一个 filename 对应的 saved_path 是一样的
    # 如果没有记录，这里返回 None，后面也不会删除文件
    path_result = await db.execute(
        select(DetectionRecord.saved_path)
        .where(DetectionRecord.filename == filename, DetectionRecord.user_id == user_id)
        .limit(1)
    )
    image_relative_url = path_result.scalars().first()

    # 2. 删除数据库中的记录
    # 使用 execute(delete(...)) 更直接
    delete_statement = delete(DetectionRecord).where(
        DetectionRecord.filename == filename,
        DetectionRecord.user_id == user_id
    )
    delete_result = await db.execute(delete_statement)
    await db.commit()

    deleted_count = delete_result.rowcount # 获取删除的行数

    logger.info(f"服务层：用户 {user_id} 已删除文件名 '{filename}' 的 {deleted_count} 条记录。")

    file_deleted = False
    # 3. 删除服务器上的图片文件 (如果找到了对应的图片路径)
    if image_relative_url:
        # 调用 utils/storage.py 中的异步删除函数
        # delete_static_file 返回布尔值表示是否成功删除
        file_deleted = await delete_static_file(image_relative_url)
        if file_deleted:
            logger.info(f"服务层：已删除文件名 '{filename}' 对应的图片文件。")
        else:
            logger.warning(f"服务层：未找到或无法删除文件名 '{filename}' 对应的图片文件: {image_relative_url}")

    # 返回包含删除结果的字典
    return {
        "deleted_count": deleted_count,
        "file_deleted": file_deleted
    }


# --- 其他可能的记录相关的服务函数 (根据你实际项目需要) ---
# 例如：获取单条记录详情的服务函数 (按ID)
# async def get_record_detail_by_id(db: AsyncSession, user_id: int, record_id: int) -> Optional[DetectionRecord]:
#    # 查询记录确保属于用户
#    # 返回记录模型或 None
#    pass