# app/api/endpoints/records.py

import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.records import (
    list_user_records,
    delete_record_by_id,
    delete_records_by_filename,
    # get_record_detail_by_id # 如果有获取详情服务函数
)

from app.api.deps import get_db, get_current_user_id

from app.schemas.user import RecordDetail # <-- 导入 RecordDetail Schema
# from app.schemas.user import DeleteRecordResponse # 如果有更详细的删除响应 Schema

logger = logging.getLogger(__name__)

# 创建 APIRouter 实例
router = APIRouter(prefix="/records", tags=["Records Management"])


# ====================== API 端点：获取用户历史检测记录列表 ======================
@router.get(
    "/",
    response_model=List[RecordDetail], # <-- 使用 RecordDetail 作为响应模型
    summary="获取当前用户历史检测记录列表",
    description="需要有效的 Access Token。返回当前用户的所有检测历史记录列表。",
)
async def api_list_my_records(
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    """
    API 层处理获取历史记录请求：调用服务层获取记录列表并返回。
    """
    logger.info(f"API层接收到获取用户 {user_id} 历史记录列表请求。")
    try:
        records_list = await list_user_records(db, user_id)
        logger.info(f"API层成功获取用户 {user_id} 的 {len(records_list)} 条历史记录。")
        # FastAPI 会自动将 SQLAlchemy 模型实例列表序列化为 List[RecordDetail] Schema
        return records_list

    except Exception as e:
         logger.error(f"API层处理获取用户 {user_id} 历史记录请求时出错: {e}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"获取历史记录失败: {e}")


# ====================== API 端点：删除指定记录 (按 ID) ======================
@router.delete(
    "/{record_id}",
    status_code=status.HTTP_200_OK,
    # response_model=DeleteRecordResponse, # 如果有详细的删除响应 Schema
    summary="删除指定的历史检测记录 (按记录ID)",
    description="需要有效的 Access Token。删除当前用户指定ID的检测记录。",
)
async def api_delete_record_by_id(
    record_id: int,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    """
    API 层处理删除记录请求：调用服务层删除记录。
    """
    logger.info(f"API层接收到用户 {user_id} 删除记录 ID {record_id} 的请求。")
    try:
        deleted_successfully = await delete_record_by_id(db, user_id, record_id)

        if not deleted_successfully:
             logger.warning(f"API层：用户 {user_id} 尝试删除记录 ID {record_id} 失败，记录未找到或不属于该用户。")
             raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="记录未找到或不属于当前用户")

        logger.info(f"API层成功处理用户 {user_id} 删除记录 ID {record_id} 的请求。")
        return {"message": "记录删除成功"}

    except RuntimeError as e:
         logger.error(f"API层处理删除记录 ID {record_id} 请求时服务层发生运行时错误: {e}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"删除记录失败: {e}")
    except Exception as e:
         logger.error(f"API层处理删除记录 ID {record_id} 请求时发生未知错误: {e}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"删除记录失败: {e}")


# ====================== API 端点：删除指定文件名所有记录 (按文件名) ======================
@router.delete(
    "/",
    status_code=status.HTTP_200_OK,
    # response_model=DeleteRecordResponse, # 如果有详细的删除响应 Schema
    summary="删除指定图片文件名的所有历史检测记录",
    description="需要有效的 Access Token。删除当前用户指定图片文件名的所有检测记录，并删除对应的图片文件。",
)
async def api_delete_records_by_filename(
    filename: str = Query(..., description="要删除记录的图片文件名"),
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    """
    API 层处理按文件名删除记录请求：调用服务层删除记录和文件。
    """
    logger.info(f"API层接收到用户 {user_id} 删除文件名 '{filename}' 所有记录的请求。")
    try:
        delete_result = await delete_records_by_filename(db, user_id, filename)

        if delete_result["deleted_count"] == 0:
             logger.warning(f"API层：用户 {user_id} 尝试删除文件名 '{filename}' 的记录但未找到任何记录。")
             return {"message": f"未找到与文件名 '{filename}' 关联的记录或文件，未执行删除操作。", "deleted_count": 0, "file_deleted": False}


        logger.info(f"API层成功处理用户 {user_id} 删除文件名 '{filename}' 所有记录的请求。")

        return {"message": f"文件名 '{filename}' 的记录已删除。", **delete_result}

    except Exception as e:
         logger.error(f"API层处理删除文件名 '{filename}' 请求时发生未知错误: {e}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"删除文件名 '{filename}' 的记录失败: {e}")