# app/api/endpoints/detect.py
import logging
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends # 导入 FastAPI 核心组件
from sqlalchemy.ext.asyncio import AsyncSession # 导入异步数据库会话类型

# 导入服务层编写的业务逻辑函数
from app.services.detection import process_uploaded_image_workflow # <-- 从 services 导入业务逻辑函数

# 导入依赖函数 (获取DB会话, 获取用户ID 等)
# 假设这些依赖函数在 app/api/deps.py 中
from app.api.deps import get_db, get_current_user_id # <-- 从 api.deps 导入依赖函数

# 导入 schemas 定义的响应模型
from app.schemas.ocr import MultiOCRResult, OCRResult # <-- 从 schemas 导入 Pydantic 模型

# 导入 utils 中的分类模块加载状态，用于启动检查
from app.utils.classification import bert_loading_error # <-- 导入模型加载状态

logger = logging.getLogger(__name__) # 获取logger实例


router = APIRouter(prefix="/detect", tags=["Detection"]) # tags 用于 API 文档分组

# ====================== API 端点：图片检测接口 ======================
# 定义 POST 路由，用于接收文件并触发检测流程
@router.post(
    "/", # 完整的 URL 将是 /detect/
    response_model=MultiOCRResult, # 指定响应数据的 Pydantic 模型
    summary="上传图片并进行敏感信息检测", # API 文档摘要
    description="接收一张或多张图片文件，执行OCR、敏感信息分类，并返回检测结果和标注图片。", # API 文档描述
)
async def detect_images(
    # 通过 File(...) 定义接收上传文件参数
    files: List[UploadFile] = File(..., description="待上传的图片文件列表"),
    # 通过 Depends(get_db) 依赖注入获取异步数据库会话
    db: AsyncSession = Depends(get_db),
    # 通过 Depends(get_current_user_id) 依赖注入获取当前认证用户的ID (如果包含用户认证)
    user_id: int = Depends(get_current_user_id), # 假设这个依赖函数能获取到用户ID
):
    """
    处理图片检测请求：接收文件，调用服务层处理，返回结果。
    这个函数主要负责调用服务层逻辑和处理API层面的错误。
    """
    logger.info(f"API层接收到文件上传请求，文件数量: {len(files)}, 用户ID: {user_id}")

    # --- 在 API 层进行模型加载状态检查 ---
    if bert_loading_error:
         logger.critical(f"API层检测到分类模型未加载成功，服务不可用。错误: {bert_loading_error}")
         raise HTTPException(status_code=503, detail=f"分类模型未加载成功，服务暂不可用: {bert_loading_error}")


    results_list = [] # 存储每个文件的处理结果，用于构建 MultiOCRResult

    # --- 遍历处理每个上传的文件 ---
    for file in files:
        logger.info(f"API层正在处理文件: {file.filename}")
        try:
            # === 调用服务层编写的业务流程函数 ===
            # 将必要的数据和依赖传递给服务层函数
            processed_data = await process_uploaded_image_workflow(
                file=file,
                user_id=user_id,
                db=db
                # 如果服务层函数需要其他依赖（如模型对象），可以在这里传递
                # classification_model=classification_model # 示例
            )
            # === 服务层调用结束 ===

            # 将服务层返回的处理结果转换为API响应模型的一部分
            # 服务层返回的是字典或内部数据结构，这里转换为 Pydantic 模型 OCRResult
            results_list.append(
                 OCRResult( # 使用 schemas/ocr.py 中定义的 Pydantic 模型
                     filename=processed_data["filename"],
                     annotated_image=f"data:image/png;base64,{processed_data['annotated_image_base64']}", # 格式化Base64字符串
                     detections=processed_data["detections"]
                 )
            )
            logger.info(f"API层成功处理文件: {file.filename}")

        # --- 错误处理 ---
        # 捕获服务层抛出的业务异常或运行时异常，并转换为HTTPException返回给客户端
        # 服务层应该抛出特定的自定义异常，这里为了示例简化，捕获 ValueError 和 RuntimeError
        except (ValueError, RuntimeError) as e:
            logger.error(f"API层捕获到服务层处理文件失败: {file.filename}, 错误: {e}", exc_info=True)
            # 抛出 HTTPException，FastAPI会自动处理并返回标准的错误响应
            raise HTTPException(status_code=500, detail=f"处理文件失败: {file.filename}, 错误: {e}")
        except Exception as e: # 捕获其他未知异常
            logger.error(f"API层处理文件时发生未知错误: {file.filename}, 错误: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"处理文件时发生未知错误: {file.filename}, 请联系管理员。")

    # --- 所有文件处理完毕，构建并返回最终结果 ---
    logger.info("API层所有文件处理完成，返回响应。")
    # MultiOCRResult 是顶层的响应模型，包含一个图片结果列表
    # FastAPI 会自动将这个 Pydantic 模型对象序列化为 JSON 响应
    return MultiOCRResult(images=results_list)


