# app/services/detection.py

import logging
from typing import List, Dict, Any, Optional
from fastapi import UploadFile, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from PIL import Image
import io

# 导入 utils 层封装的能力
from app.utils.ocr import perform_ocr
from app.utils.classification import classify_text, bert_loading_error, ID_TO_CATEGORY
from app.utils.image import image_to_base64, draw_detections_on_image
# 确保 storage.py 导入正确，并且其中的 save_uploaded_image 是异步函数
from app.utils.storage import save_uploaded_image


# 导入 db 层的数据模型和操作
from app.db.models.ocr_record import DetectionRecord

# 导入 schemas 层定义的数据结构 (如果服务层函数需要返回 Pydantic Schema 实例)
# from app.schemas.ocr import OCRResultSchema


logger = logging.getLogger(__name__)

# 定义绘制时的颜色映射 (可以在 config 或 utils 中定义)
COLOR_MAP = {
    "个人隐私信息": "red",
    "黄色信息": "yellow",
    "不良有害信息": "blue",
    "无敏感信息": "green",
    "分类失败": "gray"
}


# 这个函数由 app/api/endpoints/detect.py 中的 POST /detect 接口调用
async def process_uploaded_image_workflow(
    file: UploadFile,
    user_id: int,
    db: AsyncSession
):
    """
    处理单个上传图片的完整检测工作流程。
    """
    logger.info(f"服务层开始处理文件: {file.filename}, 用户ID: {user_id}")

    original_img = None # 定义一个变量来存原始图片，用于后面的绘制

    # --- 1. 文件处理 ---
    try:
        img_data = await file.read()
        original_img = Image.open(io.BytesIO(img_data)).convert("RGB") # 读取后存到 original_img

    except Exception as e:
        logger.error(f"服务层文件读取失败: {file.filename}, 错误: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"无效图片文件或读取失败: {file.filename}")


    # --- 2. OCR 文本检测和识别 ---
    try:
        # perform_ocr 返回检测结果列表
        # 对 original_img 进行 OCR
        detections = await perform_ocr(original_img)
        logger.info(f"服务层 [OCR] {file.filename} 检测到 {len(detections)} 条文本结果")
    except RuntimeError as e:
        logger.error(f"服务层 OCR 文本检测失败: {file.filename}, 错误: {e}", exc_info=True)
        # 注意：如果 OCR 失败，后面绘制和分类都无法进行，这里直接返回错误或根据需求处理
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"OCR 服务暂不可用或处理失败: {e}")
    except Exception as e:
        logger.error(f"服务层 OCR 文本检测失败: {file.filename}, 错误: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"OCR 文本检测过程中出错: {e}")


    # --- 3. 文本分类和结果处理 ---
    processed_detections = []

    # 如果没有检测到文本， processed_detections 将为空
    if detections:
        for det in detections:
            txt = det.get("text", "").strip()
            bbox = det.get("bbox")

            if not txt or not bbox:
                logger.warning(f"服务层 跳过无效的检测结果: {det}")
                continue

            # classify_text 内部已整合规则和模型逻辑
            category = classify_text(txt)

            det["category"] = category
            processed_detections.append(det)

    logger.info(f"服务层 {file.filename} 完成文本分类和标注绘制准备。")


    # --- 4. 绘制标注 ---
    # 即使没有检测到敏感信息，也使用原始图片进行绘制，可能绘制出“无敏感信息”或原始框
    # 如果需要只在检测到文本时才绘制，可以在这里加判断 processed_detections
    annotated_image = None # 定义变量存储绘制后的图片
    try:
        # 在 original_img 的副本上绘制标注
        annotated_image = draw_detections_on_image(original_img.copy(), processed_detections, COLOR_MAP)
        logger.info(f"服务层 {file.filename} 完成标注绘制。")
    except Exception as e:
         logger.error(f"服务层 绘制标注框失败: {file.filename}, 错误: {e}", exc_info=True)
         # 如果绘制失败，可以使用原始图片或返回错误，这里选择使用原始图片继续流程
         annotated_image = original_img.copy() # 绘制失败，用原始图片作为“标注图”继续


    # --- 5. 保存绘制标注后的图片到静态目录 ---
    # 这一步移到绘制之后，保存 annotated_image
    saved_relative_url = None # 定义变量存储保存后的 URL
    try:
        # 调用 utils/storage.py 中的异步函数保存绘制后的图片
        # save_uploaded_image 返回图片的相对 URL
        if annotated_image: # 确保 annotated_image 存在
            saved_relative_url = await save_uploaded_image(annotated_image, file.filename) # <-- 保存 annotated_image
            logger.info(f"服务层 绘制标注后的图片已保存到: {saved_relative_url}")
        else:
             logger.error(f"服务层 没有生成标注图片，无法保存: {file.filename}")
             raise Exception("没有生成标注图片") # 中断流程或根据需求处理

    except Exception as e:
        logger.error(f"服务层 绘制标注后的图片保存失败: {file.filename}, 错误: {e}", exc_info=True)
        # 保存失败通常是严重问题，抛出异常
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"标注图片保存失败: {file.filename}")


    # --- 6. 将检测记录保存到数据库 ---
    # 现在 saved_relative_url 存储的是标注图片的 URL
    try:
         # 即使 processed_detections 为空，只要图片保存成功，也保存记录
         if saved_relative_url: # 确保图片 URL 已获取
            new_record = DetectionRecord(
                user_id=user_id,
                filename=file.filename,
                saved_path=saved_relative_url, # <-- 这里存储的是标注图片的 URL
                # image_url=saved_relative_url, # <-- 如果你的模型有 image_url 字段，也在这里赋值
                detections=processed_detections # 保存处理后的检测结果 (包含分类)
            )
            db.add(new_record)
            await db.commit()

            logger.info(f"服务层 文件 {file.filename} 的检测记录已保存到数据库。记录ID: {getattr(new_record, 'id', 'N/A')}")
         else:
             # 如果图片 URL 都没拿到，就不保存记录了
             logger.error(f"服务层 未获取到标注图片 URL，无法保存记录: {file.filename}")


    except Exception as e:
         logger.error(f"服务层 保存检测记录到数据库失败: {file.filename}, 错误: {e}", exc_info=True)
         await db.rollback()
         # 如果记录保存失败，并且图片已经保存了，可能需要清理已保存的图片
         # 但这里先抛出异常，简化处理
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"保存检测记录失败: {e}")


    # --- 7. 生成标注图片的 Base64 编码 ---
    # 这一步仍然基于 annotated_image 进行，用于接口的实时响应
    annotated_base64 = "" # 初始化
    try:
        if annotated_image: # 确保有绘制后的图片
            annotated_base64 = image_to_base64(annotated_image)
            logger.info(f"服务层 文件 {file.filename} 的标注图片已生成 Base64。")
    except Exception as e:
         logger.error(f"服务层 生成标注图片 Base64 失败: {file.filename}, 错误: {e}", exc_info=True)
         annotated_base64 = "" # 失败则 Base64 为空


    # --- 8. 整理并返回用于API响应的结构化结果 ---
    # API 响应可以继续返回 Base64，或者也可以返回保存的图片 URL (saved_relative_url)
    # 如果前端在记录页面用 URL，在实时上传结果页用 Base64，那么当前返回结构是合理的
    return {
        "filename": file.filename,
        "annotated_image_base64": annotated_base64, # 实时结果返回 Base64
        # "saved_image_url": saved_relative_url, # 如果前端实时页面也需要 URL 可以加上
        "detections": processed_detections # 返回处理后的检测结果列表
    }