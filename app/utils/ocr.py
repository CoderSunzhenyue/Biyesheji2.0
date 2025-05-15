# app/utils/ocr.py
import logging
from typing import List, Dict, Any, Tuple
from PIL import Image
import numpy as np
from paddleocr import PaddleOCR # 确保你安装了 paddleocr 库
import torch # <-- 添加 torch 导入
# 需要导入 run_in_executor 来在异步函数中运行同步代码
import asyncio
import concurrent.futures # 用于创建线程池执行器

logger = logging.getLogger(__name__)

# ====================== PaddleOCR Engine Initialization ======================
# 在模块加载时初始化 OCR 引擎
# use_angle_cls=True 用于文本方向分类，lang='ch' 指定中文
# ocr_version='PP-OCRv4' 指定使用的模型版本
# 可以考虑将这些参数从 config 文件中读取
try:
    logger.info("正在初始化 PaddleOCR 引擎...")
    # 初始化时可能需要下载模型，如果网络不好或模型未下载，这里可能耗时或失败
    # 可以考虑将初始化放在应用启动时，并在失败时阻止应用启动
    ocr_engine = PaddleOCR(use_angle_cls=True, lang='ch', ocr_version='PP-OCRv4', use_gpu=True if torch.cuda.is_available() else False) # 根据是否有GPU决定是否使用GPU
    logger.info("PaddleOCR 引擎初始化成功。")
except Exception as e:
    logger.error(f"PaddleOCR 引擎初始化失败: {e}", exc_info=True)
    ocr_engine = None # 初始化失败则将引擎设为 None

# 创建一个线程池执行器，用于运行同步的 OCR 任务
# max_workers 可以根据需要调整，通常设置为 CPU 核心数或稍多
executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)


# ====================== 图像文本检测函数 ======================
# 这个函数由服务层调用
async def perform_ocr(image: Image.Image) -> List[Dict[str, Any]]:
    """
    对输入的 PIL Image 对象执行图像文本检测和识别 (OCR)。
    在线程池中运行同步的 PaddleOCR 任务，使其成为异步函数。
    Args:
        image: 待处理的 PIL Image 对象。
    Returns:
        包含检测到的文本、置信度和包围框信息的列表。
        例如: [{"text": "你好", "confidence": 0.99, "bbox": [[x1,y1], [x2,y2], ...]}, ...]
    Raises:
        RuntimeError: 如果 OCR 引擎未成功初始化或 OCR 过程失败。
    """
    if ocr_engine is None:
        logger.error("OCR 引擎未加载成功，无法执行 OCR。")
        raise RuntimeError("OCR 引擎未加载成功，服务不可用。")

    if not isinstance(image, Image.Image):
         logger.error("接收到无效的图片对象进行 OCR。")
         raise ValueError("无效的图片对象")

    logger.debug("正在将 PIL Image 转换为 numpy array 进行 OCR...")
    # PaddleOCR 需要 numpy array 作为输入
    image_np = np.array(image)
    logger.debug("PIL Image 转换为 numpy array 完成。")


    logger.info("正在执行 PaddleOCR 检测和识别...")
    # 运行同步的 ocr_engine.ocr 方法在一个线程池中，使其成为异步可等待的
    loop = asyncio.get_running_loop()
    try:
        # cls=True 表示进行文本方向分类，这与引擎初始化参数一致
        # result 是一个列表，通常 result[0] 包含主要的检测结果
        result = await loop.run_in_executor(
            executor,
            lambda: ocr_engine.ocr(image_np, cls=True)
        )
        logger.info("PaddleOCR 检测和识别完成。")

    except Exception as e:
        logger.error(f"执行 PaddleOCR 过程中出错: {e}", exc_info=True)
        raise RuntimeError(f"执行 OCR 过程中出错: {e}")

    # --- 解析 OCR 结果 ---
    detections = []
    # 检查结果格式，PaddleOCR 返回的格式可能因版本而异
    if result and result[0]:
        for line in result[0]:
            # 确保 line 是预期的格式 (列表或元组)
            if isinstance(line, (list, tuple)) and len(line) >= 2:
                points = line[0] # 包围框坐标列表
                text_info = line[1] # 文本和置信度信息

                if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                    text = text_info[0] # 文本字符串
                    confidence = text_info[1] # 置信度

                    # 确保 points 是一个列表，且包含坐标点
                    if isinstance(points, list) and points:
                         # 转换为整数坐标，虽然 PaddleOCR 返回的通常已经是整数
                         try:
                             points_int = [(int(point[0]), int(point[1])) for point in points]
                         except (ValueError, TypeError):
                             logger.warning(f"无效的包围框坐标格式: {points}, 跳过此检测结果。")
                             continue # 跳过无效坐标

                         detections.append({
                             "text": str(text), # 确保是字符串
                             "confidence": float(confidence), # 确保是浮点数
                             "bbox": points_int # 使用整数坐标
                         })
                    else:
                        logger.warning(f"无效的包围框格式: {points}, 跳过此检测结果。")
                else:
                    logger.warning(f"无效的文本信息格式: {text_info}, 跳过此检测结果。")
            else:
                logger.warning(f"无效的 OCR 行结果格式: {line}, 跳过此结果。")

    logger.info(f"OCR 结果解析完成，共 {len(detections)} 条有效检测结果。")

    # 这个函数只返回检测结果列表，不绘制在图片上
    return detections

# --- 可以在这里添加一个函数，用于在应用启动时检查 OCR 引擎状态 ---
# 例如：
# async def check_ocr_engine_status():
#     if ocr_engine is None:
#         raise RuntimeError("PaddleOCR 引擎初始化失败，请检查日志。")
#     # 可以尝试运行一个简单的 OCR 任务来验证引擎是否工作正常
#     # try:
#     #     await perform_ocr(Image.new('RGB', (10, 10))) # 对一个空白小图进行测试
#     # except Exception as e:
#     #     raise RuntimeError(f"PaddleOCR 引擎自检失败: {e}")