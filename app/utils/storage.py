# app/utils/storage.py
import os
from pathlib import Path
from PIL import Image
import logging
import uuid
from typing import Optional
import asyncio # 用于 run_in_executor
import concurrent.futures # 用于线程池执行器

logger = logging.getLogger(__name__)

# --- 文件保存路径配置 ---
# 假设静态文件根目录在项目根目录下的 'static' 文件夹
# 可以从 config 中读取
# from app.core.config import settings
# STATIC_ROOT = Path(settings.STATIC_DIR_NAME) # 例如 Path("static")

# 或者，直接根据 app 目录找到 static 目录
BASE_DIR = Path(__file__).resolve().parent.parent.parent # app 目录的上上级目录 (项目根目录)
STATIC_ROOT = BASE_DIR / "static" # 项目根目录下的 static 目录

# 上传的图片文件通常保存在 static 下的一个子目录，例如 'uploads'
UPLOAD_DIR = STATIC_ROOT / "uploads"

# 确保上传目录存在 (可以在应用启动时检查和创建)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"图片上传保存目录已设置为: {UPLOAD_DIR}")

# 创建一个线程池执行器，用于运行同步的文件保存任务
# max_workers 可以根据需要调整
executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)


async def save_uploaded_image(image: Image.Image, original_filename: str) -> str:
    """
    将 PIL Image 对象保存到上传文件目录，并返回保存的相对 URL 路径。
    使用 UUID 生成唯一文件名，保留原文件扩展名。
    Args:
        image: PIL Image 对象。
        original_filename: 原始文件的文件名 (用于提取扩展名)。
    Returns:
        图片在服务器上的相对 URL 路径 (例如: /static/uploads/unique_id.png)。
    Raises:
        IOError: 文件保存失败。
        ValueError: 输入图片对象无效。
    """
    if not isinstance(image, Image.Image):
        logger.error("无效的图片对象，无法保存。")
        raise ValueError("无效的图片对象")

    try:
        # 获取文件扩展名
        _, ext = os.path.splitext(original_filename)
        if not ext:
            ext = ".png" # 默认 png
        else:
             ext = ext.lower()

        # 生成唯一文件名
        unique_filename = f"{uuid.uuid4()}{ext}"
        save_path = UPLOAD_DIR / unique_filename # 完整的本地保存路径

        # 运行同步的 image.save 方法在一个线程池中，使其成为异步可等待的
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            executor,
            lambda: image.save(save_path)
        )

        # 返回相对 URL 路径
        relative_url = f"/static/uploads/{unique_filename}" # <-- 确保与 main.py 中挂载的路径匹配
        logger.info(f"图片 '{original_filename}' 已保存为 '{unique_filename}' 到 {save_path}，URL: {relative_url}")

        return relative_url

    except Exception as e:
        logger.error(f"保存图片文件失败: {original_filename}, 错误: {e}", exc_info=True)
        raise IOError(f"保存图片文件失败: {original_filename}")

def get_static_file_path(relative_url: str) -> Optional[Path]:
     """
     根据图片的相对 URL 路径，获取其在服务器上的本地文件系统路径。
     Args:
         relative_url: 图片在服务器上的相对 URL 路径 (例如: /static/uploads/unique_id.png)。
     Returns:
         文件的本地 Path 对象，如果路径无效或不在静态文件目录下则返回 None。
     """
     # 检查 URL 是否以静态文件挂载路径开头
     if not relative_url.startswith("/static/"):
         logger.warning(f"非静态文件 URL: {relative_url}")
         return None

     # 移除 /static/ 前缀，得到相对于 static 根目录的路径
     relative_static_path_str = relative_url[len("/static/"):].lstrip("/") # 移除前导斜杠

     # 检查路径是否包含非法字符或尝试访问 static 目录之外
     # 可以添加更严格的路径验证
     if ".." in relative_static_path_str:
         logger.warning(f"疑似恶意路径: {relative_url}")
         return None

     # 构建完整的本地文件路径
     local_path = STATIC_ROOT / relative_static_path_str

     # 额外的安全检查：确保生成的路径确实在 static 根目录及其子目录内
     try:
         if not local_path.resolve().is_relative_to(STATIC_ROOT.resolve()):
              logger.warning(f"路径超出静态文件根目录: {local_path}")
              return None
     except Exception as e:
         logger.error(f"解析路径时出错: {local_path}, 错误: {e}", exc_info=True)
         return None


     logger.debug(f"URL '{relative_url}' 对应的本地路径: {local_path}")
     return local_path


async def delete_static_file(relative_url: str) -> bool:
    """
    根据图片的相对 URL 路径，删除服务器上的图片文件。
    Args:
        relative_url: 图片在服务器上的相对 URL 路径。
    Returns:
        bool: 如果文件存在且删除成功返回 True，否则返回 False。
    """
    local_path = get_static_file_path(relative_url)

    if local_path is None:
        logger.warning(f"无效或非静态文件 URL，跳过删除: {relative_url}")
        return False

    if local_path.exists() and local_path.is_file():
        try:
            # 使用 asyncio 的 run_in_executor 运行同步的 os.remove
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(executor, lambda: os.remove(local_path))
            logger.info(f"已删除静态文件: {local_path}")
            return True
        except Exception as e:
            logger.error(f"删除静态文件失败: {local_path}, 错误: {e}", exc_info=True)
            return False
    else:
        logger.warning(f"文件不存在或不是文件，跳过删除: {local_path}")
        return False