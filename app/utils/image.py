# app/utils/image.py
import io
import base64
from PIL import Image, ImageDraw # 需要 pillow 库
import logging
from typing import List, Dict, Any # 导入必要的类型提示

logger = logging.getLogger(__name__) # 获取logger实例

# 定义绘制时的颜色映射 (从你原有代码中提取，可以放在 config 或 utils 中更通用)
COLOR_MAP = {
    "个人隐私信息": "red",
    "黄色信息": "yellow",
    "不良有害信息": "blue",
    "无敏感信息": "green",
    "分类失败": "gray"
}


def image_to_base64(image: Image.Image) -> str:
    """
    将 PIL Image 对象（内存中的图片）转换为 Base64 编码的 PNG 格式字符串。
    """
    try:
        buf = io.BytesIO()
        # 确保图片是 RGB 模式再保存为 PNG，避免RGBA透明度问题
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        logger.error(f"将图片转换为 Base64 时出错: {e}", exc_info=True)
        return "" # 转换失败返回空字符串

def draw_detections_on_image(image: Image.Image, detections: List[Dict[str, Any]], color_map: Dict[str, str] = COLOR_MAP) -> Image.Image:
    """
    在图片的副本上绘制检测到的文本框。
    Args:
        image: PIL Image 对象。
        detections: 包含 'bbox' 和 'category' 的检测结果列表。
        color_map: 类别到颜色的映射字典。
    Returns:
        绘制了标注框的图片副本。
    """
    if not isinstance(image, Image.Image):
         logger.error("无效的图片对象，无法绘制。")
         return image # 返回原图或抛出异常

    try:
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)

        for det in detections:
            bbox = det.get("bbox")
            category = det.get("category", "无敏感信息") # 默认无敏感信息颜色
            if bbox:
                draw_color = color_map.get(category, "green") # 查找颜色，默认绿色
                try:
                    # 绘制多边形或矩形，PaddleOCR返回的是四个点的列表，可以用polygon
                    draw.polygon(bbox, outline=draw_color, width=2)
                except Exception as e:
                    logger.error(f"绘制单个标注框失败: {bbox}, 错误: {e}", exc_info=True)
                    # 单个绘制失败不影响其他框，继续

        return annotated_image

    except Exception as e:
        logger.error(f"在图片上绘制所有标注框时出错: {e}", exc_info=True)
        return image # 绘制失败返回原图