# app/schemas/ocr.py

from pydantic import BaseModel, Field # 导入 BaseModel 和 Field
from typing import List # 导入 List

# 定义单个 OCR 检测结果的 Schema
class OCRDetection(BaseModel):
    """
    单个检测到的文本框的结果 Schema。
    """
    text: str = Field(..., description="识别到的文本内容")
    confidence: float = Field(..., description="文本识别的置信度")
    bbox: List[List[float]] = Field(..., description="文本包围框的坐标点列表 (格式: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]])")
    category: str = Field(..., description="敏感信息分类结果 (如: '个人隐私信息', '黄色信息', '无敏感信息')") # 新增：敏感信息分类结果

    # (可选) 配置 Pydantic 模型，允许从 SQLAlchemy 模型创建
    # class Config:
    #     orm_mode = True # deprecated in Pydantic V2, use from_attributes=True


# 定义单张图片检测结果的 Schema
class OCRResult(BaseModel):
    """
    单张图片的检测结果 Schema。
    """
    filename: str = Field(..., description="原始文件名")
    annotated_image: str = Field(..., description="绘制了标注框的图片的 Base64 编码 (PNG 格式)") # Base64 编码的图像
    detections: List[OCRDetection] = Field(..., description="检测到的所有文本及分类结果列表")

    # (可选) 配置 Pydantic 模型
    # class Config:
    #     orm_mode = True # deprecated in Pydantic V2, use from_attributes=True


# 定义多张图片检测结果的 Schema (用于 API 响应的顶层结构)
class MultiOCRResult(BaseModel):
    """
    多张图片检测结果的列表 Schema (用于 API 响应)。
    """
    images: List[OCRResult] = Field(..., description="多张图片的检测结果列表")