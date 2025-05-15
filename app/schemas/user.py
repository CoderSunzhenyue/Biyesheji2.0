# app/schemas/user.py

from pydantic import BaseModel, EmailStr, Field # 导入 Pydantic 需要的类型和 Field
from pydantic_settings import SettingsConfigDict # <-- 导入 SettingsConfigDict
from typing import Optional, List, Any # 导入 Python 标准库类型提示
from datetime import datetime # 导入 datetime

# 导入其他模块中可能需要的 Schema，例如 OCRDetection
from app.schemas.ocr import OCRDetection # 假设 OCRDetection 在 app.schemas.ocr


# --- 用户相关的 Pydantic 模型 ---
class UserCreate(BaseModel):
    username: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")
    # ... 其他字段 ...

class UserUpdate(BaseModel):
    username: Optional[str] = Field(None, description="新的用户名")
    email: Optional[EmailStr] = Field(None, description="新的邮箱")
    phone: Optional[str] = Field(None, description="新的手机号")

class ChangePasswordRequest(BaseModel):
    old_password: str = Field(..., description="旧密码")
    new_password: str = Field(..., description="新密码")
    confirm_password: str = Field(..., description="确认新密码")

class TokenOut(BaseModel):
    access_token: str = Field(..., description="Access Token")
    token_type: str = Field("bearer", description="Token 类型，通常是 'bearer'")

# 定义返回给前端的用户信息 Schema
class UserInfo(BaseModel):
    """
    返回给前端的用户信息 Schema。
    """
    username: str = Field(..., description="用户名")
    email: Optional[EmailStr] = Field(None, description="邮箱")
    phone: Optional[str] = Field(None, description="手机号")
    # 如果你的 User 模型有 created_at 字段，并且你想返回创建时间
    # created_at: datetime = Field(None, description="用户创建时间")

    # 配置 Pydantic 模型，允许从 SQLAlchemy 模型实例创建 (Pydantic v2 方式)
    model_config = SettingsConfigDict(from_attributes=True) # <-- 添加 Pydantic v2 ORM 配置

# --- 历史检测记录相关的 Pydantic 模型 ---

class RecordDetail(BaseModel):
    """
    单条历史检测记录的详细 Schema。
    """
    id: int = Field(..., description="记录ID")
    filename: str = Field(..., description="原始文件名")
    detections: List[OCRDetection] = Field(..., description="检测结果详情列表")
    created_at: datetime = Field(..., description="记录创建时间")
    saved_path: Optional[str] = Field(None)



    model_config = SettingsConfigDict(from_attributes=True)

