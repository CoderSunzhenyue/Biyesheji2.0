# app/core/config.py

# Pydantic v2 要求从 pydantic-settings 导入 BaseSettings 和 SettingsConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional


# 定义应用的配置类
class Settings(BaseSettings):
    """
    应用配置类，从环境变量或 .env 文件加载配置。
    """
    # --- 运行环境配置 ---
    ENVIRONMENT: str = "development"

    # --- 数据库配置 ---
    DATABASE_URL: str = "mysql+aiomysql://root:root@localhost:3306/test1"

    # --- 安全相关配置 (JWT) ---
    SECRET_KEY: str = "your_secret_key"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # --- CORS 配置 ---
    ALLOWED_ORIGINS: List[str] = ["*"]

    # --- 项目信息 ---
    PROJECT_NAME: str = "图像敏感信息检测系统 API"
    API_V1_STR: str = "/api/v1" # <-- 移除注释，使其成为 Settings 的属性

    # --- 其他模块可能需要的配置 ---
    # 例如模型路径、文件存储路径等，可以从这里读取
    # ORIGINAL_MODEL_PATH: str = "path/to/your/model/output"
    # QUANTIZED_FP16_MODEL_PATH: str = "path/to/your/fp16/model.pth"
    # STATIC_DIR_NAME: str = "static"


    # --- 配置模型加载设置 (Pydantic v2 使用 SettingsConfigDict) ---
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding='utf-8'
        # case_sensitive = True
    )


# 创建 Settings 类的实例
settings = Settings()