# app/db/models/user.py

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship

from app.db.base import Base

# 虽然这里仍然导入 DetectionRecord 用于其他可能的类型提示或直接使用，
# 但在 relationship 中使用字符串可以解决循环导入问题。
# from app.db.models.ocr_record import DetectionRecord # 仍然可以保留导入


# 定义用户模型
class User(Base):
    """
    数据库表模型：用户。
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=True)
    phone = Column(String(20), unique=True, index=True, nullable=True)


    # 定义与 DetectionRecord 的关系，使用字符串 "DetectionRecord"
    # SQLAlchemy 会在映射配置阶段解析这个字符串
    records = relationship("DetectionRecord", back_populates="user", cascade="all, delete") # <-- 使用字符串 "DetectionRecord"


    # def __repr__(self):
    #     return f"<User(id={self.id}, username='{self.username}')>"