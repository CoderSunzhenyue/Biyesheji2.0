# app/db/models/ocr_record.py

from datetime import datetime
from sqlalchemy.orm import relationship
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, JSON

from app.db.base import Base

# 虽然这里仍然导入 User 用于其他可能的类型提示或直接使用，
# 但在 relationship 中使用字符串可以解决循环导入问题。
# from app.db.models.user import User # 仍然可以保留导入


# 定义检测记录模型
class DetectionRecord(Base):
    """
    数据库表模型：检测记录。
    """
    __tablename__ = "detection_records"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)

    filename = Column(String(255), nullable=False)
    saved_path = Column(String(512), nullable=True)

    detections = Column(JSON, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # 定义与 User 模型的关系，使用字符串 "User"
    # SQLAlchemy 会在映射配置阶段解析这个字符串
    user = relationship("User", back_populates="records") # <-- 使用字符串 "User"


    # def __repr__(self):
    #     return f"<DetectionRecord(id={self.id}, filename='{self.filename}', user_id={self.user_id})>"