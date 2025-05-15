# app/core/security.py

import datetime
from passlib.context import CryptContext # 用于密码哈希
from jose import JWTError, jwt # 用于 JWT
import logging # 导入日志
from typing import Optional, Dict, Any # <-- 添加 typing 导入
# 导入配置，以便在安全函数中使用 SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES 等
from app.core.config import settings

logger = logging.getLogger(__name__) # 获取 logger 实例

# ====================== 密码哈希与验证 ======================
# 创建密码哈希上下文
# schemes 指定使用的哈希算法，bcrypt 是推荐的算法
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    验证纯文本密码是否与哈希密码匹配。
    """
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        logger.error(f"验证密码时出错: {e}", exc_info=True)
        return False # 验证失败时返回 False

def get_password_hash(password: str) -> str:
    """
    对纯文本密码进行哈希。
    """
    try:
        return pwd_context.hash(password)
    except Exception as e:
         logger.error(f"对密码进行哈希时出错: {e}", exc_info=True)
         # 无法哈希密码通常是严重错误，可能需要抛出异常或返回特定值
         raise ValueError("无法哈希密码") # 抛出异常

# ====================== JWT Token 创建 ======================
# 这个函数用于创建 access token
def create_access_token(data: dict, expires_delta: Optional[datetime.timedelta] = None) -> str:
    """
    创建 JWT Access Token。
    Args:
        data: 包含要编码到 token 中的数据的字典。
        expires_delta: 可选的 token 过期时间增量。如果未提供，使用配置中的默认值。
    Returns:
        生成的 JWT Access Token 字符串。
    """
    to_encode = data.copy()

    # 设置 token 过期时间
    if expires_delta:
        expire = datetime.datetime.utcnow() + expires_delta
    else:
        expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire}) # 将过期时间添加到 payload 中

    # 使用配置中的 SECRET_KEY 和 ALGORITHM 编码 token
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

    return encoded_jwt

# ====================== JWT Token 验证 (通常用于依赖注入) ======================
# 这个函数通常不会直接在业务逻辑中调用，而是由 app/api/deps.py 中的依赖函数调用
def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """
    验证 JWT Token 的有效性并解码其 payload。
    Args:
        token: 待验证的 JWT token 字符串。
    Returns:
        如果 token 有效且未过期，返回 payload 字典；否则返回 None。
    """
    try:
        # 使用配置中的 SECRET_KEY 和 ALGORITHM 解码 token
        # audience 和 issuer 等高级验证可根据需要添加
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])

        # 可以在这里添加 payload 的结构验证 (例如，验证 'sub' 或 'user_id' 字段是否存在)
        # user_id: int = payload.get("sub") # 通常用户ID保存在 'sub' 字段
        # if user_id is None:
        #     logger.warning("JWT payload 中未包含 sub 字段。")
        #     return None

        # Token 解码成功且未过期，返回 payload
        return payload

    except JWTError as e:
        # 如果 token 无效（签名错误、过期等），jose 会抛出 JWTError
        logger.warning(f"JWT Token 验证失败: {e}")
        return None # 验证失败返回 None
    except Exception as e:
        logger.error(f"验证或解码 JWT Token 时发生意外错误: {e}", exc_info=True)
        return None