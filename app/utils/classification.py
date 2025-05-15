# app/utils/classification.py
import logging
import torch
import torch.nn.functional as F
import re
import os
from transformers import BertTokenizer, BertForSequenceClassification
# from pathlib import Path # 如果用不到 Path 可以不导入
from typing import Optional, List, Dict, Any

# --- 从你原有代码中提取的规则导入 ---
# 假设你的规则定义在 app/utils/rules.py.py 中
# 如果在其他位置，请修改导入路径
try:
    from app.utils.rules import RULE_PATTERNS
except ImportError as e:
    logging.error(f"无法导入规则文件：{e}. 请检查 app/utils/rules.py 文件是否存在以及导入路径是否正确。", exc_info=True)
    RULE_PATTERNS = {} # 如果导入失败，规则匹配将不可用

# --- 从你原有代码中提取的模型加载配置 ---
# 假设你的配置信息在 app/core/config.py 中，或者在这里直接定义路径
# from app.core.config import settings # 如果使用 config.py

logger = logging.getLogger(__name__)

# 定义模型路径，请根据你的实际文件位置修改
ORIGINAL_MODEL_PATH = r"D:\biyesheji\ImageRecognition\trainModel\fine_tuned_model_output\checkpoint-1308"
QUANTIZED_FP16_MODEL_PATH = r"D:\biyesheji\ImageRecognition\Float16\quantized_models\model_fp16.pth"

# 定义敏感信息类别列表，必须与训练模型时使用的标签类别完全一致
SENSITIVE_CATEGORIES = [
    "个人隐私信息",
    "黄色信息",
    "不良有害信息",
    "无敏感信息"
]
PREDICTION_CATEGORIES = SENSITIVE_CATEGORIES

# 创建类别名称到整数 ID 的映射字典
ID_TO_CATEGORY = {i: cat for i, cat in enumerate(PREDICTION_CATEGORIES)}
CATEGORY_TO_ID = {cat: i for i, cat in enumerate(PREDICTION_CATEGORIES)}
NUM_LABELS = len(PREDICTION_CATEGORIES)

# 定义 BERT 模型处理的最大序列长度
STANDARD_CLASSIFICATION_MAX_LENGTH = 512

# --- 检测并设置推理设备 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"分类模型推理设备已设置为: {DEVICE}")
# (此处省略设备信息日志，可放在应用启动时打印)

# 初始化模型和 Tokenizer 变量
bert_tokenizer: Optional[BertTokenizer] = None
bert_model: Optional[BertForSequenceClassification] = None
bert_loading_error: Optional[str] = None

# --- 模型加载函数 (在应用启动时调用) ---
async def load_classification_model():
    global bert_tokenizer, bert_model, bert_loading_error

    if bert_model is not None and bert_tokenizer is not None and bert_loading_error is None:
        # logger.info("分类模型已加载，跳过重复加载。") # 避免频繁日志
        return

    logger.info(f"正在加载 BERT Tokenizer from {ORIGINAL_MODEL_PATH}...")
    try:
        bert_tokenizer = BertTokenizer.from_pretrained(ORIGINAL_MODEL_PATH)
        if bert_tokenizer.pad_token_id is None:
             bert_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
             bert_tokenizer.pad_token_id = bert_tokenizer.vocab['[PAD]']
             logger.info("Added [PAD] token.")

    except Exception as e:
        logger.error(f"加载 BERT Tokenizer 时出错: {e}", exc_info=True)
        bert_loading_error = f"加载 BERT Tokenizer 时出错: {e}"
        return

    if bert_tokenizer is not None:
        if not os.path.exists(QUANTIZED_FP16_MODEL_PATH):
            logger.critical(f"量化后的 FP16 模型文件未找到：{QUANTIZED_FP16_MODEL_PATH}")
            bert_loading_error = f"FP16 模型文件未找到：{QUANTIZED_FP16_MODEL_PATH}"
            return
        else:
            logger.info(f"正在加载量化后的 FP16 BERT 分类模型 state_dict from {QUANTIZED_FP16_MODEL_PATH}...")
            try:
                model_structure = BertForSequenceClassification.from_pretrained(
                    ORIGINAL_MODEL_PATH,
                    num_labels=NUM_LABELS,
                    id2label=ID_TO_CATEGORY,
                    label2id=CATEGORY_TO_ID,
                    torch_dtype=torch.float32
                )
                state_dict = torch.load(QUANTIZED_FP16_MODEL_PATH, map_location=DEVICE)
                model_structure.load_state_dict(state_dict)

                bert_model = model_structure.half()
                bert_model.to(DEVICE)
                bert_model.eval()

                logger.info("量化后的 FP16 BERT 分类模型加载成功。")

            except Exception as e:
                logger.error(f"加载量化后的 FP16 BERT 分类模型时出错: {e}", exc_info=True)
                bert_loading_error = f"加载 FP16 模型时出错: {e}"


# ====================== 分类逻辑函数 (由服务层调用) ======================

def classify_with_rules(text: str) -> Optional[str]:
    """
    使用预设的正则表达式规则对文本进行敏感信息分类。
    """
    if not RULE_PATTERNS:
        return None
    if not isinstance(text, str) or not text.strip():
        return None
    text_lower = text.lower()
    for category, patterns in RULE_PATTERNS.items(): # 直接迭代字典，假设结构是 {category: [pattern1, pattern2]}
        for pattern in patterns:
            try:
                if pattern.search(text_lower):
                    return category
            except Exception as e:
                 logger.error(f"规则匹配时发生错误，模式: {pattern.pattern[:50]}... 错误: {e}", exc_info=True)
    return None

def classify_text_with_model(text_to_classify: str) -> str:
    """
    使用加载的 BERT 分类模型对文本进行敏感信息分类。
    """
    if bert_model is None or bert_tokenizer is None:
        # logger.error("分类模型或 Tokenizer 未加载，无法进行模型分类。")
        return "分类失败"

    if not isinstance(text_to_classify, str) or not text_to_classify.strip():
         return "无敏感信息"

    CONFIDENCE_THRESHOLD = 0.90 # 可从 config 读取

    try:
        inputs = bert_tokenizer(
            text_to_classify,
            return_tensors="pt",
            truncation=True,
            padding='max_length',
            max_length=STANDARD_CLASSIFICATION_MAX_LENGTH
        )
        input_ids = inputs['input_ids'].to(DEVICE)
        attention_mask = inputs['attention_mask'].to(DEVICE)
        token_type_ids = inputs.get('token_type_ids')
        if token_type_ids is not None:
             token_type_ids = token_type_ids.to(DEVICE)

        with torch.no_grad():
            if token_type_ids is not None:
                 outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            else:
                 outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits[0]
        probabilities = F.softmax(logits, dim=-1)
        max_probability, predicted_class_id = torch.max(probabilities, dim=-1)
        predicted_category = ID_TO_CATEGORY.get(predicted_class_id.item(), "分类失败")

        if predicted_category != "无敏感信息" and max_probability.item() < CONFIDENCE_THRESHOLD:
             return "无敏感信息"
        elif predicted_category == "分类失败":
             return "无敏感信息"
        else:
             return predicted_category

    except Exception as e:
        logger.error(f"BERT 分类模型推理过程中出错: {e}", exc_info=True)
        return "分类失败"

def classify_text(text: str) -> str:
    """
    结合规则匹配和 BERT 模型分类对文本进行敏感信息分类。
    """
    if not isinstance(text, str) or not text.strip():
        return "无敏感信息"

    rule_category = classify_with_rules(text)

    if rule_category is not None:
        return rule_category
    else:
        model_category = classify_text_with_model(text)
        return model_category

# --- 用于检查模型加载状态的依赖注入提供者 (可选，但推荐) ---
# 如果你想让服务层或API层依赖于模型是否加载成功，可以定义一个依赖
# 例如：
# from fastapi import Depends, HTTPException
# async def get_classification_resources():
#     if bert_model is None or bert_tokenizer is None or bert_loading_error:
#          raise HTTPException(status_code=503, detail=f"分类模型未加载成功: {bert_loading_error or '未知错误'}")
#     return {"model": bert_model, "tokenizer": bert_tokenizer, "categories": ID_TO_CATEGORY}