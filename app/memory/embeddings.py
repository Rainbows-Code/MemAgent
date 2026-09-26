import logging
import hashlib
import numpy as np
from typing import List, Optional
from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Embedding 向量生成器
    优先加载开源 SentenceTransformers 中文模型 (如 text2vec-base-chinese / bge-small-zh-v1.5)
    如模型未下载，提供确定性 Hash 向量 fallback (确保无额外网络依赖下也可极速测试)
    """

    def __init__(
        self,
        model_name: str = settings.EMBEDDING_MODEL_NAME,
        dimension: int = settings.EMBEDDING_DIMENSION,
    ):
        self.model_name = model_name
        self.dimension = dimension
        self._model = None
        self.use_real_model = False

        try:
            from sentence_transformers import SentenceTransformer
            # 尝试装载 sentence-transformers 模型
            logger.info(f"正在加载 Embedding 模型: {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
            self.use_real_model = True
            logger.info("Embedding 模型加载成功！")
        except Exception as e:
            logger.warning(
                f"未能加载 SentenceTransformer 模型 ({e})，将切换为确定性 Mock 向量生成器 (维度 {self.dimension})。"
            )
            self.use_real_model = False

    def embed_query(self, text: str) -> List[float]:
        """
        生成单个文本的 Embedding 向量
        """
        if self.use_real_model and self._model:
            vec = self._model.encode(text, normalize_embeddings=True)
            return vec.tolist()

        # Fallback: 基于 SHA256 + 随机种子的伪向量（L2归一化，维度为 dimension）
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = np.random.RandomState(seed)
        raw_vec = rng.randn(self.dimension)
        norm_vec = raw_vec / np.linalg.norm(raw_vec)
        return norm_vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量生成文本的 Embedding 向量
        """
        if self.use_real_model and self._model:
            vecs = self._model.encode(texts, normalize_embeddings=True)
            return vecs.tolist()

        return [self.embed_query(t) for t in texts]


# 全局单例
embedding_model = EmbeddingModel()
