from typing import Literal, Optional

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.base import EmbeddingBase
import requests
import json

class LindormAIEmbedding(EmbeddingBase):
    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)

        if self.config.model is None:
            raise ValueError("`model` parameter is required")

        if self.config.embedding_dims is None:
            raise ValueError("`embedding_dims` parameter is required")

        self.model_name = self.config.model
        self.ai_url = self.config.lindorm_base_url
        self.headers = {"x-ld-ak": self.config.lindorm_username,
                        "x-ld-sk": self.config.lindorm_password}

        # check dim
        self.check_dim(self.config.embedding_dims)

    def check_dim(self, dim):
        data = {"input": ["Hello"]}
        result = self.post_model_request(data)
        if len(result)  != dim:
            raise Exception("Dim mismatch")

    def post_model_request(self, data: dict):
        url = 'http://{}/v1/ai/models/{}/infer'.format(self.ai_url, self.model_name)
        try:
            result = requests.post(url, data=json.dumps(data), headers=self.headers, verify=False)
            result.raise_for_status()
            return result.json()['data'][0]
        except Exception as e:
            print(f"Lindormai happend error: {e}")
            return None

    def embed(self, text, memory_action: Optional[Literal["add", "search", "update"]] = None):
        data = {"input": [text]}
        return self.post_model_request(data)
