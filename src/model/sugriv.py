from src.model.registry.default_config import SugrivConfig
from src.model.registry.base_model import TextPredictionModel
from src.utils.initialize_weights import initialize_weights


class Sugriv():
    def __init__(self) -> None:
        self.sugriv_config = SugrivConfig(block_type="bottleneck", stem_width=32, stem_type="deep", avg_down=True)
        self.sugriv = TextPredictionModel(self.sugriv_config)
        initialize_weights(self.sugriv)

    def get_model(self):
        return self.sugriv
    
    def generate_text(self,prompt):
      return self.sugriv.generate_text(prompt)

    def pretrain_text(self,text):
      return self.sugriv.pretrain_text(text)

