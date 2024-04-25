
import os
import torch
from src.heads.next_token_prediciton_head import NextTokenPredictionModel
from transformers import PreTrainedModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


MAXIMUM_SEQUENCE_LENGTH = int(os.getenv("MAXIMUM_SEQUENCE_LENGTH"))
NUMBER_OF_HEADS = int(os.getenv("NUMBER_OF_HEADS"))
NUMBER_OF_LAYERS = int(os.getenv("NUMBER_OF_LAYERS"))
PAD_INDEX = int(os.getenv("PAD_INDEX"))
FEED_FORWARD_LAYER_DIMENTION = int(os.getenv("FEED_FORWARD_LAYER_DIMENTION"))
DROP_OUT_RATE = float(os.getenv("DROP_OUT_RATE"))
VOCABULARY_SIZE = os.getenv("VOCABULARY_SIZE")
D_MODEL = int(os.getenv("D_MODEL"))
BATCH_SIZE =  int(os.getenv("BATCH_SIZE"))



class TextPredictionModel(PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.sugriv_config = config
        self.sugriv_config.save_pretrained("sugriv")
        self.text_model = NextTokenPredictionModel(PAD_INDEX,D_MODEL, MAXIMUM_SEQUENCE_LENGTH, NUMBER_OF_HEADS, NUMBER_OF_LAYERS, FEED_FORWARD_LAYER_DIMENTION, DROP_OUT_RATE,VOCABULARY_SIZE,BATCH_SIZE )
    
    def forward(self,input_ids, attention_mask=None,text=[],label=None,pipeline=None):
      # if the pipeline is text then predict the next token
      if pipeline == 'text':
        loss,logits= self.text_model(input_ids,attention_mask,text,label)
        return {'loss':loss,'logits':logits }

    def generate_text(self,prompt):
      return self.text_model.generate_text(prompt)

    def pretrain_text(self,text):
      return self.text_model.pretrain_text(text)

