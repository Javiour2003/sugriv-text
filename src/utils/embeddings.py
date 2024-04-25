import os
import torch
import torch.nn as nn
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from nltk import word_tokenize
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# get API key and
OPEN_AI_API_KEY = str(os.getenv("OPENAI_API_KEY"))
MAXIMUM_SEQUENCE_LENGTH = int(os.getenv("MAXIMUM_SEQUENCE_LENGTH"))

class Embeddings():
    def __init__(self) -> None:

        # we will use the open ai embeddingand make the dimentions 1024
        self.embed_model = OpenAIEmbedding(
            model="text-embedding-3-large",
            dimensions=1024,
        )

    def get_text_embedding(self,text):
        prompt_tokens = word_tokenize(text, language='english', preserve_line=True)

        # Check if the size of prompt exceeds the maximum sequence length
        if len(prompt_tokens) < MAXIMUM_SEQUENCE_LENGTH:
            # Calculate the difference in length
            diff = MAXIMUM_SEQUENCE_LENGTH - len(prompt_tokens)

            # Pad missing tokens
            for i in range(diff):
                prompt_tokens.append(' ')

        # Create an array to store the embeddings
        prompt_embedding_array = []

        # Split the text into tokens and iterate through each token
        for i, prompted_word in enumerate(prompt_tokens):
            # Preprocess the input string
            embeddings = self.embed_model.get_text_embedding(prompted_word)

            # Append the text embedding to the array
            prompt_embedding_array.append(embeddings)

        # Create input tensor for the transformer
        prompt_input = torch.tensor(prompt_embedding_array)

        # return the input tensor
        return prompt_input
