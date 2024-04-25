import os
import numpy as np
import torch
import torch.nn as nn
from nltk import word_tokenize
from src.transformers.gpt.transformer import Transformer
from src.tokenizers.gpt_2_tokenizer import GPTTokenizer
from src.utils.rotate_array import rotate_array
from src.utils.embeddings import Embeddings

class NextTokenPredictionModel(nn.Module):
    def __init__(self, pad_index, d_model, max_seq_length, n_heads, n_layers, ff_dim, dropout_rate, vocab_size, batch_size):
        super(NextTokenPredictionModel, self).__init__()

        # the base model
        self.base_model = Transformer(pad_index, vocab_size, d_model, max_seq_length, n_heads, n_layers, ff_dim, dropout_rate, batch_size)

        # get the tokenizer
        self.tokenizer = GPTTokenizer()
        self.tokenizer = self.tokenizer.get_tokenizer()

        # get the embeddings
        self.embed_model = Embeddings()

        self.max_seq_length = max_seq_length

        # create a linear layer
        self.linear = nn.Linear(int(vocab_size), int(d_model))  # Input size: 50258, Output size: 1024

        # Define cross-entropy loss function
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, text=[], label=None):

        prompt_tokens = self.tokenizer.decode(input_ids[0].int(), skip_special_tokens=True)

        # split the tokens using a word tokenizer
        prompt_tokens = word_tokenize(prompt_tokens, language='english', preserve_line=True)

        # since the maximum sequence length is 10
        # if the size of prompt > seq_len then add padding otherrwise trim to seq_len
        if len(prompt_tokens) < self.max_seq_length:

          # calculate the diference in length
            diff = self.max_seq_length - len(prompt_tokens)

          # pad missing string
            for i in range(diff):
                prompt_tokens.append(' ')
        else:
            prompt_tokens = prompt_tokens[:self.max_seq_length]

        # rotate the prompts to get the labeles
        label_tokens = rotate_array(prompt_tokens, 1)

        # create an array to store the embedding
        prompt_embedding_array = []

        # for the given text of length split the text
        # at last word and iterate through all previous
        for i, prompted_word in enumerate(prompt_tokens):

            # Preprocess the input string
            embeddings = self.embed_model.get_text_embedding(prompted_word)

            # append the text embedding to an array
            prompt_embedding_array.append(embeddings)

        # create input for the transformer
        prompt_input = torch.from_numpy(np.array(prompt_embedding_array))

        # get the transformer output
        outputs, attention = self.base_model(prompt_input)

         # create an array to store the target_ids
        target_ids = []

        # for the given text of length split the text
        # at last word and iterate through all previous
        for i, token in enumerate(label_tokens):

            # get the target id
            target_id = self.tokenizer.convert_tokens_to_ids(token)
            target_ids.append(target_id)

            # get the predicted id
            prediction = outputs[0][i]
            predicted_id = np.argmax(prediction.detach().numpy())

            # Assuming predicted_ids and target_ids are your predicted and target IDs respectively
            predicted_id = torch.tensor([predicted_id]).float()  # predicted IDs
            target_id = torch.tensor([target_id]).float()     #target IDs

        # the cross entropy
        logits = outputs[0]

        # get the target ids
        target_ids = torch.tensor(target_ids)

        # calcualte the loss function
        loss = self.cross_entropy_loss(logits, target_ids)

        return loss,logits

    def pretrain_text(self,text):

        prompt_tokens = word_tokenize(text, language='english', preserve_line=True)

        # since the maximum sequence length is 10
        # if the size of prompt > seq_len then add padding otherrwise trim to seq_len
        if len(prompt_tokens) < self.max_seq_length:

          # calculate the diference in length
            diff = self.max_seq_length - len(prompt_tokens)

          # pad missing string
            for i in range(diff):
                prompt_tokens.append(' ')

        # Create an array to store the embedding
        prompt_embedding_array = []

        # For the given text of length, split the text
        # at last word and iterate through all previous
        for i, prompted_word in enumerate(prompt_tokens):

            # Preprocess the input string
            embeddings = self.embed_model.get_text_embedding(prompted_word)

            # Append the text embedding to an array
            prompt_embedding_array.append(embeddings)

        # Create input for the transformer
        prompt_input = torch.from_numpy(np.array(prompt_embedding_array))

        outputs,attention = self.base_model(prompt_input)

        return outputs,attention

    def generate_text(self,prompt,top_k=1):
        ''' generate the next K token probalities given a prompt
            https://stackoverflow.com/questions/76397904/generate-the-probabilities-of-all-the-next-possible-word-for-a-given-text
        '''

        outputs,attention = self.pretrain_text(prompt)

        next_token_candidates_tensor = outputs[0][0]

        topk_candidates_indexes = torch.topk(next_token_candidates_tensor, top_k).indices.tolist()

      # Get the token probabilities for all candidates.
        all_candidates_probabilities = torch.nn.functional.softmax(next_token_candidates_tensor, dim=-1)

        # Filter the token probabilities for the top k candidates.
        topk_candidates_probabilities = all_candidates_probabilities[topk_candidates_indexes].tolist()

        # Decode the top k candidates back to words.
        topk_candidates_tokens = [self.tokenizer.decode([idx]).strip() for idx in topk_candidates_indexes]

        # Return the top k candidates and their probabilities.
        return list(zip(topk_candidates_tokens, topk_candidates_probabilities))