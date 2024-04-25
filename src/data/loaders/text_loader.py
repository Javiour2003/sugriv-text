# todo make custom dataset using data loader
from datasets import load_dataset
from src.tokenizers.gpt_2_tokenizer import GPTTokenizer
from torch.utils.data import DataLoader

class Load():
    
    def __init__(self,name) -> None:

        # load the text dataset 
        self.text_dataset = load_dataset(name)

        # initiate the tokenizer
        self.tokenizer = GPTTokenizer().get_tokenizer()

        # create the dataset
        self.tokenized_datasets = self.text_dataset.map(self.tokenize_function)

    def tokenize_function(self,examples):
        
            # pipe the prompt and completions
            text = examples['text']
            print(text)
        
            # tokenize the pipe
            return self.tokenizer(text, padding="max_length", truncation=True)
    
    def get_data(self,batch_size):
         
         # split the datset into test and validation
        train_dataset = self.tokenized_datasets["test"].shuffle(seed=42)
        valid_dataset = self.tokenized_datasets["test"].shuffle(seed=42)

        # Then when creating your DataLoader
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

         # return the dataset
        return {'train':train_dataloader,'valid':valid_dataloader}