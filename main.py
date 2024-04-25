import torch
from src.data.loaders.text_loader import Load
from transformers import AdamW
from src.model.sugriv import Sugriv
from transformers import AdamW
from app import finetune
from  src.tokenizers.gpt_2_tokenizer import GPTTokenizer
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from mangum import Mangum

# load the model
sugriv = Sugriv().get_model()

# get the optimizer 
optimizer = AdamW(sugriv.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=1)

# get the tokenizer 
tokenizer = GPTTokenizer().get_tokenizer()

# add the api 
app = FastAPI()
handler = Mangum(app)

# Define a Pydantic model to validate the request body
class GenerateText(BaseModel):
    prompt: str
    description: str | None = None

class Dataset(BaseModel):
    dataset_name: str
    batch_size: int | None = None


@app.post("/finetune/")
async def finetune_on_dataset(dataset:Dataset):
    
    try:
        # load the data set
        dataloader = Load(dataset.dataset_name)

        # create the dataloader
        train_dataloader = dataloader.get_data(dataset.batch_size)
        train_dataloader = train_dataloader['train']

        # finetune model
        finetune(train_dataloader,sugriv,tokenizer,optimizer,scheduler)

        # Assuming you've trained the model and want to save its weights
        torch.save(sugriv.state_dict(), 'model_weights.pt')

        return {'success':'ok'}
    except RuntimeError:
        return {'error':'run time rror occured'}

@app.post("/generate/")
async def generate(prompt:GenerateText):
    # generate text for endpoint
    try:
        text = sugriv.generate_text(prompt.prompt)
        return {"completion":text}
    except RuntimeError:
        return {'error':'run time rror occured'}

    

if __name__ == "__main__":
 
    uvicorn.run(app, host="127.0.0.1", port=8000)