from tqdm import tqdm
from transformers import TrainingArguments

# declare the training args
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    logging_dir='./logs',
)


def finetune(train_dataloader,sugriv,tokenizer,optimizer,scheduler):
        
    '''
    For a next token generation task, where you want to predict the next token in a sequence given the previous tokens,
    you typically generate labels by shifting the input sequence by one token.Shift the input tokens by 1 position to the right
    '''

    # Compute the number of training steps
    num_training_steps = len(train_dataloader) * training_args.num_train_epochs

    # Training loop
    progress_bar = tqdm(total=num_training_steps, desc="Training")

    average = []

    for epoch in range(10):

        # train the model
        sugriv.train()

        # total loss per epoch
        total_loss = 0.0

        #number of batches
        num_batches = 0

        # for each step in the train loader
        for step, batch in enumerate(train_dataloader):

            # get the prompt for sentence completion
            prompt = "{0}".format(batch['text'][0])

            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

            # generate the input ids and attention mask
            prompt_tokens = tokenizer(prompt, max_length=1024, truncation=True, return_tensors="pt", padding='max_length')

            # forward pass though LLM
            outputs = sugriv(input_ids=prompt_tokens['input_ids'].long(), label=None, attention_mask=prompt_tokens['attention_mask'].long(),pipeline="text")

            # get the calculated loss
            loss = outputs['loss']

            # do a backward pass through LLM
            optimizer.zero_grad()
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Update the learning rate
            scheduler.step()

            # calculate the toal loss
            total_loss += loss.item()

            # update the progress bar
            progress_bar.update(1)

        num_batches += 1
        avg_loss =  (total_loss / num_batches)
        average.append(avg_loss)
        print(f"Epoch {epoch + 1}, Average loss: {avg_loss:.4f}")