import os
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer
from transformers import Trainer, TrainingArguments
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW
import pandas as pd
import pickle
from transformers import pipeline

#################################################################################################################
### Other Functions

def LoadDF(DF_path):
    dataframe = pd.read_csv(DF_path)
    return dataframe

def SavePKL(data,data_filename):
    with open(data_filename,'wb') as file:
        pickle.dump(data,file)
    return

def LoadPKL(data_filename):
    with open(data_filename,'rb') as file:
        data = pickle.load(file)
    return data

def LoadDistilBERTModel():
    distilBERT_model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-cased-distilled-squad')
    return distilBERT_model

def CreateTrainAndTestLoader(prepared_Q_and_A_DF,train_loader_fine_tuned_filepath,test_loader_fine_tuned_filepath):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased-distilled-squad')
    encoded_data = tokenizer(prepared_Q_and_A_DF['Questions'].tolist(),return_tensors='pt',padding=True,truncation=True)
    
    input_ids = encoded_data['input_ids']
    attention_mask = encoded_data['attention_mask']
    
    dataset = TensorDataset(input_ids,attention_mask)
    
    train_size = int(0.7*len(dataset))
    test_size = len(dataset) - train_size
    train_dataset,test_dataset = random_split(dataset,[train_size,test_size])
    
    train_loader_fine_tuned = DataLoader(train_dataset,batch_size=8,shuffle=True)
    test_loader_fine_tuned = DataLoader(test_dataset,batch_size=8,shuffle=False)
    torch.save(train_loader_fine_tuned,train_loader_fine_tuned_filepath)
    torch.save(test_loader_fine_tuned,test_loader_fine_tuned_filepath)
    return train_loader_fine_tuned,test_loader_fine_tuned

def LoadTrainAndTestLoader(train_loader_fine_tuned_filepath,test_loader_fine_tuned_filepath):
    train_loader_fine_tuned = torch.load(train_loader_fine_tuned_filepath)
    test_loader_fine_tuned = torch.load(test_loader_fine_tuned_filepath)
    return train_loader_fine_tuned,test_loader_fine_tuned

def CreateOrLoadTrainAndTestLoader(prepared_Q_and_A_DF,train_loader_fine_tuned_filepath,test_loader_fine_tuned_filepath,create_or_load_string_train_and_test_loader_fine_tuned='load'):
    if create_or_load_string_train_and_test_loader_fine_tuned in ['Create','create']:
        train_loader_fine_tuned,test_loader_fine_tuned = CreateTrainAndTestLoader(prepared_Q_and_A_DF,train_loader_fine_tuned_filepath,test_loader_fine_tuned_filepath)
    else:
        train_loader_fine_tuned,test_loader_fine_tuned = LoadTrainAndTestLoader(train_loader_fine_tuned_filepath,test_loader_fine_tuned_filepath)
    return train_loader_fine_tuned,test_loader_fine_tuned

def CalculateStartEndPositions(start_logits,end_logits):
    start_positions = torch.argmax(start_logits,dim=1)
    end_positions = torch.argmax(end_logits,dim=1)
    return start_positions,end_positions

def CreateDistilBERTModelFineTuned(model,train_loader_fine_tuned,test_loader_fine_tuned,training_args,ditilBERT_model_fine_tuned_filepath):
    checkpoint_filepath = ditilBERT_model_fine_tuned_filepath + '_checkpoint.pth'
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=1, verbose=True)
    loss_function = torch.nn.CrossEntropyLoss()

    start_epoch = 0

    train_losses = []
    test_losses = []

    if not os.path.exists(checkpoint_filepath):
        os.makedirs(checkpoint_filepath)

    checkpoints = [f for f in os.listdir(checkpoint_filepath) if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]
    if checkpoints:
        latest_checkpoint = max(checkpoints)
        checkpoint = torch.load(os.path.join(checkpoint_filepath, latest_checkpoint))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        train_losses = checkpoint.get('train_losses', [])
        test_losses = checkpoint.get('test_losses', [])
        print(f"Resuming training from epoch {start_epoch + 1}")

    for epoch in range(start_epoch, start_epoch + training_args.num_train_epochs):
        model.train()
        total_train_loss = 0.0
        for batch in train_loader_fine_tuned:
            optimizer.zero_grad()
            outputs = model(input_ids=batch[0], attention_mask=batch[1])
            
            start_positions, end_positions = CalculateStartEndPositions(outputs.start_logits, outputs.end_logits)
            
            loss = loss_function(outputs.start_logits, start_positions) + loss_function(outputs.end_logits, end_positions)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        average_train_loss = total_train_loss/len(train_loader_fine_tuned)
        train_losses.append(average_train_loss)
        print(f'Epoch {epoch + 1}/{start_epoch + training_args.num_train_epochs}, Training Loss: {average_train_loss}')

        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader_fine_tuned:
                outputs = model(input_ids=batch[0], attention_mask=batch[1])
                
                # Use the question-answering pipeline to calculate start and end positions
                start_positions, end_positions = CalculateStartEndPositions(outputs.start_logits, outputs.end_logits)
                
                loss = loss_function(outputs.start_logits, start_positions) + loss_function(outputs.end_logits, end_positions)
                total_test_loss += loss.item()

        average_test_loss = total_test_loss/len(test_loader_fine_tuned)
        test_losses.append(average_test_loss)
        print(f'Epoch {epoch + 1}/{start_epoch + training_args.num_train_epochs}, Testing Loss: {average_test_loss}')

        scheduler.step(average_test_loss)

    SavePKL(train_losses,average_train_loss_fine_tuned_filename)
    SavePKL(test_losses,average_test_loss_fine_tuned_filename)
    model.save_pretrained(ditilBERT_model_fine_tuned_filepath)
    return model,train_losses,test_losses

if __name__ == "__main__":
     #Variables
    prepared_Q_and_A_DF_filepath = 'Additional_Data/PreparedQ_and_A_Gita.csv'
    train_loader_fine_tuned_filepath = 'Additional_Data/DistilBERT_fine_tuned_train_loader_fine_tuned_Gita.pth'
    test_loader_fine_tuned_filepath = 'Additional_Data/DistilBERT_fine_tuned_test_loader_fine_tuned_Gita.pth'
    ditilBERT_model_fine_tuned_filepath = 'Models/DistilBERT_model_fine_tuned_Gita'
    average_train_loss_fine_tuned_filename = 'Additional_Data/DistilBERT_fine_tuned_training_loss_Gita.pkl'
    average_test_loss_fine_tuned_filename = 'Additional_Data/DistilBERT_fine_tuned_test_loss_Gita.pkl'
    
    create_or_load_string_train_and_test_loader_fine_tuned = 'Create'
    
    #Main Code
    training_args = TrainingArguments(output_directory=ditilBERT_model_fine_tuned_filepath,num_train_epochs=5,per_device_train_batch_size=8,per_device_eval_batch_size=8,warmup_steps=500,weight_decay=0.01,logging_dir='./logs',save_total_limit=3)

    # Prepare data
    prepared_Q_and_A_DF = LoadDF(prepared_Q_and_A_DF_filepath)
    distilBERT_model = LoadDistilBERTModel()
    train_loader_fine_tuned,test_loader_fine_tuned = CreateOrLoadTrainAndTestLoader(prepared_Q_and_A_DF,train_loader_fine_tuned_filepath,test_loader_fine_tuned_filepath,create_or_load_string_train_and_test_loader_fine_tuned)
    trained_model = CreateDistilBERTModelFineTuned(distilBERT_model,train_loader_fine_tuned,test_loader_fine_tuned,training_args,ditilBERT_model_fine_tuned_filepath)
