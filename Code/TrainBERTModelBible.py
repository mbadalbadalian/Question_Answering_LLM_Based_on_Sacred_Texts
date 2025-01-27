import os
import torch
from torch.utils.data import DataLoader,TensorDataset,random_split
from transformers import BertForMaskedLM,BertTokenizer
from torch.optim import AdamW
import pandas as pd
import pickle

#################################################################################################################
### Other Functions

#LoadDF Function
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

def CreateTokenizedESVBibleDF(ESV_Bible_DF,ESV_Bible_DF_tokenized_filepath):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    ESV_Bible_DF_tokenized = tokenizer(ESV_Bible_DF['Text'].tolist(),return_tensors='pt',max_length=512,truncation=True,padding=True)
    torch.save(ESV_Bible_DF_tokenized,ESV_Bible_DF_tokenized_filepath)
    return ESV_Bible_DF_tokenized

def LoadTokenizedESVBibleDF(ESV_Bible_DF_tokenized_filepath):
    ESV_Bible_DF_tokenized = torch.load(ESV_Bible_DF_tokenized_filepath)
    return ESV_Bible_DF_tokenized

def CreateOrLoadTokenizedESVBibleDF(ESV_Bible_DF,ESV_Bible_DF_tokenized_filepath,create_or_load_string_Bible_DF_tokenized='load'):
    if create_or_load_string_Bible_DF_tokenized in ['Create','create']:
        ESV_Bible_DF_tokenized = CreateTokenizedESVBibleDF(ESV_Bible_DF,ESV_Bible_DF_tokenized_filepath)
    else:
        ESV_Bible_DF_tokenized = LoadTokenizedESVBibleDF(ESV_Bible_DF_tokenized_filepath)
    return ESV_Bible_DF_tokenized

def CreateTrainAndTestLoader(ESV_Bible_DF_tokenized,train_loader_filepath,test_loader_filepath):
    train_size_value = 0.7
    
    dataset = TensorDataset(ESV_Bible_DF_tokenized['input_ids'], ESV_Bible_DF_tokenized['attention_mask'], ESV_Bible_DF_tokenized['input_ids'])    
    train_size = int(train_size_value*len(dataset))
    test_size = len(dataset) - train_size
    train_dataset,test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    torch.save(train_loader,train_loader_filepath)
    torch.save(test_loader,test_loader_filepath)
    return train_loader,test_loader

def LoadTrainAndTestLoader(train_loader_filepath,test_loader_filepath):
    train_loader = torch.load(train_loader_filepath)
    test_loader = torch.load(test_loader_filepath)
    return train_loader,test_loader

def CreateOrLoadTrainAndTestLoader(ESV_Bible_DF_tokenized,train_loader_filepath,test_loader_filepath,create_or_load_string_train_and_test_loader='load'):
    if create_or_load_string_train_and_test_loader in ['Create','create']:
        train_loader,test_loader = CreateTrainAndTestLoader(ESV_Bible_DF_tokenized,train_loader_filepath,test_loader_filepath)
    else:
        train_loader,test_loader = LoadTrainAndTestLoader(train_loader_filepath,test_loader_filepath)
    return train_loader,test_loader

def LoadCheckpoint(BERT_model,optimizer,checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    BERT_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    average_train_loss_list = checkpoint['train_loss_list']
    average_test_loss_list = checkpoint['test_loss_list']
    return BERT_model,optimizer,start_epoch,average_train_loss_list,average_test_loss_list

def CreateTrainedBERTModel(train_loader,test_loader,BERT_model_filepath,average_train_loss_filename,average_test_loss_filename,checkpoint_directory):
    num_epochs = 5
    
    BERT_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    optimizer = AdamW(BERT_model.parameters(), lr=5e-5)
    average_train_loss_list = []
    average_test_loss_list = []
    start_epoch = 0

    if os.path.exists(checkpoint_directory):
        checkpoints = [f for f in os.listdir(checkpoint_directory) if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]
        if checkpoints:
            latest_checkpoint = max(checkpoints)
            BERT_model,optimizer,start_epoch,_,_,average_train_loss_list,average_test_loss_list = LoadCheckpoint(BERT_model,optimizer,os.path.join(checkpoint_directory,latest_checkpoint))
            print("average_train_loss_list:")
            print(average_train_loss_list)
            print()
            print("average_test_loss_list:")
            print(average_test_loss_list)

    for epoch in range(start_epoch, num_epochs):
        BERT_model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = BERT_model(input_ids=batch[0],attention_mask=batch[1],labels=batch[2])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        average_train_loss = total_train_loss/len(train_loader)
        average_train_loss_list.append(average_train_loss)
        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {average_train_loss}')
        BERT_model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                outputs = BERT_model(input_ids=batch[0],attention_mask=batch[1],labels=batch[2])
                total_test_loss += outputs.loss.item()

        average_test_loss = total_test_loss/len(test_loader)
        average_test_loss_list.append(average_test_loss)
        print(f'Epoch {epoch + 1}/{num_epochs}, Testing Loss: {average_test_loss}')

        checkpoint = {'epoch': epoch + 1,'model_state_dict': BERT_model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'train_loss': average_train_loss,'test_loss': average_test_loss,'train_loss_list': average_train_loss_list,'test_loss_list': average_test_loss_list}
        os.makedirs(checkpoint_directory, exist_ok=True)
        torch.save(checkpoint,os.path.join(checkpoint_directory, f'checkpoint_epoch_{epoch + 1}.pt'))

    BERT_model.save_pretrained(BERT_model_filepath)
    SavePKL(average_train_loss_list, average_train_loss_filename)
    SavePKL(average_test_loss_list, average_test_loss_filename)
    return BERT_model,average_train_loss_list,average_test_loss_list

#################################################################################################################
### Main Functions

#The main function is the driver for the code
if __name__ == "__main__":
    
    #Variables
    ESV_Bible_DF_filepath = 'Additional_Data/ESV_Bible_DF.csv'
    ESV_Bible_DF_tokenized_filepath = 'Additional_Data/ESV_Bible_DF_tokenized.pt'
    train_loader_filepath = 'Additional_Data/BERT_train_loader.pth'
    test_loader_filepath = 'Additional_Data/BERT_test_loader.pth'
    BERT_model_filepath = 'Models/BERT_model_trained'
    average_train_loss_filename = 'Additional_Data/BERT_training_loss.pkl'
    average_test_loss_filename = 'Additional_Data/BERT_test_loss.pkl'
    checkpoint_directory = 'Additional_Data/BERT_model_checkpoints'
    
    create_or_load_string_Bible_DF_tokenized = 'Load'
    create_or_load_string_train_and_test_loader = 'Load'
    
    #Main Code
    ESV_Bible_DF = LoadDF(ESV_Bible_DF_filepath)
    ESV_Bible_DF_tokenized = CreateOrLoadTokenizedESVBibleDF(ESV_Bible_DF,ESV_Bible_DF_tokenized_filepath,create_or_load_string_Bible_DF_tokenized)
    train_loader,test_loader = CreateOrLoadTrainAndTestLoader(ESV_Bible_DF_tokenized,train_loader_filepath,test_loader_filepath,create_or_load_string_train_and_test_loader)
    BERT_model,average_train_loss,average_test_loss = CreateTrainedBERTModel(train_loader,test_loader,BERT_model_filepath,average_train_loss_filename,average_test_loss_filename)
