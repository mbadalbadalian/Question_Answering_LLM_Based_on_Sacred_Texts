import pandas as pd
from transformers import BertTokenizer,BertModel
import torch
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split
import os
import pickle
import torch.nn.utils.rnn as rnn_utils

#################################################################################################################
### Other Functions

def LoadDF(DF_path):
    dataframe = pd.read_csv(DF_path)
    return dataframe

def SavePKL(data, data_filename):
    with open(data_filename, 'wb') as file:
        pickle.dump(data,file)
    return

def LoadPKL(data_filename):
    with open(data_filename, 'rb') as file:
        data = pickle.load(file)
    return data

class BibleDataset(Dataset):
    def __init__(self,data,tokenizer,model):
        self.data = data
        self.tokenizer = tokenizer
        self.model = model

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        verse = self.data.iloc[index]['Text']
        encoded_verse = self.tokenizer(verse,return_tensors='pt',padding=True,truncation=True)

        input_ids = encoded_verse['input_ids'].squeeze()
        attention_mask = encoded_verse['attention_mask'].squeeze()

        with torch.no_grad():
            target_embedding = self.model(**encoded_verse).last_hidden_state[:,0,:]
        item = {'verse':verse,'input_ids':input_ids,'attention_mask':attention_mask,'target_embedding':target_embedding}
        return item
    
def LoadBERTModel():
    model = BertModel.from_pretrained('bert-base-uncased')
    return model

def CreateTrainAndTestLoader(prepared_Q_and_A_DF, model, train_loader_filepath, test_loader_filepath):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_data, test_data = train_test_split(prepared_Q_and_A_DF,test_size=0.3,random_state=20777980)

    train_dataset = BibleDataset(train_data, tokenizer, model)
    test_dataset = BibleDataset(test_data, tokenizer, model)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, CollateFunction=CollateFunction)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, CollateFunction=CollateFunction)
    torch.save(train_loader, train_loader_filepath)
    torch.save(test_loader, test_loader_filepath)
    return train_loader, test_loader

def LoadTrainAndTestLoader(train_loader_filepath, test_loader_filepath):
    train_loader = torch.load(train_loader_filepath)
    test_loader = torch.load(test_loader_filepath)
    return train_loader, test_loader

def CreateOrLoadTrainAndTestLoader(prepared_Q_and_A_DF,model,train_loader_filepath,test_loader_filepath,create_or_load_string_train_and_test_loader='load'):
    if create_or_load_string_train_and_test_loader in ['Create','create']:
        train_loader,test_loader = CreateTrainAndTestLoader(prepared_Q_and_A_DF,model,train_loader_filepath,test_loader_filepath)
    else:
        train_loader,test_loader = LoadTrainAndTestLoader(train_loader_filepath,test_loader_filepath)
    return train_loader,test_loader

def CollateFunction(batch):
    batch = sorted(batch, key=lambda x: len(x['input_ids']), reverse=True)

    verses = [sample['verse'] for sample in batch]
    input_ids = [sample['input_ids'] for sample in batch]
    attention_masks = [sample['attention_mask'] for sample in batch]
    target_embeddings = [sample['target_embedding'] for sample in batch]

    packed_input_ids = rnn_utils.pack_sequence(input_ids, enforce_sorted=False)
    packed_attention_masks = rnn_utils.pack_sequence(attention_masks, enforce_sorted=False)
    new_batch = {'verses': verses, 'packed_input_ids': packed_input_ids, 'packed_attention_masks': packed_attention_masks, 'target_embedding': target_embeddings}
    return new_batch

def CreateBERTModelFineTuned(model,train_loader,test_loader,BERT_model_fine_tuned_filepath,average_train_loss_filename,average_test_loss_filename,num_epochs=5):
    learning_rate = 1e-5
    checkpoint_filepath = BERT_model_fine_tuned_filepath + '_checkpoint.pth'
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    if os.path.exists(checkpoint_filepath):
        checkpoint = torch.load(checkpoint_filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint['train_losses']
        test_losses = checkpoint['test_losses']
    else:
        start_epoch = 0
        train_losses = []
        test_losses = []

    for epoch in range(start_epoch,num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            packed_input_ids, attention_masks, target_embeddings = batch['packed_input_ids'], batch['packed_attention_masks'], batch['target_embedding']
            optimizer.zero_grad()
            unpacked_input_ids, _ = rnn_utils.pad_packed_sequence(packed_input_ids,batch_first=True)
            attention_masks, _ = rnn_utils.pad_packed_sequence(attention_masks,batch_first=True)
            predicted_embeddings = model(unpacked_input_ids, attention_mask=attention_masks).last_hidden_state[:,0,:]
            target_embeddings = torch.stack(target_embeddings).squeeze(1) 
            loss = criterion(predicted_embeddings, target_embeddings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss/len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {average_loss}")
        train_losses.append(average_loss)

        model.eval()
        test_loss = 0

        with torch.no_grad():
            for batch in test_loader:
                packed_input_ids,attention_masks,target_embeddings = batch['packed_input_ids'], batch['packed_attention_masks'], batch['target_embedding']
                unpacked_input_ids, _ = rnn_utils.pad_packed_sequence(packed_input_ids, batch_first=True)
                attention_masks, _ = rnn_utils.pad_packed_sequence(attention_masks, batch_first=True)
                predicted_embeddings = model(unpacked_input_ids, attention_mask=attention_masks).last_hidden_state[:,0,:]
                target_embeddings = torch.stack(target_embeddings).squeeze(1)
                loss = criterion(predicted_embeddings, target_embeddings)
                test_loss += loss.item()

        average_test_loss = test_loss/len(test_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Testing Loss: {average_test_loss}")
        test_losses.append(average_test_loss)

        torch.save({'epoch': epoch,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'train_losses': train_losses,'test_losses': test_losses}, checkpoint_filepath)

    SavePKL(train_losses,average_train_loss_filename)
    SavePKL(test_losses,average_test_loss_filename)
    model.save_pretrained(BERT_model_fine_tuned_filepath)
    return model,train_losses,test_losses

if __name__ == "__main__":
    #Variables
    ESV_Bible_DF_filepath = 'Additional_Data/ESV_Bible_DF.csv'
    prepared_Q_and_A_DF_filepath = 'Additional_Data/Prepared_QandA_Dataset.csv'
    train_loader_filepath = 'Additional_Data/BERT_most_relevant_verse_fine_tuned_train_loader_fine_tuned.pth'
    test_loader_filepath = 'Additional_Data/BERT_most_relevant_verse_fine_tuned_test_loader_fine_tuned.pth'
    BERT_most_relevant_verse_model_fine_tuned_filepath = 'Models/BERT_most_relevant_verse_model_fine_tuned'
    average_train_loss_filename = 'Additional_Data/BERT_most_relevant_verse_fine_tuned_training_loss.pkl'
    average_test_loss_filename = 'Additional_Data/BERT_most_relevant_verse_fine_tuned_test_loss.pkl'
    create_or_load_string_train_and_test_loader = 'Create'
    num_epochs = 6

    ESV_Bible_DF = LoadDF(ESV_Bible_DF_filepath)
    prepared_Q_and_A_DF = LoadDF(prepared_Q_and_A_DF_filepath)

    model = LoadBERTModel()

    train_loader,test_loader = CreateOrLoadTrainAndTestLoader(prepared_Q_and_A_DF,model,train_loader_filepath,test_loader_filepath,create_or_load_string_train_and_test_loader)

    model,train_losses,test_losses = CreateBERTModelFineTuned(model,train_loader,test_loader,BERT_most_relevant_verse_model_fine_tuned_filepath,average_train_loss_filename,average_test_loss_filename,num_epochs)
