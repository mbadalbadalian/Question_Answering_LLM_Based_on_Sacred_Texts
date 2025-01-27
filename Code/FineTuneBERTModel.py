from torch.nn.utils.rnn import pad_sequence
import os
import torch
from torch.utils.data import DataLoader,Dataset
from transformers import BertForQuestionAnswering,BertTokenizer
from torch.optim import AdamW
import pandas as pd
import pickle

#################################################################################################################
### Other Functions

#LoadDF Function
def LoadDF(DF_path):
    dataframe = pd.read_csv(DF_path)
    return dataframe

#SavePKL Function
def SavePKL(data,data_filename):
    with open(data_filename,'wb') as file:
        pickle.dump(data,file)
    return

#LoadPKL Function
def LoadPKL(data_filename):
    with open(data_filename,'rb') as file:
        data = pickle.load(file)
    return data

#LoadBERTModel Function
def LoadBERTModel(BERT_model_filepath):
    BERT_model = BertForQuestionAnswering.from_pretrained(BERT_model_filepath)
    return BERT_model

#QADataset Function
class QADataset(Dataset):
    def __init__(self,questions,answers,b,c,v,tokenizer):
        self.questions = questions
        self.answers = answers
        self.b = b
        self.c = c
        self.v = v
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        question = str(self.questions[index])
        answer = str(self.answers[index])
        encoding = self.tokenizer.encode_plus(text=question,text_pair=answer,truncation=True,padding="longest",max_length=512,return_tensors="pt")
        item = {"input_ids":encoding["input_ids"].squeeze(),"attention_mask": encoding["attention_mask"].squeeze(),"start_positions": torch.tensor(0),"end_positions": torch.tensor(0)}
        return item

#CollateBatch Function
def CollateBatch(batch):
    input_ids = pad_sequence([item["input_ids"] for item in batch], batch_first=True)
    attention_mask = pad_sequence([item["attention_mask"] for item in batch], batch_first=True)
    start_positions = torch.stack([item["start_positions"] for item in batch])
    end_positions = torch.stack([item["end_positions"] for item in batch])
    new_batch = {"input_ids": input_ids,"attention_mask": attention_mask,"start_positions": start_positions,"end_positions": end_positions}
    return new_batch

#CreateTrainAndTestLoader Function
def CreateTrainAndTestLoader(prepared_Q_and_A_DF,train_loader_fine_tuned_filepath,test_loader_fine_tuned_filepath):
    train_size_value = 0.7
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    Q_and_A_dataset = QADataset(questions=prepared_Q_and_A_DF['Questions'].tolist(),answers=prepared_Q_and_A_DF['Answers'].tolist(),b=prepared_Q_and_A_DF['b'].tolist(),c=prepared_Q_and_A_DF['c'].tolist(),v=prepared_Q_and_A_DF['v'].tolist(),tokenizer=tokenizer)
    
    train_size = int(train_size_value*len(Q_and_A_dataset))
    test_size = len(Q_and_A_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(Q_and_A_dataset, [train_size, test_size])

    train_loader_fine_tuned = DataLoader(train_dataset, batch_size=8, shuffle=True, CollateFunction=CollateBatch)
    test_loader_fine_tuned = DataLoader(test_dataset, batch_size=8, shuffle=False, CollateFunction=CollateBatch)
    torch.save(train_loader_fine_tuned,train_loader_fine_tuned_filepath)
    torch.save(test_loader_fine_tuned,test_loader_fine_tuned_filepath)
    return train_loader_fine_tuned,test_loader_fine_tuned

#LoadTrainAndTestLoader Function
def LoadTrainAndTestLoader(train_loader_fine_tuned_filepath,test_loader_fine_tuned_filepath):
    train_loader_fine_tuned = torch.load(train_loader_fine_tuned_filepath)
    test_loader_fine_tuned = torch.load(test_loader_fine_tuned_filepath)
    return train_loader_fine_tuned,test_loader_fine_tuned

#CreateOrLoadTrainAndTestLoader Function
def CreateOrLoadTrainAndTestLoader(prepared_Q_and_A_DF,train_loader_fine_tuned_filepath,test_loader_fine_tuned_filepath,create_or_load_string_train_and_test_loader_fine_tuned='load'):
    if create_or_load_string_train_and_test_loader_fine_tuned in ['Create','create']:
        train_loader_fine_tuned,test_loader_fine_tuned = CreateTrainAndTestLoader(prepared_Q_and_A_DF,train_loader_fine_tuned_filepath,test_loader_fine_tuned_filepath)
    else:
        train_loader_fine_tuned,test_loader_fine_tuned = LoadTrainAndTestLoader(train_loader_fine_tuned_filepath,test_loader_fine_tuned_filepath)
    return train_loader_fine_tuned,test_loader_fine_tuned

#CreateBERTModelFineTuned Function
def CreateBERTModelFineTuned(BERT_model_fine_tuned, train_loader_fine_tuned, test_loader_fine_tuned, BERT_model_fine_tuned_filepath):
    num_epochs = 5

    checkpoint_filepath = BERT_model_fine_tuned_filepath + '_checkpoint.pth'
    if os.path.exists(checkpoint_filepath):
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_filepath)
        else:
            checkpoint = torch.load(checkpoint_filepath, map_location=torch.device('cpu'))
        start_epoch = checkpoint.get('epoch',0)
        BERT_model_fine_tuned.load_state_dict(checkpoint['model_state_dict'])
        optimizer = AdamW(BERT_model_fine_tuned.parameters(),lr=5e-5)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        average_train_loss_fine_tuned_list = checkpoint.get('train_loss_list', [])
        average_test_loss_fine_tuned_list = checkpoint.get('test_loss_list', [])
    else:
        start_epoch = 0
        optimizer = AdamW(BERT_model_fine_tuned.parameters(), lr=5e-5)
        average_train_loss_fine_tuned_list = []
        average_test_loss_fine_tuned_list = []
        checkpoint = {}

    for epoch in range(start_epoch,num_epochs):
        BERT_model_fine_tuned.train()
        total_train_loss = 0.0
        for batch in train_loader_fine_tuned:
            optimizer.zero_grad()
            batch['input_ids'] = batch['input_ids'].squeeze(1)
            batch['attention_mask'] = batch['attention_mask'].squeeze(1)   
            outputs = BERT_model_fine_tuned(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        average_train_loss_fine_tuned = total_train_loss/len(train_loader_fine_tuned)
        average_train_loss_fine_tuned_list.append(average_train_loss_fine_tuned)
        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {average_train_loss_fine_tuned}')

        BERT_model_fine_tuned.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader_fine_tuned:
                batch['input_ids'] = batch['input_ids'].squeeze(1)
                batch['attention_mask'] = batch['attention_mask'].squeeze(1) 
                outputs = BERT_model_fine_tuned(**batch)
                total_test_loss += outputs.loss.item()
        average_test_loss_fine_tuned = total_test_loss/len(test_loader_fine_tuned)
        average_test_loss_fine_tuned_list.append(average_test_loss_fine_tuned)
        print(f'Epoch {epoch + 1}/{num_epochs}, Testing Loss: {average_test_loss_fine_tuned}')
        torch.save({'epoch': epoch + 1,'model_state_dict': BERT_model_fine_tuned.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'train_loss_list': average_train_loss_fine_tuned_list,'test_loss_list': average_test_loss_fine_tuned_list}, checkpoint_filepath)
    BERT_model_fine_tuned.save_pretrained(BERT_model_fine_tuned_filepath)
    SavePKL(average_train_loss_fine_tuned_list,average_train_loss_fine_tuned_filename)
    SavePKL(average_test_loss_fine_tuned_list,average_test_loss_fine_tuned_filename)
    return BERT_model_fine_tuned,average_train_loss_fine_tuned_list,average_test_loss_fine_tuned_list

#################################################################################################################
### Main Functions

if __name__ == "__main__":
    
    #Variables
    prepared_Q_and_A_DF_filepath = 'Additional_Data/Prepared_QandA_Dataset.csv'
    BERT_model_filepath = 'Models/BERT_model_trained'
    train_loader_fine_tuned_filepath = 'Additional_Data/BERT_fine_tuned_train_loader_fine_tuned.pth'
    test_loader_fine_tuned_filepath = 'Additional_Data/BERT_fine_tuned_test_loader_fine_tuned.pth'
    BERT_model_fine_tuned_filepath = 'Models/BERT_model_fine_tuned'
    average_train_loss_fine_tuned_filename = 'Additional_Data/BERT_fine_tuned_training_loss.pkl'
    average_test_loss_fine_tuned_filename = 'Additional_Data/BERT_fine_tuned_test_loss.pkl'
    
    create_or_load_string_train_and_test_loader_fine_tuned = 'Create'
    
    #Main Code
    prepared_Q_and_A_DF = LoadDF(prepared_Q_and_A_DF_filepath)
    BERT_model = LoadBERTModel(BERT_model_filepath)
    train_loader_fine_tuned,test_loader_fine_tuned = CreateOrLoadTrainAndTestLoader(prepared_Q_and_A_DF,train_loader_fine_tuned_filepath,test_loader_fine_tuned_filepath,create_or_load_string_train_and_test_loader_fine_tuned)
    BERT_model_fine_tuned,average_train_loss_fine_tuned_list,average_test_loss_fine_tuned_list = CreateBERTModelFineTuned(BERT_model,train_loader_fine_tuned,test_loader_fine_tuned,BERT_model_fine_tuned_filepath)
