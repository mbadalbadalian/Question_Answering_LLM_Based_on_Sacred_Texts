import pandas as pd
from transformers import pipeline,BertTokenizer,BertModel,DistilBertForQuestionAnswering, DistilBertTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import warnings

#################################################################################################################
### Other Functions

#Ignoring warnings
warnings.filterwarnings("ignore")

#Creating a model trained on Gita (very similar script)

def LoadDF(DF_path):
    dataframe = pd.read_csv(DF_path)
    return dataframe

#LoadContext Function
def LoadContext(Bible_DF):
    #Gets the Bible text
    context = Bible_DF['Text'].str.cat(sep=' ')
    return context

#LoadBertModelAndTokenizer Function
def LoadBertModelAndTokenizer(BERT_most_relevant_verse_model_fine_tuned_filepath):
    #Gets the pretrained BERT model and tokenizer
    model = BertModel.from_pretrained(BERT_most_relevant_verse_model_fine_tuned_filepath)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model,tokenizer

#LoadDistilbertModelAndTokenizer Function
def LoadDistilbertModelAndTokenizer(ditilBERT_model_fine_tuned_filepath):
    #Gets the pretrained DistilBERT model and tokenizer
    fine_tuned_model = DistilBertForQuestionAnswering.from_pretrained(ditilBERT_model_fine_tuned_filepath)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased-distilled-squad')
    return fine_tuned_model,tokenizer

#LoadBibleDFAndText Function
def LoadBibleDFAndText(Gita_filepath):
    ESV_Bible_DF = pd.read_csv(Gita_filepath)
    ESV_Bible_DF = ESV_Bible_DF[ESV_Bible_DF['c']  == 17][ESV_Bible_DF['v']  == 23]
    
    #Gets a list of the Bible verses
    ESV_Bible_text = ESV_Bible_DF['Text'].tolist()
    return ESV_Bible_DF,ESV_Bible_text

#CreateBibleEmbeddings Function
def CreateBibleEmbeddings(ESV_Bible_text,model,tokenizer,file_path):
    #Encodes the Bible using the tokenizer
    encoded_bible = tokenizer(ESV_Bible_text,return_tensors='pt',padding=True,truncation=True)

    with torch.no_grad():
        bible_embeddings = model(**encoded_bible).last_hidden_state[:, 0, :]
    #Saves the Bible embeddings
    torch.save(bible_embeddings,file_path)
    return bible_embeddings

#LoadBibleEmbeddings Function
def LoadBibleEmbeddings(bible_embeddings,file_path):
    #Loads the Bible embeddings
    bible_embeddings = torch.load(file_path)
    return bible_embeddings

#CreateOrLoadBibleEmbeddings Function
def CreateOrLoadBibleEmbeddings(ESV_Bible_text,model,tokenizer,file_path,create_or_load_string_embeddings='Load'):
    if create_or_load_string_embeddings in ['Create','create']:
        bible_embeddings = CreateBibleEmbeddings(ESV_Bible_text,model,tokenizer,file_path)
    else:
        bible_embeddings = LoadBibleEmbeddings(bible_embeddings,file_path)
    return bible_embeddings

#ProcessQuestion Function
def ProcessQuestion(question,model,tokenizer,bible_embeddings,ESV_Bible_DF):
    encoded_question = tokenizer(question, return_tensors='pt', padding=True, truncation=True)

    #Embeds the question
    with torch.no_grad():
        question_embedding = model(**encoded_question).last_hidden_state[:, 0, :]

    #Calculates the cosine similarity between the question the Bible embeddings
    similarities = cosine_similarity(question_embedding, bible_embeddings)

    #Gets the most similar verse
    most_similar_verse_index = similarities.argmax().item()
    metadata = ESV_Bible_DF.iloc[most_similar_verse_index][['c','v']]
    chapter, verse = metadata['c'], metadata['v']
    answer = f"{ESV_Bible_DF.iloc[most_similar_verse_index]['Text']} ({chapter}:{verse})"
    return answer

#AnswerQuestion Function
def AnswerQuestion(question,fine_tuned_model,tokenizer,context):
    #Generates the answer to the question
    question_answering_pipeline = pipeline('question-answering', model=fine_tuned_model, tokenizer=tokenizer)
    result = question_answering_pipeline(context=context, question=question)
    answer = result['answer']
    return answer

#GetInfo Function
def GetInfo():
    #Variables
    Gita_filepath = 'Additional_Data/Gita_DF.csv'
    BERT_most_relevant_verse_model_fine_tuned_filepath = 'Models/Gita_BERT_most_relevant_verse_model_fine_tuned'
    ditilBERT_model_fine_tuned_filepath = 'Models/Gita_DistilBERT_model_fine_tuned'
    bible_embeddings_filepath = 'Additional_Data/Gita_embeddings_original.pt'
    create_or_load_string_embeddings = 'Create'
    
    #Main code
    #Load the dataframes
    ESV_Bible_DF = LoadDF(Gita_filepath)
    
    #Load the models and tokenizers
    BERTmodel,BERTtokenizer = LoadBertModelAndTokenizer(BERT_most_relevant_verse_model_fine_tuned_filepath)
    distilBERTmodel,distilBERTtokenizer = LoadDistilbertModelAndTokenizer(ditilBERT_model_fine_tuned_filepath)

    #Get the useful information
    ESV_Bible_DF,ESV_Bible_text = LoadBibleDFAndText(Gita_filepath)
    bible_embeddings = CreateOrLoadBibleEmbeddings(ESV_Bible_text,BERTmodel,BERTtokenizer,bible_embeddings_filepath,create_or_load_string_embeddings)
    LoadBibleEmbeddings(bible_embeddings,'Additional_Data/Gita_embeddings_original.pt')
    context = LoadContext(ESV_Bible_DF)
    return BERTmodel,BERTtokenizer,bible_embeddings,ESV_Bible_DF,distilBERTmodel,distilBERTtokenizer,context

#GetAnswer Function
def GetAnswer(question,BERTmodel,BERTtokenizer,bible_embeddings,ESV_Bible_DF,distilBERTmodel,distilBERTtokenizer,context):
    #Get the most relevant verse
    answerBERT = ProcessQuestion(question,BERTmodel,BERTtokenizer,bible_embeddings,ESV_Bible_DF)
    
    #Get the generated answer
    answerDistilBERT = AnswerQuestion(question,distilBERTmodel,distilBERTtokenizer,context)
    print(f'Question: {question}')
    print(f'The answer is: {answerDistilBERT}.'f' The most relevant verse found was: {answerBERT}')
    print()
    return