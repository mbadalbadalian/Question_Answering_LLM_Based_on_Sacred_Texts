#This is the entire core of our algorithm that connects all three models and uses them in sync
import Base_Chatbot
import Sentiment_Analysis as Sentiment
import warnings
import AnswerUserBible,AnswerUserGita

# To suppress all warnings
warnings.filterwarnings("ignore")

#This function runs the Religion model for the Bible or Gita while paying attention to the user's satisfaction
def Religious_Questions(model,book,Bible_BERTmodel,Bible_BERTtokenizer,bible_embeddings,ESV_Bible_DF,bible_books,Bible_distilBERTmodel,Bible_distilBERTtokenizer,Bible_context,Gita_BERTmodel,Gita_BERTtokenizer,Gita_embeddings,Gita_DF,Gita_distilBERTmodel,Gita_distilBERTtokenizer,Gita_context):
    while(True):
        if book == 0:
            #Here the user asks a question in context to the Bible
            user_input = input('What is the question you wanted to ask about the Bible ESV edition?\n')
            #The model prints answer to user input question
            AnswerUserBible.GetAnswer(user_input,Bible_BERTmodel,Bible_BERTtokenizer,bible_embeddings,ESV_Bible_DF,bible_books,Bible_distilBERTmodel,Bible_distilBERTtokenizer,Bible_context)
        elif book == 1:
            #Here the user asks a question in context to the Gita
            user_input = input('What is the question you wanted to ask about the Bhagwad Gita translated by Swami Sivananda?\n')
            #The model prints answer to user input question
            AnswerUserGita.GetAnswer(user_input,Gita_BERTmodel,Gita_BERTtokenizer,Gita_embeddings,Gita_DF,Gita_distilBERTmodel,Gita_distilBERTtokenizer,Gita_context)
        #Accepts user input to check satisfaction
        user_satisfaction = input("Are you satisfied with the answer\n")
        #Finds whether the user feels positive or negative
        opinion = Sentiment.model_use(user_satisfaction, model)
        #If the user feels negative, loop the function else leave it
        if opinion == 'neg':
            print('I am sorry to hear that. Can you please repeat the question more concisely?')
        else:
            print('That\'s Great! Returning to general chatbot now')
            break
    return

if __name__ == "__main__":  
    #First load the sentiment analysis model
    model = Sentiment.sentiment()
    #Load the model, tokenizers, embeddings and other important parameter for the Bible
    Bible_BERTmodel,Bible_BERTtokenizer,bible_embeddings,ESV_Bible_DF,bible_books,Bible_distilBERTmodel,Bible_distilBERTtokenizer,Bible_context = AnswerUserBible.GetInfo()
    #Load the model, tokenizers, embeddings and other important parameter for the Bhagwad Gita
    Gita_BERTmodel,Gita_BERTtokenizer,Gita_embeddings,Gita_DF,Gita_distilBERTmodel,Gita_distilBERTtokenizer,Gita_context = AnswerUserGita.GetInfo()
    #A list of responses mapped to the indexes of a particular label
    responses = ['Hello! It is a pleasure meeting you',
                  'I was made by Chris Binoi Verghese and Matthew Badal-Badalian',
                  'I am supposed to answer verses from the Bible and Bhagwad Gita in response to user questions',
                  'I am supposed to answer verses from the Bible and Bhagwad Gita in response to user questions',
                  'I work using DistilBERT and Zero Shot Classification',
                  'I work using DistilBERT and Zero Shot Classification',
                  'Sure thing \nloading in the ML model now!!',
                   'Thank you for using our chatbot' ,
                   'Sure thing \nloading in the ML model now!!',
                   'Thank you for using our chatbot']
    #Labels of chatting topics with the chatbot which will cause a response of change in model
    labels1 = ['greeting', 'creator', 'functions', 'What you do','architecture', 'How you work', 'want to ask a question about the Bible','No', 'want to ask a question about the Bhagwad Gita','No Questions']
    
    print('If you want to ask questions about the Bible or Bhagwad Gita, let me know first')
    while(True):
    # Accepting string input from the user
        user_input = input("Any Questions: (Enter 'No' to leave)\n")
        #Get the scores of the input statement for every label
        labels, scores = Base_Chatbot.zero_shot_classification(user_input,labels1)
        #Get the response of the best score
        label_ind = Base_Chatbot.responses(responses,scores)
        #Label which indicates interest in knowing about the Bible
        if label_ind == 6:
            Religious_Questions(model,0,Bible_BERTmodel,Bible_BERTtokenizer,bible_embeddings,ESV_Bible_DF,bible_books,Bible_distilBERTmodel,Bible_distilBERTtokenizer,Bible_context,Gita_BERTmodel,Gita_BERTtokenizer,Gita_embeddings,Gita_DF,Gita_distilBERTmodel,Gita_distilBERTtokenizer,Gita_context)
        #Label which indicates interest in knowing about the Gita
        elif label_ind == 8:
            Religious_Questions(model,1,Bible_BERTmodel,Bible_BERTtokenizer,bible_embeddings,ESV_Bible_DF,bible_books,Bible_distilBERTmodel,Bible_distilBERTtokenizer,Bible_context,Gita_BERTmodel,Gita_BERTtokenizer,Gita_embeddings,Gita_DF,Gita_distilBERTmodel,Gita_distilBERTtokenizer,Gita_context)
        #Label which indicates the user's wish to stop the model
        elif label_ind == 7 or label_ind == 9:
            break


