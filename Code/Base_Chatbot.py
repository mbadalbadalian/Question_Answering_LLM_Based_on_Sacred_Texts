#This code uses the Hugging Face 'deepset/sentence_bert' and Zero Shot Classification to run as a chatbot
from transformers import AutoTokenizer, AutoModel
from torch.nn import functional as F
import warnings

# To suppress all warnings
warnings.filterwarnings("ignore")


def zero_shot_classification(sentence,labels):
    # load the sentence-bert model from the HuggingFace model hub
    tokenizer = AutoTokenizer.from_pretrained('deepset/sentence_bert')
    model = AutoModel.from_pretrained('deepset/sentence_bert')


    # run inputs through model and batch encode the sentences and labels
    inputs = tokenizer.batch_encode_plus([sentence] + labels,
                                         return_tensors='pt',
                                         pad_to_max_length=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    output = model(input_ids, attention_mask=attention_mask)[0]
    #Take the mean over dimension 1 for all the outputs and labels
    sentence_rep = output[:1].mean(dim=1)
    label_reps = output[1:].mean(dim=1)

    # now find the labels with the highest cosine similarities to the input sentence
    similarities = F.cosine_similarity(sentence_rep, label_reps)
    #closest = similarities.argsort(descending=True)
    #for ind in closest:
    #    print(f'label: {labels[ind]} \t similarity: {similarities[ind]}')
    return labels,similarities

#Program to accept the best scores and responses
def responses(responses,scores):
    #Index containing the best score is selected
    score = scores.argmax()
    #If the score is surpasses the threshold print and return the appropriate score
    if scores[score]<0.35:
        print('Sorry!! Can you be more specific.')
        print('If your question is about the bible , please let me know')
        score = 0
    else:
        print(responses[score])
    return score

if __name__ == "__main__": 
    #Labels that we want a sentence to be mapped to 
    labels = ['greeting', 'creator', 'functions', 'What you do','architecture', 'How you work','weakness']
    #Sentence that we want mapped to a label
    zero_shot_classification('How do you do, dear sir?', labels)
