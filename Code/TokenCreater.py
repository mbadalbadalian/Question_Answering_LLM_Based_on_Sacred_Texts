#Code which reads the Bible and breaks it into lemmatized nouns, proper nouns and verbs and returns them to be written
import spacy
import PrepareBibleData


def Tokenizer(sentence):
    # List of lines to compare
    # Load the SpaCy English language model
    nlp = spacy.load("en_core_web_sm")
    # Process the text with SpaCy
    lemmatized_text = ' '.join([token.lemma_  for token in nlp(sentence)])
     # Process the text with SpaCy
    doc = nlp(lemmatized_text)
    # Extract nouns and verbs from the tagged words
    nouns = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN","VERB"]]
    return nouns

#CreateBibleTokens Function
def CreateBibleTokens(json_filename):
    Bible = PrepareBibleData.GetBibleLines()
    #Loop through each book, chapter and verse to get tokenized verse
    ESV_Bible_tokens = {}
    #Loop to iterate through each book in the Bible
    for book in Bible.keys(): 
        #print(book)
        ESV_Bible_tokens[book] = {}
        #Run through each chapter in the book
        for chapter in Bible[book].keys(): 
            ESV_Bible_tokens[book][chapter] = {}
            #Loop to iterate through each verse in a chapter
            for verse in Bible[book][chapter].keys():
                ESV_Bible_tokens[book][chapter][verse] = Tokenizer(Bible[book][chapter][verse])
    #Writes the dictionary into a JSON file
    PrepareBibleData.SaveJSONData(ESV_Bible_tokens,json_filename)
    return ESV_Bible_tokens

#CreateOrLoad Function
def CreateOrLoad(json_filename,create_or_load_string='load'):
    if create_or_load_string in ['Create','create']:
        ESV_Bible_tokens = CreateBibleTokens(json_filename)
    else:
        ESV_Bible_tokens = PrepareBibleData.LoadJSONData(json_filename)
    return ESV_Bible_tokens

#The main function is the driver for the code
if __name__ == "__main__":  
    #Variables
    json_filename = "Additional_Data\\ESV_Bible_Tokens.json"
    create_or_load_string = 'Create'
    #Main Code
    ESV_Bible_tokens = CreateOrLoad(json_filename,create_or_load_string)