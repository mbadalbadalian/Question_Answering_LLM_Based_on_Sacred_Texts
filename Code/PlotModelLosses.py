from TrainMostRelevantVerseModelBible import BibleDataset,collate_fn
import pickle
import matplotlib.pyplot as plt

#################################################################################################################
### Other Functions

#LoadPKL Function
def LoadPKL(data_filename):
    with open(data_filename,'rb') as file:
        data = pickle.load(file)
    return data

#PlotModelLoss Function
def PlotModelLoss(train_loss,test_loss,title):
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, 'b', label='Training Loss')
    plt.plot(epochs, test_loss, 'r', label='Testing Loss')
    plt.title(title)
    plt.xlabel('Number of epochs')
    plt.ylabel('Model Loss')
    plt.legend()
    plt.show()
    return

#################################################################################################################
### Main Functions

#The main function is the driver for the code
if __name__ == "__main__":
    
    #Main Code
    Bible_DistilBERT_fine_tuned_training_loss = LoadPKL('Additional_Data/DistilBERT_fine_tuned_training_loss_Bible.pkl')
    Bible_DistilBERT_fine_tuned_testing_loss = LoadPKL('Additional_Data/DistilBERT_fine_tuned_test_loss_Bible.pkl')

    Bible_BERT_most_relevant_verse_fine_tuned_training_loss = LoadPKL('Additional_Data/BERT_most_relevant_verse_fine_tuned_training_loss_Bible.pkl')
    Bible_BERT_most_relevant_verse_fine_tuned_testing_loss = LoadPKL('Additional_Data/BERT_most_relevant_verse_fine_tuned_test_loss_Bible.pkl')

    Gita_DistilBERT_fine_tuned_training_loss = LoadPKL('Additional_Data/DistilBERT_fine_tuned_training_loss_Gita.pkl')
    Gita_DistilBERT_fine_tuned_testing_loss = LoadPKL('Additional_Data/DistilBERT_fine_tuned_test_loss_Gita.pkl')

    Gita_BERT_most_relevant_verse_fine_tuned_training_loss = LoadPKL('Additional_Data/BERT_most_relevant_verse_fine_tuned_training_loss_Gita.pkl')
    Gita_BERT_most_relevant_verse_fine_tuned_testing_loss = LoadPKL('Additional_Data/BERT_most_relevant_verse_fine_tuned_test_loss_Gita.pkl')

    PlotModelLoss(Bible_DistilBERT_fine_tuned_training_loss,Bible_DistilBERT_fine_tuned_testing_loss,'Question-Answer Loss Using DistilBERT on The Bible')
    PlotModelLoss(Bible_BERT_most_relevant_verse_fine_tuned_training_loss,Bible_BERT_most_relevant_verse_fine_tuned_testing_loss,'Question-Most Relevant Verse Loss Using BERT on The Bible')
    PlotModelLoss(Gita_DistilBERT_fine_tuned_training_loss,Gita_DistilBERT_fine_tuned_testing_loss,'Question-Answer Loss Using DistilBERT on The Gita')
    PlotModelLoss(Gita_BERT_most_relevant_verse_fine_tuned_training_loss,Gita_BERT_most_relevant_verse_fine_tuned_testing_loss,'Question-Most Relevant Verse Loss Using BERT on The Gita')


