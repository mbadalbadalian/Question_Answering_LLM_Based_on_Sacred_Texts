#Sentiment Analysis code which uses DistilBERT to understand the positive or negative nuances of a sentence
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
import os
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
# To suppress all warnings
warnings.filterwarnings("ignore")

def model_create():
    #Pretrained model from Hugging Face
    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

    # Inputs which can be used to train the accuracy of the model after training
    raw_inputs = [
        "This is not what I wanted here",
        "Thanks a lot",
        "Correct answer",
        "Nooooooooooo",
        "Repeat that step",
        "Nice work",
        "Redo everything",
        "I rejoice in the correctness of the answer"
        "Retry this algorithm"  
    ]

    # Fine-tuning data which will be used to finetune the pretrained model
    fine_tuning_data = [
        ("This is not what I want", 0), 
        ("Thanks", 1),
        ("Correct", 1),
        ("No", 0),
        ("Repeat",0),
        ("Nice",1),
        ("This is it!",1),
        ("Redo", 0),
        ("I rejoice in the answer",1),
        ("Retry", 0)  # Adding "Retry" as a negative example
    ]

    #Finetuning Data
    ########################################################################################################################
    #A large list of negative sentences to train the model and prevent overfitting
    fine_tuning_data.extend([
    ("I'm not happy with this", 0),
    ("Terrible", 0),
    ("Unsatisfied", 0),
    ("Displeased", 0),
    ("That was awful", 0),
    ("I'm dissatisfied", 0),
    ("Worst", 0),
    ("This is not right", 0),
    ("I'm disappointed", 0),
    ("Not good", 0),
    ("I hate it", 0),
    ("I dislike it", 0),
    ("This is a failure", 0),
    ("I'm not content", 0),
    ("This is frustrating", 0),
    ("Awful", 0),
    ("Very bad", 0),
    ("Not pleased", 0),
    ("I'm angry", 0),
    ("This is terrible", 0),
    ("I'm not satisfied", 0),
    ("I'm not happy", 0),
    ("This is unacceptable", 0),
    ("That was a mistake", 0),
    ("I'm furious", 0),
    ("This is wrong", 0),
    ("Disappointed", 0),
    ("Redo the algorithm",0),
    ("I'm not okay with this", 0),
    ("I'm upset", 0),
    ("This is not satisfactory", 0),
    ("I'm not pleased with this", 0),
    ("This is not acceptable", 0),
    ("I'm not content with this", 0),
    ("This is awful", 0),
    ("That's not good", 0),
    ("This is not what I wanted", 0),
    ("This is not what I expected", 0),
    ("I'm not happy with the answer", 0),
    ("I'm not satisfied with the answer", 0),
    ("This is not the answer I wanted", 0),
    ("I expected a better answer", 0),
    ("This is not a good answer", 0),
    ("This is not helpful", 0),
    ("I'm disappointed with the answer", 0),
    ("I'm not pleased with the answer", 0),
    ("Can you redo this?", 0),
    ("I think you should try again", 0),
    ("Please retry this", 0),
    ("Can you run it again?", 0),
    ("I'm not satisfied, please redo it", 0),
    ("This isn't what I wanted, try again", 0),
    ("This needs to be redone", 0),
    ("Could you retry this task?", 0),
    ("Please run the algorithm again", 0),
    ("I'm not happy with this, can you redo it?", 0),
    ("This needs another attempt", 0),
    ("I'm not pleased, please try again", 0),
    ("Can you try a different approach?", 0),
    ("Could you run it once more?", 0),
    ("I'm dissatisfied, please redo it", 0),
    ("Please retry the process", 0),
    ("This requires a redo", 0),
    ("Could you try another method?", 0),
    ("Please run this again", 0),
    ("I'm not content, can you redo it?", 0),
    ("This doesn't meet expectations, redo it", 0),
    ("Can you try once more?", 0),
    ("I'm disappointed, please retry", 0),
    ("This needs a retry", 0),
    ("Can you give it another shot?", 0),
    ("Please try again, this isn't right", 0),
    ("I'm not satisfied, can you redo it?", 0),
    ("This needs another run-through", 0),
    ("Can you rework this?", 0),
    ("This wasn't done properly, try again", 0),
    ("I'm not okay with this, redo it", 0),
    ("Could you run this one more time?", 0),
    ("This requires another attempt", 0),
    ("Please retry, it's not good enough", 0),
    ("I'm not pleased, can you redo it?", 0),
    ("This needs another try", 0),
    ("Can you reevaluate and redo?", 0),
    ("I'm dissatisfied, run it again", 0),
    ("This isn't acceptable, try again", 0),
    ("Could you give it another go?", 0),
    ("Please redo, this isn't what I expected", 0),
    ("I'm not happy, please retry", 0),
    ("This needs to be done again", 0),
    ("Can you retry, this isn't satisfactory", 0),
    ("Please redo, it's not what I wanted", 0),
    ("I'm not content, redo it", 0),
    ("This wasn't executed well, try again", 0),
    ("Can you rerun this task?", 0),
    ("This needs a rework", 0),
    ("Please try again, it's disappointing", 0),
    ("I'm not satisfied, another attempt", 0),
    ("Can you redo this work?", 0),
    ("This needs another round", 0),
    ("Please run it again, not acceptable", 0),
    ("I'm not pleased, please redo", 0),
    ("This isn't right, can you retry?", 0),
    ("Can you retry, this isn't satisfactory", 0),
    ("I'm not happy with this, redo it", 0),
    ("This needs another attempt", 0),
    ("Please redo, it's not what I wanted", 0),
    ("I'm not content with this, redo it", 0),
    ("This wasn't executed well, try again", 0),
    ("Can you rerun this task?", 0),
    ("This needs a rework", 0),
    ("Please try again, it's disappointing", 0),
    ("I'm not satisfied, another attempt", 0),
    ("Can you redo this work?", 0),
    ("This needs another round", 0),
    ("Please run it again, not acceptable", 0),
    ("I'm not pleased, please redo", 0),
    ("This isn't right, can you retry?", 0),
    ("Can you retry, this isn't satisfactory", 0),
    ("I'm not happy with this, redo it", 0),
    ("This needs another attempt", 0),
    ("Please redo, it's not what I wanted", 0),
    ("I'm not content with this, redo it", 0),
    ("This wasn't executed well, try again", 0),
    ("Can you rerun this task?", 0),
    ("This needs a rework", 0),
    ("Please try again, it's disappointing", 0),
    ("I'm not satisfied, another attempt", 0),
    ("Can you redo this work?", 0),
    ("This needs another round", 0),
    ("Please run it again, not acceptable", 0),
    ("I'm not pleased, please redo", 0),
    ("This isn't right, can you retry?", 0),
    ("Can you retry, this isn't satisfactory", 0),
    ("This is not a satisfactory answer", 0),
    ("I'm not content with the answer", 0),
    ("I'm upset with the answer", 0),
    ("This is not what I was looking for", 0),
    ("I'm not okay with the answer", 0),
    ("This is not the answer I was expecting", 0),
    ("That's not the right answer", 0),
    ("This is not what I needed", 0),
    ("I'm dissatisfied with the answer", 0),
    ("This is not a good response", 0),
    ("I'm not happy with this solution", 0),
    ("I'm not satisfied with this solution", 0),
    ("This is not the solution I wanted", 0),
    ("I expected a better solution", 0),
    ("This is not a good solution", 0),
    ("This is not a helpful solution", 0),
    ("I'm disappointed with this solution", 0),
    ("I'm not pleased with this solution", 0),
    ("This is not a satisfactory solution", 0),
    ("I'm not content with this solution", 0),
    ("I'm upset with this solution", 0),
    ("This is not what I was expecting as a solution", 0),
    ("That's not the right solution", 0),
    ("This is not what I needed as a solution", 0),
    ("I'm dissatisfied with this solution", 0),
    ("This is not a good response as a solution", 0)])
   #A large list of positive sentences to train the model and prevent overfitting
    fine_tuning_data.extend([
    ("I'm happy with this", 1),
    ("Great", 1),
    ("Satisfied", 1),
    ("Pleased", 1),
    ("Wonderful", 1),
    ("Awesome", 1),
    ("Good job", 1),
    ("I'm content", 1),
    ("This is satisfactory", 1),
    ("I'm delighted", 1),
    ("I'm pleased with this", 1),
    ("I'm really happy with this outcome", 1),
    ("Absolutely fantastic", 1),
    ("Very pleased", 1),
    ("I'm extremely satisfied", 1),
    ("Impressive", 1),
    ("This exceeded my expectations", 1),
    ("I'm thrilled with the result", 1),
    ("Incredible job", 1),
    ("Delighted", 1),
    ("Perfectly done", 1),
    ("Exactly what I was hoping for", 1),
    ("I couldn't be happier", 1),
    ("Brilliant work", 1),
    ("Remarkable", 1),
    ("I'm overjoyed", 1),
    ("This is outstanding", 1),
    ("Excellently executed", 1),
    ("This is top-notch", 1),
    ("Exceptional", 1),
    ("This is a masterful solution", 1),
    ("I'm extremely pleased", 1),
    ("This is beyond satisfactory", 1),
    ("I'm elated", 1),
    ("This is just perfect", 1),
    ("This solution is remarkable", 1),
    ("I'm thoroughly impressed", 1),
    ("This is exactly what I wanted", 1),
    ("This is excellent", 1),
    ("This is wonderfully done", 1),
    ("This solution is superb", 1),
    ("This is incredibly helpful", 1),
    ("I'm immensely satisfied", 1),
    ("This is outstanding work", 1),
    ("I'm really pleased with this answer", 1),
    ("This is exceptional work", 1),
    ("This is exceptionally good", 1),
    ("This is truly impressive", 1),
    ("This is amazing", 1),
    ("This solution is exceptional", 1),
    ("I'm extremely content", 1),
    ("This is brilliantly done", 1),
    ("This is just wonderful", 1),
    ("This is exactly what I needed", 1),
    ("This is brilliantly executed", 1),
    ("This is a fantastic solution", 1),
    ("This is terrific", 1),
    ("I'm really happy with this answer", 1),
    ("This is extremely beneficial", 1),
    ("This solution is incredibly helpful", 1),
    ("This is just outstanding", 1),
    ("This is exactly right", 1),
    ("This is exactly what I was looking for", 1),
    ("This is absolutely perfect", 1),
    ("This solution is exactly what I wanted", 1),
    ("This is truly remarkable", 1),
    ("This is just what I needed", 1),
    ("This is simply perfect", 1),
    ("This is absolutely wonderful", 1),
    ("This is really impressive", 1),
    ("This is exactly as expected", 1),
    ("This is really helpful", 1),
    ("This is just fantastic", 1),
    ("This is incredibly good", 1),
    ("This is really good work", 1),
    ("This is top-quality work", 1),
    ("This is incredibly beneficial", 1),
    ("This is an outstanding solution", 1),
    ("This is truly exceptional", 1),
    ("This is absolutely amazing", 1),
    ("This is just remarkable", 1),
    ("This solution is incredibly effective", 1),
    ("This is truly excellent", 1),
    ("This is exceptionally well done", 1),
    ("This is incredibly impressive", 1),
    ("This is exactly what I was hoping for", 1),
    ("This solution is top-notch", 1),
    ("This is absolutely fantastic", 1),
    ("This is really beneficial", 1),
    ("This is truly outstanding", 1),
    ("This is really impressive work", 1),
    ("This is brilliantly executed work", 1),
    ("This is simply exceptional", 1),
    ("This is incredibly impressive work", 1),
    ("This is just superb", 1),
    ("This is brilliantly accomplished", 1),
    ("This is just what I wanted", 1),
    ("This solution is just perfect", 1),
    ("This is incredibly effective", 1),
    ("This is really exceptional", 1),
    ("This is top-of-the-line", 1),
    ("This is top-quality", 1),
    ("This is top-level work", 1),
    ("This is incredibly well done", 1),
    ("This is really well executed", 1),
    ("This solution is exceptional in every way", 1),
    ("This is incredibly well thought out", 1),
    ("This is top-notch work", 1),
    ("This is really exceptional work", 1),
    ("This solution is truly phenomenal", 1),
    ("This is truly exceptional work", 1),
    ("This solution is top-of-the-line", 1),
    ("This is top-tier work", 1),
    ("This is absolutely top-notch", 1),
    ("This is truly exceptional work", 1),
    ("This solution is truly top-notch", 1),
    ("This is truly top-quality work", 1),
    ("This solution is absolutely phenomenal", 1),
    ("This is incredibly top-notch work", 1),
    ("This solution is absolutely outstanding", 1),
    ("This is absolutely top-tier work", 1),
    ("This is incredibly top-quality", 1),
    ("This is truly exceptional work", 1),
    ("This solution is absolutely top-of-the-line", 1),
    ("This is truly top-notch work", 1),
    ("This is what I wanted", 1),
    ("This is what I expected", 1),
    ("I'm happy with the answer", 1),
    ("I'm satisfied with the answer", 1),
    ("This is the answer I wanted", 1),
    ("This is a good answer", 1),
    ("This is helpful", 1),
    ("I'm pleased with the answer", 1),
    ("This is a satisfactory answer", 1),
    ("I'm content with the answer", 1),
    ("I'm happy with this solution", 1),
    ("I'm satisfied with this solution", 1),
    ("This is the solution I wanted", 1),
    ("This is a good solution", 1),
    ("This is a helpful solution", 1),
    ("I'm pleased with this solution", 1),
    ("This is a satisfactory solution", 1),
    ("I'm content with this solution", 1)])
########################################################################################################################

    #Auto tokenizer  from the pretrained DistilBERT
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    #Get the tokens from the inputs obtained for testing
    tokens = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="tf")
    inputs = (tokens["input_ids"], tokens["attention_mask"])

    # Process fine-tuning data
    fine_tuning_texts = [item[0] for item in fine_tuning_data]
    fine_tuning_labels = tf.convert_to_tensor([item[1] for item in fine_tuning_data])
    #Tokenize the fine tuning data as well using the pretrained model
    tokenizer_ft = AutoTokenizer.from_pretrained(checkpoint)
    inputs_ft = tokenizer_ft(fine_tuning_texts, padding=True, truncation=True, return_tensors="tf")

    # Split the data into training and validation sets
    input_ids = np.array(inputs_ft["input_ids"])
    attention_masks = np.array(inputs_ft["attention_mask"])
    #Randomly split the data into training and validation set
    train_input_ids, val_input_ids, train_attention_masks, val_attention_masks, train_labels, val_labels = train_test_split(
    input_ids, attention_masks, np.array(fine_tuning_labels), test_size=0.2, random_state=42)

    #Add the attention masks with the the training and validation
    train_inputs = (train_input_ids, train_attention_masks)
    val_inputs = (val_input_ids, val_attention_masks)

    # Convert input_ids and attention_mask to tensors
    train_input_ids_tensor = tf.convert_to_tensor(train_inputs[0])
    train_attention_mask_tensor = tf.convert_to_tensor(train_inputs[1])
    val_input_ids_tensor = tf.convert_to_tensor(val_inputs[0])
    val_attention_mask_tensor = tf.convert_to_tensor(val_inputs[1])

    # Fine-tune the model with additional data
    model_ft = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)
    #The optimizer is an Adam optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    #The loss is categorical Cross Entropy Loss
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    #Compile the finetuned model
    model_ft.compile(optimizer=optimizer, loss=loss)

    # Use the tensors for model training
    history = model_ft.fit(
        (train_input_ids_tensor, train_attention_mask_tensor),
        train_labels,
        epochs=5,
        validation_data=((val_input_ids_tensor, val_attention_mask_tensor), val_labels)
    )

    # Print training and validation losses for each epoch
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss For Sentiment Analysis Using finetuned DistilBERT')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Make predictions with the finetuned model
    outputs = model_ft(inputs)
    #Softmax the score to convert them into probabilities
    predictions = tf.math.softmax(outputs.logits, axis=-1)
    #Calculate the test accuracy of all raw_inputs
    sentiments = ['neg' if max(i) == i[0] else 'pos' for i in predictions]
    sentiments_correct = ['neg', 'pos', 'pos', 'neg', 'neg', 'pos', 'neg', 'pos', 'neg']
    # Calculate accuracy
    correct_predictions = sum(1 for pred, correct in zip(sentiments, sentiments_correct) if pred == correct)
    accuracy = correct_predictions / len(sentiments)
    #print("Test Accuracy:", accuracy)
    return model_ft

#This function runs the Sentiment analysis model on the input line to find its sentiment
def model_use(input_sentence, model):
    #Model from Hugging Face
    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    #Tokenize the input sentence
    tokens = tokenizer(input_sentence, padding=True, truncation=True, return_tensors="tf")
    inputs = (tokens["input_ids"], tokens["attention_mask"])
    #Get the predicted scores of the sentiments
    outputs = model(inputs)
    #Softmax the score to convert them into probabilities
    predictions = tf.math.softmax(outputs.logits, axis=-1)
    #print(['neg' if max(i) == i[0] else 'pos' for i in predictions])
     # Get the index of the maximum value in predictions[0]
    #The prediction with the highest probability is the correct one
    max_index = tf.argmax(predictions[0], axis=-1).numpy()
    # Assign sentiment based on the max_index
    sentiment = 'neg' if max_index == 0 else 'pos'
    return sentiment

#This function is called to load and return the Sentiment Analysis model
def sentiment():    
    # Check if the 'model' folder exists and contains a model
    model_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model")
    model_path = os.path.join(model_folder, "sentiment_model")
    #If the model exists
    if os.path.exists(model_folder):
        #Recreate the model by loading the weights onto the pretrained DistilBERT
        model = TFAutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')
        model.load_weights(model_path)
        print("Model loaded successfully.")

    else:
        #Finetune the model using set data and save the weights for future use
        model = model_create()
        model.save_weights(model_path)
        print("Model saved successfully.")
    return model

if __name__ == "__main__":  
    #Load the model
    model = sentiment()
    #Run a list of examples to see working of the model
    print("Demonstration of working of the Sentiment Analysis")
    print("This is not what I was looking for: " , model_use("This is not what I was looking for", model))
    print("Run Again: ",model_use("Run Again", model))
    print("This is perfect: ", model_use("This is perfect", model))
    print("That was incorrect: ", model_use("That was incorrect", model))
    print("I am satisfied:", model_use("I am satisfied", model))
 
