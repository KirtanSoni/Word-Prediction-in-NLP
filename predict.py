from tensorflow.keras.models import load_model
import numpy as np
import pickle

# Load the model and tokenizer

def predict_next_words(model, tokenizer, text):
    """
        In this function we are using the tokenizer and models trained
        and we are creating the sequence of the text entered and then
        using our model to predict and return the the predicted word.
    
    """
    sequence = tokenizer.texts_to_sequences([text])[0]
    sequence = np.array([sequence])
    
    try :
        preds = model.predict(sequence)
        preds = np.argmax(preds,axis=1)
        #print(preds)
        predicted_word = ""
    except :    
        return "-"
    for key, value in tokenizer.word_index.items():
        if value == preds:
            predicted_word = key
            break
    
    #print(predicted_word)
    return predicted_word