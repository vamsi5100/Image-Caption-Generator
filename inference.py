from PIL import Image
import sys
import matplotlib.pyplot as plt
import tensorflow
import numpy as np
import cv2
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import pickle
path = sys.argv[1]
os.chdir(path)
Data = pickle.load(open('data.pkl','rb'))
features = pickle.load(open('features.pkl','rb'))
tokenizer = pickle.load(open('tokenizer.pkl','rb'))
model = tensorflow.keras.models.load_model('best_model.h5')
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None
def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq '
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == ' endseq':
            break     
    return in_text
def generate_caption(image_name):
    img_path = os.path.join(path,'Images',image_name)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    captions = Data[image_name]
    print('---------------------Actual---------------------')
    for caption in captions:
        print(caption)
    # predict the caption
    y_pred = predict_caption(model, features[image_name], tokenizer, 30)
    print('--------------------Predicted--------------------')
    print(y_pred)
    cv2.imshow('image',image)
    cv2.waitKey(1)
    cv2.destroyAllWindows()

generate_caption(sys.argv[2])