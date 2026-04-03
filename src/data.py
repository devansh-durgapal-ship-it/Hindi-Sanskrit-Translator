import pandas as pd
import numpy as np
import tensorflow 
from tensorflow.keras.preprocessing.text import Tokenizer



def tokensizer(input_text, output_text):
    tokenizer = Tokenizer(oov_token= "<Nothing>")
    input_text_tokenizer= tokenizer.fit_on_texts(input_text)

    
if __name__ == "__main__":
    df= pd.read_csv("ULCA_merged_output.csv", usecols= ['Hindi', 'Sanskrit'])
    x,y = df['Hindi'].astype(str), df['Sanskrit'].astype(str)
    token= tokensizer(x,y)

    print("DONE")