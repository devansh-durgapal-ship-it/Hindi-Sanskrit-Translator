import pandas as pd
import numpy as np
import tensorflow 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Input, RepeatVector, TimeDistributed
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

def tokenize_texts(input_text, output_text):
    input_tokenizer = Tokenizer(num_words=500, oov_token="<Nothing>")
    output_tokenizer = Tokenizer(num_words=500, oov_token="<Nothing>")

    input_tokenizer.fit_on_texts(input_text)
    output_tokenizer.fit_on_texts(output_text)

    input_sequences = input_tokenizer.texts_to_sequences(input_text)
    output_sequences = output_tokenizer.texts_to_sequences(output_text)
    return input_sequences, output_sequences, input_tokenizer, output_tokenizer

def padding(input_text, output_text, max_input_len=None, max_output_len=None):
    if max_input_len is None:
        max_input_len = max(len(i) for i in input_text)
    if max_output_len is None:
        max_output_len = max(len(i) for i in output_text)
    
    input_text_pad = pad_sequences(input_text, maxlen=max_input_len, padding='post')
    output_text_pad = pad_sequences(output_text, maxlen=max_output_len, padding='post')
    return input_text_pad, output_text_pad, max_input_len, max_output_len

def build_model(input_vocab_size, output_vocab_size, input_length, output_length, optimizer="adam"):
    # Encoder
    inputs = Input(shape=(input_length,))
    embedding = Embedding(input_dim=min(input_vocab_size, 500), output_dim=128)(inputs)
    embedding = Dropout(0.35)(embedding)
    encoder = LSTM(256, return_state=True)
    encoder_outputs, state_h, state_c = encoder(embedding)
    
    # Decoder
    decoder_inputs = RepeatVector(output_length)(encoder_outputs)
    decoder = LSTM(256, return_sequences=True)(decoder_inputs, initial_state=[state_h, state_c])
    decoder = Dropout(0.35)(decoder)
    decoder = LSTM(256, return_sequences=True)(decoder)
    decoder = Dropout(0.35)(decoder)
    
    # Output layer - predict vocab for each position
    outputs = TimeDistributed(Dense(output_vocab_size, activation='softmax'))(decoder)
    
    model = Model(inputs=inputs, outputs=outputs)
    optimizer_obj = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer_obj, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    df = pd.read_csv("ULCA_merged_output.csv", usecols=['Hindi', 'Sanskrit'])
    x, y = df['Hindi'].astype(str), df['Sanskrit'].astype(str)

    input_sequences, output_sequences, input_tokenizer, output_tokenizer = tokenize_texts(x, y)
    input_padded, output_padded, max_input_len, max_output_len = padding(input_sequences, output_sequences)

    # Use FULL output sequences (all tokens), not just the first token
    # This preserves the entire Sanskrit translation for training
    y_labels = output_padded

    input_vocab_size = min(len(input_tokenizer.word_index) + 1, 500)
    output_vocab_size = min(len(output_tokenizer.word_index) + 1, 500)

    trained_model = build_model(
        input_vocab_size=input_vocab_size,
        output_vocab_size=output_vocab_size,
        input_length=input_padded.shape[1],
        output_length=output_padded.shape[1]
    )

    print(f"Model Summary:")
    trained_model.summary()
    print(f"Input vocab size: {input_vocab_size}, Output vocab size: {output_vocab_size}")
    print(f"Input sequence length: {max_input_len}, Output sequence length: {max_output_len}")
    print(f"Training samples: {input_padded.shape[0]}")

    callback = tensorflow.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    history = trained_model.fit(
        input_padded,
        y_labels,
        epochs=50,
        batch_size=16,
        validation_split=0.2,
        callbacks=[callback],
        verbose=1
    )

    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Save model
    trained_model.save('hindi_sanskrit_model.h5')
    print("Model saved to hindi_sanskrit_model.h5")
    print("DONE")