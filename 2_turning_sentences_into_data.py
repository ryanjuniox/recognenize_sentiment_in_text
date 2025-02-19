import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]

tokenizer = Tokenizer(num_words = 100, oov_token="<OOV>") # Esse atribuido de OOV são para sentenças que não foram indexadas, na lista original

tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences) # Este módulo cria sequências de tokens representando cada sentença

test = [ # Teste do OOV, usando sequência de tokens do tokenizer treinado com as sentences (fit_on_texts)
    'I continue love my dog',
    'I dont love my bird'
]
test_sequences = tokenizer.texts_to_sequences(test)


padded = pad_sequences(sequences, padding='post') # Um módulo do Keras que garante que todas as sequências de tokens tenham o mesmo comprimento. Esse processo é necessario pois para treinar redes neurais, todas as entradas devem ter o mesmo cumprimento.

print(word_index)
print("\n")
print(sequences)
print("\n")
print(padded)
