import tensorflow as tf # Biblioteca para ML e DL

from tensorflow import keras # Módulo Keras, API utilizada para construir e treinar modelos de DL

from tensorflow.keras.preprocessing.text import Tokenizer # Importa a classe 'Tokenizer' do módulo 'preprocessing.text'. Classe utilizada para converter textos em sequência de números (tokens).

senteces = [ #Frases utilizadas para treinar o Tokenizer
    'I love my dog',
    'I love my cat',
    'You love my dog!'
]

tokenizer = Tokenizer(num_words = 100) #Instância da classe Tokenizer

tokenizer.fit_on_texts(senteces) # Treina o Tokenizer com as frases fornecidas na lista senteces. Esse método analisa o texto, identifica as palavras únicas e atribui a cada uma delas um índice único (token), em que o índice é atribuído com base na frequência da palavra

word_index = tokenizer.word_index # Criação do dicionário, mapeando cada palavra única para um índice inteiro

print(word_index)