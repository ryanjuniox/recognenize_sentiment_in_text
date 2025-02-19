**Repositório GitHub: Detecção de Sarcasmo em Manchetes de Notícias**

Este repositório contém um projeto de aprendizado de máquina focado na detecção de sarcasmo em manchetes de notícias. O modelo foi desenvolvido utilizando TensorFlow e Keras, e é capaz de classificar manchetes como sarcásticas ou não sarcásticas com base em um conjunto de dados disponível no arquivo `sarcasm.json`.
Todo o código e aprendizado foi com base na playlist de vídeos 'Natural Language Processing' do canal oficial do TensorFlow.

### **Descrição do Projeto**

O objetivo deste projeto é construir e treinar um modelo de rede neural que possa identificar sarcasmo em manchetes de notícias. O sarcasmo é uma forma de linguagem complexa e muitas vezes difícil de ser detectada por máquinas, mas com técnicas de processamento de linguagem natural (NLP) e aprendizado profundo, é possível criar modelos que realizam essa tarefa com uma precisão razoável.

### **Estrutura do Código**

1. **Carregamento dos Dados**:
   - Os dados são carregados a partir de um arquivo JSON (`sarcasm.json`), que contém manchetes de notícias, labels indicando se a manchete é sarcástica ou não, e links para os artigos originais.

2. **Pré-processamento**:
   - As manchetes são tokenizadas e convertidas em sequências de inteiros.
   - As sequências são preenchidas (`padding`) ou truncadas para garantir que todas tenham o mesmo comprimento.

3. **Construção do Modelo**:
   - O modelo é uma rede neural sequencial que inclui uma camada de embedding, uma camada de pooling global, e duas camadas densas.
   - A camada de embedding converte as palavras em vetores de dimensão fixa.
   - A camada de pooling reduz a dimensionalidade dos dados.
   - As camadas densas são responsáveis pela classificação final.

4. **Treinamento**:
   - O modelo é treinado utilizando a função de perda `binary_crossentropy` e o otimizador `adam`.
   - O treinamento é realizado por 30 épocas, com validação em um conjunto de teste separado.

5. **Avaliação**:
   - A precisão do modelo é avaliada tanto no conjunto de treino quanto no conjunto de teste.


### **Dependências**

- Python 3.x
- TensorFlow 2.x
- NumPy
