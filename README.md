# Recipe Generator Project README

## Table of Contents
1. [Introduction](#introduction)
2. [Data Preprocessing](#data-preprocessing)
3. [Embedding and Encoding](#embedding-and-encoding)
4. [Train-Test Split](#train-test-split)
5. [Building the Annoy Index](#building-the-annoy-index)
6. [Querying the Model](#querying-the-model)
7. [Generating New Recipes](#generating-new-recipes)
8. [Using Ollama](#using-ollama)
9. [Running the Script](#running-the-script)
10. [Acknowledgements](#acknowledgements)

## Introduction
This project is designed to generate new and creative recipes based on an existing dataset of recipes. It utilizes various machine learning models and libraries including `pandas`, `scikit-learn`, `SentenceTransformer`, and `Annoy`. Additionally, it leverages a Language Model from Ollama for querying and generating text.

## Data Preprocessing
The initial step involves loading the dataset and selecting a subset for the project:

```python
import pandas as pd

df = pd.read_csv("/Users/felixtong/Desktop/public_repo/recipe_generator/dataset/full_dataset.csv")
df = df[:100]
```

## Embedding and Encoding
We use SentenceTransformer to encode the text data. The `tokenize_and_encode` function is defined to convert text into embeddings:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def tokenize_and_encode(text):
    embedding = model.encode([text])
    return embedding[0]

# Apply the function to the dataset
df2['ingredients'] = df2['ingredients'].apply(tokenize_and_encode)
df2['directions'] = df2['directions'].apply(tokenize_and_encode)
df2['NER'] = df2['NER'].apply(tokenize_and_encode)
```

## Train-Test Split
The dataset is split into training and testing sets using `train_test_split` from scikit-learn:

```python
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df2, test_size=0.2)
```

## Building the Annoy Index
We use Annoy for efficient similarity search and indexing:

```python
from annoy import AnnoyIndex

ingredients_dim = len(ingredients_embeddings[0])
ingredients_t = AnnoyIndex(ingredients_dim, 'angular')

directions_dim = len(directions_embeddings[0])
directions_t = AnnoyIndex(directions_dim, 'angular')

for i, vector in enumerate(ingredients_embeddings):
    ingredients_t.add_item(i, vector)
ingredients_t.build(10)
ingredients_t.save('ingredients_test.ann')

for i, vector in enumerate(directions_embeddings):
    directions_t.add_item(i, vector)
directions_t.build(10)
directions_t.save('directions_test.ann')
```

## Querying the Model
To query the model, we define a function to run the LLM:

```python
import subprocess

def query_llm(query):
    command = ['ollama', 'run', 'llama3']
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    output, _ = process.communicate(input=query)
    process.wait()
    return output
```

## Generating New Recipes
We generate new recipes by combining ingredients and cooking directions with a creative twist using the queried LLM direction:

```python
prompt = f'''
Ingredient document: {ingredients_doc}
cooking direction: {directions_doc}
recipe format: {recipe_format}
required ingredient: {query}

Imagine you are a creative and trustworthy recipe generator acting as an assistant...
'''

response = query_llm(prompt)
```

## Using Ollama
Ollama is used for natural language understanding and generation tasks:

```python
command = ['ollama', 'run', 'llama3']
process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
output, _ = process.communicate(input=query)
process.wait()
```

## Running the Script
To run the script, ensure all dependencies are installed, and execute the script in your Python environment.

## Acknowledgements
Special thanks to all the open-source libraries and tools that made this project possible, including `pandas`, `scikit-learn`, `SentenceTransformer`, `Annoy`, and Ollama.

---