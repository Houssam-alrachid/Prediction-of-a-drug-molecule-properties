# Prediction-of-a-drug-molecule-properties

## I - Introduction

The prediction of a drug molecule properties plays an important role in the drug design process. The molecule properties are the cause of failure for 60% of all drugs in the clinical phases. A multi parameters optimization using machine learning methods can be used to choose an optimized molecule to be subjected to more extensive studies and to avoid any clinical phase failure.

The objective of this project is to develop multiple deep learning models to predict one or more basic properties of a molecule using its fingerprint features.
In addiction:

- A Flask Api to make predictions;
- The application, called **servier**, is installed;
- Then packaged in a Docker to facilitate the deployment of your model.

### Project Structure

This project is organized as follow

```
.
├── molecule-properties-prediction.ipynb
├── Dockerfile
├── requirements.txt
├── README.md
├── servier
│   ├── cli.py
│   ├── data
│   │   ├── dataset_multi.csv
│   │   ├── dataset_single.csv
│   │   └── dataset_single_test.csv
│   ├── __init_.py
│   ├── models
│   │   ├── FF.h5
│   │   └── model_pred_single.bin
│   │   └── cnn.h5
│   │   └── lstm.h5
│   │   └── gru.h5
│   │   └── lstm_multi.h5
│   └── src
│       ├── config.py
│       ├── feature_extractor.py
│       ├── flask_api.py
│       └── main.py
└── setup.py
```


## II - Exploratory Data Analysis

Let's start creating statistics and visualizations on my dataset to identify good columns for modeling, potential data quality issues and anticipate potential feature transformations necessary...

First import data from `servier\data\dataset_single.csv`.

```python
P1	mol_id	smiles
0	1	CID2999678	Cc1cccc(N2CCN(C(=O)C34CC5CC(CC(C5)C3)C4)CC2)c1C
1	0	CID2999679	Cn1ccnc1SCC(=O)Nc1ccc(Oc2ccccc2)cc1
2	1	CID2999672	COc1cc2c(cc1NC(=O)CN1C(=O)NC3(CCc4ccccc43)C1=O...
3	0	CID5390002	O=C1/C(=C/NC2CCS(=O)(=O)C2)c2ccccc2C(=O)N1c1cc...
4	1	CID2999670	NC(=O)NC(Cc1ccccc1)C(=O)O
```

`dataset_single.csv` contains `4999` rows and `3` columns and no missing data nor duplicates:

- smiles : a string representation for a molecule
- mol_id : Unique id for the molecule
- P1 : binary property to predict

We remark that `17.8%` only of the entries are considered of property `P1 = 0` and `82.2%` of property `P1 = 1`. That's mean the our dataset is unbalanced, which may have negative consequences later!!!

## III. Modeling and evaluation

We recall the goal of this project to develop a best suited algorithm to predict the class of one or more molecule properties. We will apply 3 types of Deep Learning models:

1. Type 1: Feed Frward Neural Network
2. Type 2: Sequential Deep Learning models bases on `smile` sequences as input : CNN, LSTM, GRU
3. Type 3: An extension of Type 1 to predict the `P1, P2,...,P9` properties of the `servier\data\dataset_multi.csv` dataset.

### III.1 Type 1: Feed Frward Neural Network
This model takes the extracted features of a molecule as input and predict the `P1` property. We use the function `fingerprint_features` of the
`feature_extractor.py` module to extract the features from a molecule smile. The output of this module is a vector of size `2048` that we will use as an input for our models. We concatenate this vector to the dataset for each entry (molecule).

```python
Train a one dimensional classification model
Model: "sequential"
model.add(Dense(units =16 , kernel_initializer = 'uniform', activation = 'relu', input_dim = input_length))
model.add(Dense(units = 24, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 24, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))     
```
Evaluation of the model:
```python
accuracy	f1-score	Confusion Matrix
0.729	0.589463	[[73, 103], [168, 656]]
```

### III.1 Type 2: Sequential Deep Learning models bases on 'smile' sequences as input : CNN, LSTM, GRU 
These models take the *smile string* character as input and predict the *P1* property.In this work, a Deep Learning model that automatically learns features from `smiles` to predict chemical properties, without the need for additional explicit feature engineering.

#### III.1.1 CNN model

```python
Model: "sequential"
model.add(Embedding(num_words+1, 50, input_length=input_length))
model.add(Conv1D(192, 10, activation='relu'))
model.add(BatchNormalization())
model.add(Conv1D(192, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(1, activation='sigmoid'))

```
Evaluation of the model:

```python
accuracy	f1-score	Confusion Matrix
0.75	0.559111	[[46, 132], [118, 704]]
```


#### III.1.2 LSTM model

```python
Model: "sequential"
model.add(Embedding(num_words+1, 50, input_length=input_length))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Bidirectional(LSTM(128)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(output_dim, activation='sigmoid'))

```
Evaluation of the model:

```python
accuracy	f1-score	Confusion Matrix
0.719	0.610949	[[96, 82], [199, 623]]
```

#### III.1.3 GRU model

```python
Model: "sequential"
model.add(Embedding(num_words+1, 50, input_length=input_length))
model.add(Bidirectional(GRU(128, return_sequences=True)))
model.add(Bidirectional(GRU(128)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(1, activation='sigmoid'))

```
Evaluation of the model:

```python
accuracy	f1-score	Confusion Matrix
0.762	0.587879	[[56, 122], [116, 706]]
```

### III.1 Type 3: Feed Forward Neural Network Multi-class classification
This model is an extension of Type 1 to predict the `P1, P2,...,P9` properties of the `servier\data\dataset_multi.csv` dataset.
```python
Model: "sequential"
model.add(Dense(units =16 , kernel_initializer = 'uniform', activation = 'relu', input_dim = input_length))
model.add(Dense(units = 24, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 24, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'sigmoid'))     
```
Evaluation of the model:

```python
accuracy	f1-score	
0.281	0.909	
```
The model is not very performant and it needs optimisation and more data to support high dimension requirements.

## IV. Setup the development environment

The `feature_extractor.py` module use `rdkit` library. This library is available on conda. Run the following command to install the package and
prepare environment:
```python
wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda
export PATH=~/miniconda/bin:$PATH
conda update -n base conda
conda create -y --name servier python=3.6
conda activate servier
pip install -r requirements.txt
conda install -c conda-forge rdkit
```

## V. Flask api
To distribute and use the App is by packaging it as an API, I use a `Flask` API framework used to send your molecule smile and get the prediction.

A python file for the api `servier/src/flask_api.py` to configure the Api with just one route `/predict`:
```python
@app.route('/predict_from_file/<path:path_X_test>')
def predict_from_file(path_X_test):
    """Let's predict basic molecule properties.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
    """
    y_pred = Predict(path_X_test, model_type = 'FF')
    return {'Hello the answer is y_pred': y_pred.tolist()}
    
    if __name__=='__main__':
    app.run(host='0.0.0.0',port=8000)
    ```
    Modify you model in the file, then run the API from the terminal:
    
 ```python
 python servier/src/flask_api.py
 ```
Copy the link from terminal (http://x.x.x.x:8000/); Choose you test data and open the browser and type:
 ```python
http://x.x.x.x:8000/predict_from_file/servier/data/dataset_single_test.csv
 ```

## VI. Packaging
The app is installable using a `setup.py`  and `cli.py`.

1. `setup.py` configures the application:
 ```python
setup(
    name="predict_molecule_property",
    version="1.0.0",
    author='Houssam ALRACHID',
    author_email='houssam.alrachid@devoteam.com',
    description='The objective of this application uses a deep learning model to predict basic properties of a molecule using its fingerprint features and smiles strings.',
    packages=["servier"],
    entry_points='''
        [console_scripts]
        predict_molecule_property = servier.cli:main
        ''')
 ```
2. `cli.py` contains four commands: train, predict, evaluate:
 ```python
@main.command()
@click.option('--model_type', '-d', type=str, required=True, default='FF', help="Please enter the name of the model you want to train")
def train(model_type):
    """Train a Deep learning model for prediction and save the pretrained model to disk"""
    click.echo(Train(model_type))
@main.command()
@click.option('--path_x_test', '-p', type=str, required=True, default='servier/data/dataset_single.csv', help="Please enter the path of data in order to perform prediction")
@click.option('--model_type', '-p', type=str, required=True, default='FF', help="Please enter the model name")

def predict(path_x_test, model_type):
    """Perform prediction using a pretrained Deep Learning prediction model"""
    click.echo(Predict(path_x_test, model_type))
@main.command()
@click.option('--y_test', '-t', type=float, required=True, help='Enter the true array')
@click.option('--y_pred', '-p', type=float, required=True, help='Enter the predicted array')
def evaluate(y_test, y_pred):
    """Evaluate the prediction model"""
    click.echo(Evaluate(y_test, y_pred))
 ```
 3. Install the `predict_molecule_property` using the command from the root of the project:
```python
pip install -e .
```
4. Run the app:
```python
predict_molecule_property --help
```
To train model:
```python
predict_molecule_property train --model_type lstm 
```
To predict:
```python
predict_molecule_property predict --path_x_test servier/data/dataset_single.csv --model_type lstm 
```
To evaluate model:
```python
predict_molecule_property evaluate --y_test test-value --y_pred real-value
```

## VII. Dockerization
Another method for packaging the App and ensure scalability, we use `Docker` by creating a `Dockerfile`.
```python
# Mention the base image 
FROM continuumio/anaconda3:4.4.0
LABEL Author, Houssam ALRACHID
ENV APP_HOME /servier
# Copy the current folder structure and content to docker folder
COPY . $APP_HOME
# Expose the port within docker 
EXPOSE 5000
# Set current working directory
WORKDIR $APP_HOME
# Install the required libraries
RUN conda install -c conda-forge rdkit
# Container start up command
CMD python flask_api.py
```
To build a docker image, use the command from the root of the project:
```python
docker build . -t servier
```
To run docker:
```python
docker run -p 8000:8000 money_api
```
Browser run : use the link from terminal or my machine IP:
```python
http://x.x.x.x:8000/apidocs
```
