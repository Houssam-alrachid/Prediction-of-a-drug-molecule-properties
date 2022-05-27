import os
import time
import joblib
import numpy as np
import pandas as pd
import servier.src.config as config
import servier.src.feature_extractor as feature_extractor

from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from keras.models import load_model, Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Dropout, BatchNormalization, Flatten
from keras.layers import LSTM, Conv1D, GRU, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import callbacks
from tensorflow.keras.optimizers import Adam 
import pickle


def read_data(data_path=config.path_single, col_smiles='smiles', col_target=config.COL_TARGET_SINGLE):
    """
    data_path: str, path to the a CSV data file
    col_smiles: str, name of smiles column
    col_target: str, name of target column(s)
    """
    # read data
    df = pd.read_csv(data_path)
    X = df[col_smiles]
    y = df[col_target]#.values
    
    return X, y, df

def df_extract_ft(data_path=config.path_single):
    """
    Extract features
    """
    
    # read data
    df = pd.read_csv(data_path)
    
    df['smiles_features'] = df['smiles'].apply(lambda x: np.array(feature_extractor.fingerprint_features(x)))
    df_feat = pd.DataFrame()
    
    for i in range(len(df['smiles_features'])):
        df_feat['ft_'+str(i)] = df['smiles_features'][i]
    
    df_feat = df_feat.T.reset_index(drop=True)
    # [ft_0, ft_2, ... ft_2047]
    df_feat.columns = ['ft_'+str(i) for i in range(len(df['smiles_features'][0]))]
    
    df_all = pd.concat([df, df_feat], axis=1)
    X_ft = df_all.drop(columns=['mol_id', 'smiles_features', 'smiles'])
    
    X = X_ft.drop(columns=['P1'], axis=1)
    y = X_ft['P1']
    return X,y

def generate_tokens(smiles, len_percentile=100):
    """
    Create tokens for sequences
    """
    # Get max length of smiles
    smiles_len = smiles.apply(lambda p: len(p))
    max_phrase_len = int(np.percentile(smiles_len, len_percentile))
    # print('True max length is ' + str(np.max(smiles_len)) + ', ' + str(max_phrase_len) + ' is set the length cutoff.')
        
    # Get unique words
    #['#', '(', ')', '+', '-', '/', '1', '2', '3', '4', '5', '6', '=', 'B', 'C', 'F', 'H', 'N', 'O', 'S', '[', '\\', ']', 'c', 'l', 'n', 'o', 'r', 's']
    unique_words = np.unique(np.concatenate(smiles.apply(lambda p: np.array(list(p))).values, axis=0))
    num_words = len(unique_words)
    # print('Vocab size is ' + str(num_words))
    
    tokenizer = Tokenizer(
        num_words = num_words,
        filters = '$',
        char_level = True,
        oov_token = '_'
    )
    
    tokenizer.fit_on_texts(smiles)
    sequences = tokenizer.texts_to_sequences(smiles)
    tokens = pad_sequences(sequences, maxlen = max_phrase_len, padding='post', truncating='post')
    
    return tokens, num_words, max_phrase_len 

def save_model(model_type):
    models_folder = 'servier\models'
    os.makedirs(models_folder, exist_ok=True)
    classifier = os.path.join(models_folder, model_type + '.pkl')
    pickle_out = open(classifier,"wb")
    pickle.dump(classifier, pickle_out)
    pickle_out.close()
    return classifier

def get_prediction_score(y_label, y_predict):
    """Evaluate predictions using different evaluation metrics.
    y_label: list, contains true label
    y_predict: list, contains predicted label
    :return scores: dict, evaluation metrics on the prediction
    """
    scores = {}
    scores['accuracy'] = accuracy_score(y_label, y_predict)
    scores['f1-score'] = f1_score(y_label, y_predict, labels=None, average='macro', sample_weight=None)
    scores['Confusion Matrix'] = confusion_matrix(y_label, y_predict)
    
    return scores

def create_model(model_type, num_words, input_length, output_dim=1, dropout_rate=0.0):
    """
    Choose one of several sequential DL models
    """
    # Initialising the DL model
    model = Sequential()
    if model_type == 'FF': # Feed Forward Neural Network
        model.add(Dense(units =16 , kernel_initializer = 'uniform', activation = 'relu', input_dim = input_length))
        model.add(Dense(units = 24, kernel_initializer = 'uniform', activation = 'relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))
        model.add(Dense(units = 24, kernel_initializer = 'uniform', activation = 'relu'))
        model.add(Dense(units = output_dim, kernel_initializer = 'uniform', activation = 'sigmoid')) 
    elif model_type == 'lstm': # LSTM
        model.add(Embedding(num_words+1, 50, input_length=input_length))
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(Bidirectional(LSTM(128)))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(output_dim, activation='sigmoid'))
    elif model_type == 'gru': # GRU
        model.add(Embedding(num_words+1, 50, input_length=input_length))
        model.add(Bidirectional(GRU(128, return_sequences=True)))
        model.add(Bidirectional(GRU(128)))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(output_dim, activation='sigmoid'))
    elif model_type == 'cnn-gru': # 1D CNN - GRU
        model.add(Embedding(num_words+1, 50, input_length=input_length))
        model.add(Conv1D(192,3,activation='relu'))
        model.add(Bidirectional(GRU(224, return_sequences=True)))
        model.add(Bidirectional(GRU(384)))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(output_dim, activation='sigmoid'))
    elif model_type == 'cnn': # 1D CNN
        model.add(Embedding(num_words+1, 50, input_length=input_length))
        model.add(Conv1D(192, 10, activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv1D(192, 3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(output_dim, activation='sigmoid'))
    else:
        raise ValueError(model_type + ' is not supported.')
 
    model.summary()    
    return model

def Train(model_type = config.MODEL_TYPE):
    """
    Train a Deep Learning model to perform prediction of basic preperties of a molecule from its fingerprint features
    Input: path to csv file of the dataset
    """
    if model_type == 'FF':
        # Load and split  dataset
        X, y = df_extract_ft(config.path_single)
        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
        
        print(' Train a one dimensional classification model')
        model = create_model(model_type=model_type, num_words = 0, input_length=X_train.shape[1], output_dim=config.output_dim, dropout_rate=config.dropout_rate)    
    
    else:
        # Load and split  dataset
        smiles, y, df = read_data(config.path_single, col_smiles='smiles', col_target='P1')
        tokens, num_words, max_phrase_len = generate_tokens(smiles, len_percentile=100)
        X_train, X_test, y_train, y_test = train_test_split(tokens, y, test_size=0.2, shuffle=True, stratify=y, random_state=0)
        
        print(' Train a one dimensional classification model')
        model = create_model(model_type=model_type, num_words = num_words, input_length=max_phrase_len, output_dim=config.output_dim, dropout_rate=config.dropout_rate)
                
    # Callback list
    callback_list = []
    # monitor val_loss and terminate training if no improvement
    early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=20, verbose=2, mode='auto', restore_best_weights=True)
    callback_list.append(early_stop)

    # Compute class weights
    weight_list = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    weight_dict = {}
    for i in range(len(np.unique(y_train))):
        weight_dict[np.unique(y_train)[i]] = weight_list[i]
    
    # Train only classification head
    optimizer = Adam(lr=config.lr, decay=1e-6)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    models_folder = 'servier/models'
    os.makedirs(models_folder, exist_ok=True)
    
    checkpointer = ModelCheckpoint(os.path.join(models_folder, model_type + '.h5'), monitor='val_acc', verbose=1, save_weights_only=False, save_best_only=False, mode='auto') 
    callback_list.append(checkpointer)
          
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=config.nb_epochs, class_weight=weight_dict, callbacks=callback_list, verbose=2)
    
    return X_test, y_test, model, history

def Predict(X_test, model_type = config.MODEL_TYPE):
    """
    Predict from your pretrained model
    """
    models_folder = 'servier/models'
    path = os.path.join(models_folder, model_type + '.h5')
    # Load pretrained model
    model = load_model(path)
    
    if isinstance(X_test, str):
        if model_type == 'FF':
            X_test, y_test = df_extract_ft(X_test)
        else:
            smiles, y, df = read_data(X_test, col_smiles='smiles', col_target='P1')
            X_test, num_words, max_phrase_len = generate_tokens(smiles, len_percentile=100)

    prediction = model.predict(X_test)
    y_pred = (prediction > 0.5).astype('uint8')
    return y_pred

def Evaluate(y_test, y_pred):
    '''
    Evaluate the prediction model by measuring accuracy
    - Accuracy
    - F1-score
    - Confusion matrix
    '''
    if isinstance(y_test, str): # y_test is a csv file containg input-output
        X_test, y_test, df = read_data(y_test, col_smiles='smiles', col_target=config.COL_TARGET_SINGLE)
        
    scores = get_prediction_score(y_test, y_pred)    
    # Summarize model performance
    model_eval = pd.DataFrame({'accuracy': [scores['accuracy']],
                            'f1-score': [scores['f1-score']],
                            'Confusion Matrix': [scores['Confusion Matrix']]})
    model_eval = model_eval[['accuracy', 'f1-score', 'Confusion Matrix']]
    return model_eval


def main():

    print(80*'*')
    print('This application uses a Deep Learning to predict basic properties of a molecule from its fingerprint features and molecules strings')
    print(80*'*')
    print(' Start processing ...')

    # Train machine learning model (and save pretrained model to disk)
    X_test, y_test, model, history = Train(model_type = config.MODEL_TYPE)

    # prediction
    y_pred = main.Predict(X_test, model_type = config.MODEL_TYPE)

    # evaluation
    Evaluate(y_test, y_pred)

    print(' DONE...')


if __name__ == "__main__":
    main()
