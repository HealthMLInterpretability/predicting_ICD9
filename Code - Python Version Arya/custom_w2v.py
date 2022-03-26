import pandas as pd
import numpy as np
from gensim import models
import time
import multiprocessing
cores = multiprocessing.cpu_count()
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from gensim.models import Word2Vec, KeyedVectors  
import warnings
from typing import Optional
warnings.simplefilter(action='ignore', category=FutureWarning)


def TSNE_PLOT(model, word:str, embedding_dim:Optional[int] = 50):
    
    arr = np.empty((0,embedding_dim), dtype='f')
    word_labels = [word]

    # get close words
    close_words = model.wv.similar_by_word(word,topn=20)
    
    # add the vector for each of the closest words to the array
    arr = np.append(arr, np.array([model.wv[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model.wv[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)
        

    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]

    plt.figure(figsize =(10,8))
    plt.scatter(x_coords, y_coords)

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.show()


def build_w2v_model(text_list : list, model_name:str, model_type: Optional[int] = 1, window_size: Optional[int] = 150, embedding_dim: Optional[int] = 50):
    start_time = time.time()
    model = models.Word2Vec(sentences= text_list,
                    window = window_size, #The maximum distance between a target word and words around the target word. 
                    sg=model_type, # The training algorithm, either CBOW (0) or skip gram (1). ## initially used sg=1 # https://stackoverflow.com/questions/39224236/word2vec-cbow-skip-gram-performance-wrt-training-dataset-size
                    vector_size=embedding_dim,  #The number of dimensions of the embedding, 300, 100, 50
                    workers=cores, #The number of threads to use while training.
                    sample=6e-5,
                    min_count=1) 
    print(f'Vocabulary size: {len(list(model.wv.index_to_key))}')
    print(f'Time taken : {(time.time() - start_time) / 60:.2f} mins')
    model.wv.save_word2vec_format(f"{model_name}.bin", binary=True)
    model.wv.save_word2vec_format(f"{model_name}.txt")
    return model


def text2vec(text, model):
    return np.mean([model.wv[x] for x in text.split() if x in model.wv.key_to_index], axis=0, dtype=np.float64).reshape(1,-1)




def build_numerical_dataset(embedding_dim, df_kms:pd.DataFrame, num_classes:int, isCategorical= False):
  columns = np.arange(0,embedding_dim)
  df = df_kms.copy()
  for x in range(0,df.shape[0]):
    df.vectors[x] = df.vectors[x].tolist()[0]
  print(f"Input dataframe shape: {df.shape}" )
  df_split = pd.DataFrame(df["vectors"].to_list(), columns=columns)
  df_split.iloc[:1, :]
  df_split['label'] = df_kms.label
  print(f"Output dataframe shape: {df_split.shape}")
  if isCategorical == True:
    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    df_split['label'] = labelencoder.fit_transform(df_split['label'])
    return df_split
  return df_split


def build_dataset(dataframe: pd.DataFrame, model, dataset_name: str,  column_name:str, features:int, num_classes = 2, isCategorical = False):
  text = column_name
  df_experimental = dataframe.copy()
  df_experimental['vectors'] = df_experimental.text.apply(text2vec, model = model)
  df = build_numerical_dataset(features, df_experimental, num_classes = num_classes, isCategorical = isCategorical)
  df.to_csv(f"{dataset_name}.csv", index=False)
  # return df
  
  
class CustomW2VGen():
    """_summary_
    """
    def __init__(self, dataset: pd.DataFrame, w2v_dataset_name :str, similar_word:str, text_column: Optional[str] = "text",  label_column : Optional[str] = "label", embedding_dim: Optional[int] = 50, num_classes: Optional[int] = 2, model_type: Optional[int] = 1, isCategorical:Optional[bool] = False):
        self.dataset = dataset
        self.label_column = label_column
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.model_type = model_type
        self.similar_word = similar_word
        self.custom_dataset_name = w2v_dataset_name
        text_list = list(self.dataset[text_column].apply(str.split).values)
        self.model = build_w2v_model(text_list = text_list, model_type=self.model_type)
        TSNE_PLOT(model, word = self.similar_word)
        build_dataset(dataframe = self.dataset, model = self.model, 
                      dataset_name = self.custom_dataset_name, column_Name = text_column, 
                      features = self.embedding_dim , num_classes = self.num_classes, isCategorical = False)

        
        