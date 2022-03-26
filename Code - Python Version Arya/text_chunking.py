import sys
import pandas as pd
from typing import Optional
from time import time
from tqdm import tqdm
tqdm.pandas()

class ChunkText():
    """
    Reference paper: https://arxiv.org/abs/1910.10781
    @inproceedings{pappagari2019hierarchical, title={Hierarchical transformers for long document classification}, author={Pappagari, Raghavendra and Zelasko, Piotr and Villalba, Jes{\'u}s and Carmiel, Yishay and Dehak, Najim}, booktitle={2019 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)},pages={838--844}, year={2019}, organization={IEEE}}
    :param dataframe: dataframe must be of type 'pd.DataFrame'
    :param text_column_name: Name of the column having text/clinical text in the dataframe, by default it is considered to be 'text'
    :param device: Name of the column having corresponding labels of text in the dataframe, by default it is considered to be 'label'
    :param split_size: Size of the text split
    :param overlap_size: Number of words/token to overlap with each chunks/splits of a text
    """
    
    def __init__(self, dataframe:pd.DataFrame, 
                 text_column_name: Optional[str] = 'text',
                 label_column_name:Optional[str]='label',
                 split_size : Optional[int] = 200,
                 overlap_size : Optional[int] = 50): 
        try:
            if split_size <= overlap_size:
                raise ValueError("Incorrect inputs: split_size cannot be <= overlap_size")
            else:
                self.split_size = split_size
                self.overlap_size = overlap_size
                self.dataframe = dataframe
                self.text_column_name = text_column_name
                self.label_column_name = label_column_name
        except ValueError as Error:
            print(Error)

    def split_text(self, input_text:str):
        total_text = []
        partial_text = []
        full_text_length = int(self.split_size - self.overlap_size)
        if len(input_text.split())//full_text_length > 0:
            n = len(input_text.split())//full_text_length
        else: 
            n = 1
        for w in range(n):
            if w == 0:
                partial_text = input_text.split()[:self.split_size]
                total_text.append(" ".join(partial_text))
            else:
                
                partial_text = input_text.split()[w*full_text_length:w*full_text_length + self.split_size]
                total_text.append(" ".join(partial_text))
        return total_text

    def get_chunked(self):
        print("Old dataframe shape before chunking:", self.dataframe.shape)
        self.dataframe['text_split'] = self.dataframe[self.text_column_name].progress_apply(self.split_text)
        text_list = list()
        corresponding_label = list()
        index_list = list()
        
        for idx,row in self.dataframe.iterrows():
            for text in row['text_split']:
                text_list.append(text)
                corresponding_label.append(row[self.label_column_name])
                index_list.append(idx)
        dataframe_chunked = pd.DataFrame({'text':text_list, 'label':corresponding_label})
        print("New dataframe shape after chunking:", dataframe_chunked.shape)
        return dataframe_chunked

# df = pd.read_csv("../dataset.csv")
# ob = ChunkText(dataframe=df)
# df_temp = ob.get_chunked()
# print(df_temp.shape)