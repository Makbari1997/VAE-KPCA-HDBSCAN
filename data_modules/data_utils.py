import os
import numpy as np
from tensorflow.data import Dataset
from statistics import mean, median, mode


def __max_policy__(sentences:list) -> int:
    return max([len(sen.split()) for sen in sentences])

def __mean_policy__(sentences:list) -> int:
    return int(mean([len(sen.split()) for sen in sentences]))

def __mode_policy__(sentences:list) -> int:
    return int(mode([len(sen.split()) for sen in sentences]))

def __median_policy__(sentences:list) -> int:
    return int(median([len(sen.split()) for sen in sentences]))


def max_sentence_length(sentences:list, policy:str='max') -> int:
    if policy == 'max':
        return __max_policy__(sentences)
    elif policy == 'mode':
        return __mode_policy__(sentences)
    elif policy == 'mean':
        return __mean_policy__(sentences)
    elif policy == 'median':
        return __median_policy__(sentences)
    else:
        raise 'the \'policy\' parameter can only take one of these values: max, mean, median, mode'


#####################################################################################################

def __load_lbl_2_indx__(path):
  dic = {}
  with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
      _ = line.split()
      dic[_[0]] = int(_[1])
  return dic

def get_lbl_2_indx(path, intents=None):
  if os.path.isfile(path):
        return __load_lbl_2_indx__(path)
  else:
    lbl_2_indx = {}
    unique = list(set(intents))
    
    for i, j in enumerate(unique):
        lbl_2_indx[j] = i

    with open(path, 'w', encoding='utf-8') as f:
      for i in lbl_2_indx.items():
          f.write(str(i[0]) + ' ' + str(i[1]) + '\n')
    
    return lbl_2_indx

def one_hot_encoder(intents:list, lbl_2_indx:dict) -> np.ndarray:
    unique = list(set(intents))

    encoded = np.zeros((len(intents), len(lbl_2_indx.keys())))
    for i, j in enumerate(intents):
        encoded[i][lbl_2_indx[j]] = 1
    return encoded
  
    

#####################################################################################################

def preprocessing(tokenizer, text, max_length, return_tensors='tf', padding='max_length', truncation=True):
  '''
  makes text data ready for bert input\n
  Parameters:\n
  ------------
  tokenizer: transformers tokenizer object\n
  text: str or list of strings\n
  max_length: int\n
    defines maximum padding length\n
  Return: numpy.array\n
    input_ids, attention_mask, token_type_ids\n
  '''
  tokenized = tokenizer(text, return_tensors=return_tensors, padding=padding, max_length=max_length, truncation=True)
  input_ids = tokenized[tokenizer.model_input_names[0]]
  attention_mask = tokenized[tokenizer.model_input_names[2]]
  token_type_ids = tokenized[tokenizer.model_input_names[1]]
  return input_ids, attention_mask, token_type_ids 

def to_tf_format(x, y=None, buffer_size=None, batch_size=16):
  '''
  converts given data to batched tensorflow dataset\n
  Parameters:\n
  -------------
  x: tuple or iterable objects like list or numpy array\n
  y: iterable objects like list or numpy array\n
  buffer_size: int\n
  batch_size: int\n
  -------------
  Return: tf.Data.Dataset 
  '''
  data = None
  if y is None:
    data = x
  elif type(x) is tuple:
    data = x + (y,)
  else:
    data = (x, y)
  tf_dataset = Dataset.from_tensor_slices(data)
  tf_dataset = tf_dataset.shuffle(buffer_size=buffer_size).batch(batch_size)
  return tf_dataset