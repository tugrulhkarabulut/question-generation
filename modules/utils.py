import ntpath
from pathlib import Path
import pickle

def get_file_name(path):
    return ntpath.basename(path).split('.')[0]

def get_file_format(path):
    return ntpath.basename(path).split('.')[-1]

def build_file_path(path, file_name, format_):
    file_path = Path(path)
    file_path = file_path / (file_name + '.' + format_)
    return file_path

def load_tokenizers(path):
    with open(path, 'rb') as f:
        sq_tokenizer = pickle.load(f)
    
    return sq_tokenizer.tokenizers

def padded_input_to_sentence(padded_input, word_lang):
  sentence = ''
  for i in padded_input:
    if i == 0:
      break

    sentence += word_lang.index_word[i] + ' '
  
  return sentence.strip()