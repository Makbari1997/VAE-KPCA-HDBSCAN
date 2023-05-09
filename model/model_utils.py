from transformers import AutoTokenizer, TFBertModel


def get_bert(name:str='bert-base-uncased'):
  bert = TFBertModel.from_pretrained(name)
  tokenizer = AutoTokenizer.from_pretrained(name)
  return bert, tokenizer

