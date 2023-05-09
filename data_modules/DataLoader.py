class DataLoader:
  def __init__(self, path : str, encoding : str ='utf-8'):
    self.encoding = encoding
    self.path = path
    self.train_path = self.path + '/train/'
    self.dev_path = self.path + '/dev/'
    self.test_path = self.path + '/test/'
    self.ood_path = self.path + '/ood/'
  
  def train_loader(self) -> tuple:
    sentences = []
    intents = []
    with open(self.train_path+'sentences.txt', 'r', encoding=self.encoding) as f:
      lines = f.readlines()
      for line in lines:
        sentences.append(line.split('\n')[0])
    with open(self.train_path+'intents.txt', 'r', encoding=self.encoding) as f:
      lines = f.readlines()
      for line in lines:
        intents.append(line.split('\n')[0]) 
    return sentences, intents

  def dev_loader(self) -> tuple:
    sentences = []
    intents = []
    with open(self.dev_path+'sentences.txt', 'r', encoding=self.encoding) as f:
      lines = f.readlines()
      for line in lines:
        sentences.append(line.split('\n')[0])
    with open(self.dev_path+'intents.txt', 'r', encoding=self.encoding) as f:
      lines = f.readlines()
      for line in lines:
        intents.append(line.split('\n')[0]) 
    return sentences, intents

  def test_loader(self) -> tuple:
    sentences = []
    intents = []
    with open(self.test_path+'sentences.txt', 'r', encoding=self.encoding) as f:
      lines = f.readlines()
      for line in lines:
        sentences.append(line.split('\n')[0])
    with open(self.test_path+'intents.txt', 'r', encoding=self.encoding) as f:
      lines = f.readlines()
      for line in lines:
        intents.append(line.split('\n')[0])    
    return sentences, intents
  
  def ood_loader(self) -> tuple:
    sentences = []
    intents = []
    with open(self.ood_path+'sentences.txt', 'r', encoding=self.encoding) as f:
      lines = f.readlines()
      for line in lines:
        sentences.append(line.split('\n')[0])
    with open(self.ood_path+'intents.txt', 'r', encoding=self.encoding) as f:
      lines = f.readlines()
      for line in lines:
        intents.append(line.split('\n')[0])    
    return sentences, intents