# AI Cup 2019 Abstract-Labeling-and-Classification 方法 & code 整理
|2019/11 edited by Andrew
## Data 介紹
Train set:
    ID: 序列編號 (EX: D00001)
    Title: 文章標題 (英文)
    Abstract: 摘要內容 (英文，句子與句子間用"$$$"分隔)
    Authors:  作者 (英文，作者間以"/"分隔)
    Categories: 論文分類 (Ex: cs.CR/cs.CV) 
    Created Date: 上傳日期 (YYYY/MM/DD)
    Task1: 共6種(Background/Objective/Methods/Results/Conclutions/Others 可重複)
    
## 目標
判別Abstract中每句話分別屬於Task1中的哪個類別
Ex: A Brain-Inspired Trust Management Model to Assure Security in a Cloud based IoT Framework for Neuroscience Applications. --> Background
以one-hot方式輸出 Ex: Background/Objective 輸出 1 1 0 0 0 0

## 使用函式及環境
pandas
nltk
numpy
sklearn
pickle
tqdm
json
## 資料處理
1. 刪除多餘資料並random拆分validation set: 現階段僅使用abstract內容進行訓練
```python
dataset = pd.read_csv(dataPath+'task1_trainset.csv', dtype=str)
dataset.drop('Title',axis=1,inplace=True)
dataset.drop('Categories',axis=1,inplace=True)
dataset.drop('Created Date',axis=1, inplace=True)
dataset.drop('Authors',axis=1,inplace=True)
	
trainset, validset = train_test_split(dataset, test_size=0.1, random_state=42)
	
trainset.to_csv(dataPath+'trainset.csv',index=False)
validset.to_csv(dataPath+'validset.csv',index=False)
	
dataset = pd.read_csv(dataPath+'task1_public_testset.csv', dtype=str)
dataset.drop('Title',axis=1,inplace=True)
dataset.drop('Categories',axis=1,inplace=True)
dataset.drop('Created Date',axis=1, inplace=True)
dataset.drop('Authors',axis=1,inplace=True)
dataset.to_csv('testset.csv',index=False)
print('[INFO] Dataset splited!')
```
2. 製作dictionary
```python
def collect_words(data_path):
  df = pd.read_csv(data_path, dtype=str)

  tokens = set()
  for i in df.iterrows():
    sents  = i[1]['Abstract'].split('$$$')
    sents = ' '.join(sents)
    tokens |= set(word_tokenize(sents))
  return tokens

#Dictionary
words = set()
words |= collect_words(dataPath+'trainset.csv')

PAD_TOKEN = 0
UNK_TOKEN = 1

word_dict = {'<pad>':PAD_TOKEN,'<unk>':UNK_TOKEN}

for word in words:
  word_dict[word]=len(word_dict)

with open(dataPath+'dicitonary.pkl','wb') as f:
  pickle.dump(word_dict, f)

with open(dataPath+'dicitonary.pkl','rb') as f:
  word_dict = pickle.load(f)

print('[INFO] Dicitonary built!')
```
3. Data formatting
```python
def get_dataset(data_path, word_dict, n_workers=4):
  """ Load data and return dataset for training and validating.

  Args:
      data_path (str): Path to the data.
  Return:
      output (list of dict): [dict, dict, dict ...]
  """
  dataset = pd.read_csv(data_path, dtype=str)
  formatData = []
  for (idx,data) in dataset.iterrows():
    """
    processed: {
      'Abstract': [[4,5,6],[3,4,2],...]
      'Label': [[0,0,0,1,1,0],[1,0,0,0,1,0],...]
    }
    """
    processed = {}
    processed['Abstract'] = [sentence_to_indices(sent, word_dict) for sent in data['Abstract'].split('$$$')]
    if 'Task 1' in data:
      processed['Label'] = [label_to_onehot(label) for label in data['Task 1'].split(' ')]
    formatData.append(processed)
  
  return formatData
```
```python
def sentence_to_indices(sentence, word_dict):
  """ Convert sentence to its word indices.
  Args:
      sentence (str): One string.
  Return:
      indices (list of int): List of word indices.
  """
  return [word_dict.get(word,UNK_TOKEN) for word in word_tokenize(sentence)]
  
def label_to_onehot(labels):
  """ Convert label to onehot .
      Args:
          labels (string): sentence's labels.
      Return:
          outputs (onehot list): sentence's onehot label.
  """
  label_dict = {'BACKGROUND': 0, 'OBJECTIVES':1, 'METHODS':2, 'RESULTS':3, 'CONCLUSIONS':4, 'OTHERS':5}
  onehot = [0,0,0,0,0,0]
  for l in labels.split('/'):
    onehot[label_dict[l]] = 1
  return onehot

def preprocess_sample(data, word_dict):
    """
    Args:
        data (dict)
    Returns:
        dict
    """
    processed = {}
    processed['Abstract'] = [sentence_to_indices(sent, word_dict) for sent in data['Abstract'].split('$$$')]
    if 'Task 1' in data:
        processed['Label'] = [label_to_onehot(label) for label in data['Task 1'].split(' ')]
        
    return processed
```
```python=
print('[INFO] Start processing trainset...')
train = get_dataset('trainset.csv', word_dict, n_workers=4)
print('[INFO] Start processing validset...')
valid = get_dataset('validset.csv', word_dict, n_workers=4)
print('[INFO] Start processing testset...')
test = get_dataset('testset.csv', word_dict, n_workers=4)
```
4. Data packing
```python=
class AbstractDataset(Dataset):
  def __init__(self, data, pad_idx, max_len = 500):
    self.data = data
    self.pad_idx = pad_idx
    self.max_len = max_len
  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    return self.data[index]
      
  def collate_fn(self, datas):
    """
    returns:
    Tensor(batch,sentence,words) : input data
    Tensor(batch,sentence,words) : corresponding answer
    list(sentence quantity in each abstract): use in prediction, to remove the redundant sentences (the sentences we padded)
    
    """
    # get max length in this batch
    max_sent = max([len(data['Abstract']) for data in datas])
    max_len = max([min(len(sentence), self.max_len) for data in datas for sentence in data['Abstract']])
    batch_abstract = []
    batch_label = []
    sent_len = []
    for data in datas:
      # padding abstract to make them in same length
      pad_abstract = []
      for sentence in data['Abstract']:
        if len(sentence) > max_len:
          pad_abstract.append(sentence[:max_len])
        else:
          pad_abstract.append(sentence+[self.pad_idx]*(max_len-len(sentence)))
      sent_len.append(len(pad_abstract))
      pad_abstract.extend([[self.pad_idx]*max_len]*(max_sent-len(pad_abstract)))
      batch_abstract.append(pad_abstract)

      # gather labels
      if 'Label' in data:
          pad_label = data['Label']
          pad_label.extend([[0]*6]*(max_sent-len(pad_label)))
          batch_label.append(pad_label)

    return torch.LongTensor(batch_abstract), torch.FloatTensor(batch_label), sent_len

```
5. F1 score
```python=
class F1():
	def __init__(self):
		self.threshold = 0.5
		self.n_precision = 0
		self.n_recall = 0
		self.n_corrects = 0
		self.name = 'F1'

	def reset(self):
		self.n_precision = 0
		self.n_recall = 0
		self.n_corrects = 0

	def update(self, predicts, groundTruth):
		predicts = (predicts > self.threshold).float()
		self.n_precision += torch.sum(predicts).data.item()
		self.n_recall += torch.sum(groundTruth).data.item()
		self.n_corrects += torch.sum(groundTruth * predicts).data.item()

	def get_score(self):
		recall = self.n_corrects / self.n_recall
		precision = self.n_corrects / (self.n_precision + 1e-20) #prevent divided by zero
		return 2 * (recall * precision) / (recall + precision + 1e-20)
	
	def print_score(self):
		score = self.get_score()
		return '{:.5f}'.format(score)
```
6. Model
    - GRU
    - LSTM
    - +Attention
7. Result?
![](https://i.imgur.com/iGcEiPV.jpg)
