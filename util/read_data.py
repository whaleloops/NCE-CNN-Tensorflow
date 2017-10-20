# NON-debugged code

from Vocab import Vocab
import numpy as np

def read_embedding(vocab_path, emb_path):
  vocab = Vocab(vocab_path)
  embedding = np.load(emb_path)
  return vocab, embedding

def read_sentences(path, vocab):
  sentences = {}
  file = open(path, "r")
  while True:
    line = file.readline().rstrip('\n')
    if line == "":
      break
    tokens = line.split()  # TODO0: check if this splits str well?
    length = len(tokens)
    sent = np.zeros(max(length,3))
    counter = 0
    for i in range(0,length):
      token = tokens[i]
      sent[i] = vocab.index(token)
    if length < 3:
      for i in range(length, 3):
        sent[i] = vocab.index('unk') # sent[len]
    if sent.sum() == 0:
      print('line: '+line)
    sentences[len(sentences)] = sent
  file.close()
  return sentences

def read_relatedness_dataset(direc, vocab, task):
  dataset = {}
  dataset['vocab'] = vocab
  if task == 'twitter':
    file1 = 'tokenize_query2.txt'
    file2 = 'tokenize_doc2.txt'
  else:
    file1 = 'a.toks'
    file2 = 'b.toks'
  dataset['lsents'] = read_sentences(direc + file1, vocab)
  dataset['rsents'] = read_sentences(direc + file2, vocab)
  dataset['size'] = len(dataset['lsents'])
  id_file = open(direc + 'id.txt', 'r')
  sim_file = open(direc + 'sim.txt')
  dataset['ids']= {}
  dataset['labels'] = np.zeros(dataset['size'])
  if task == 'twitter' or task == 'qa':
    boundary_file = open(direc + 'boundary.txt')
    numrels_file = open(direc + 'numrels.txt')
    boundary = {}
    counter = 0
    while True:
      line = boundary_file.readline().rstrip('\n')
      if line == "":
        break
      boundary[counter] = float(line)
      counter = counter + 1
    boundary_file.close()  
    dataset['boundary'] = np.zeros(len(boundary))
    for counter, bound in boundary.iteritems():
      dataset['boundary'][counter] = bound  
    # read numrels data
    dataset['numrels'] = np.zeros(len(boundary)-1) #torch.IntTensor(#boundary-1)
    for i in range(0, len(boundary)-1):
      tmp = numrels_file.readline().rstrip('\n')
      dataset['numrels'][i] = int(tmp)
    numrels_file.close()

  for i in range(0, dataset['size']):
    dataset['ids'][i] = id_file.readline().rstrip('\n')
    if task == 'sic':
      tmp = float(sim_file.readline().rstrip('\n'))
      dataset['labels'][i] = 0.25 * (tmp - 1) # sic data
    elif task == 'vid':
      tmp = float(sim_file.readline().rstrip('\n'))
      dataset['labels'][i] = 0.2 * (tmp) # vid data
    else:
      tmp = float(sim_file.readline().rstrip('\n'))
      dataset['labels'][i] = tmp # twi and msp
  id_file.close()
  sim_file.close()
  return dataset