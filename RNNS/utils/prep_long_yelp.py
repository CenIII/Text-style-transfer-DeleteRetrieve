
import csv
import re
import os
import string
import pickle
from collections import Counter
import numpy as np
import gensim

datapath = '../../Data/longyelp/'

def words_preprocess(phrase):
	""" preprocess a sentence: lowercase, clean up weird chars, remove punctuation """
	replacements = {
		'½': 'half',
		'—' : '-',
		'™': '',
		'¢': 'cent',
		'ç': 'c',
		'û': 'u',
		'é': 'e',
		'°': ' degree',
		'è': 'e',
		'…': '',
		}
	for k, v in replacements.items():
		phrase = phrase.replace(k, v)
	return str(phrase).lower().translate(str.maketrans('','',string.punctuation)).split()

def preprocess(data):
	maxLen = 0
	for k,v in data.items():
		tokens = re.split(' |\n|\"|\'|\.|!',v)
		tokens = list(filter(lambda a: a != '', tokens))
		proc_v = words_preprocess(' '.join(tokens))
		proc_v = list(map(lambda a: '_num_' if a.isdigit() else a, proc_v))
		data[k] = proc_v
		maxLen = max(maxLen,len(proc_v))
	return data, maxLen

def build_word2inds(descDict,min_token_instances=8,verbose=True):
	token_counter = Counter()
	for k,v in descDict.items():
		token_counter.update(v)
	vocab = set()
	for token, count in token_counter.items():
		if count >= min_token_instances:
			vocab.add(token)
	if verbose:
		print(('Keeping %d / %d tokens with enough instances'
			  % (len(vocab), len(token_counter))))
	
	if len(vocab) < len(token_counter):
		vocab.add('<UNK>')
		if verbose:
			print('adding special <UNK> token.')
	else:
		if verbose: 
			print('no <UNK> token needed.')

	# build word2inds
	word2inds = {}
	next_idx = 1
	for token in vocab:
		word2inds[token] = next_idx
		next_idx = next_idx + 1
	return word2inds # start from 1, leave 0 to pad.

def encodeT(tokens,word2inds,maxLen,f):
	encoded = np.zeros(maxLen, dtype=np.int32)
	# isallunk = 1
	tokens_for_observe = []
	for i, token in enumerate(tokens):
		if token in word2inds:
			# if token not in stopwords:
			# 	isallunk = 0
			encoded[i] = word2inds[token]
			tokens_for_observe.append(token)
		else:
			encoded[i] = word2inds['<UNK>']
			tokens_for_observe.append('<UNK>')
	f.write(' '.join(tokens_for_observe)+'\n')
	# if cnt==0:
	# 	print("all <unk> desc...")
	return encoded#, isallunk

def encodeTexts(descDict, word2inds, maxLen):
	encoded_desc = np.zeros([len(descDict), maxLen])
	lengths = np.zeros(len(descDict))
	cnt = 0
	# allunk_cnt = 0
	f = open(os.path.join(datapath, 'tokens_for_observe'),'w')
	for k,v in descDict.items():
		assert(k==cnt)
		lengths[cnt] = len(v)
		encoded = encodeT(v,word2inds,maxLen,f) #, allunk
		# allunk_cnt += allunk
		encoded_desc[cnt] = encoded
		cnt += 1
	f.close()
	# print('allunk_cnt: '+str(allunk_cnt))
	return encoded_desc, lengths

# read file
print('reading data...')
data = {}
label = {}
with open(os.path.join(datapath, 'yelp_training_set_review.csv'),'r') as f:
	rd = csv.DictReader(f)
	cnt = 0
	for line in rd:
		if int(line['stars'])==3:
			continue
		text = line['text']
		ispos = int(int(line['stars'])>3)
		data[cnt] = text
		label[cnt] = ispos
		cnt += 1

label = list(label.values())

print('preprocessing data...')
# preprocess
dataprep, maxLen = preprocess(data)

print('building word2inds...')
# filter out low freq words
word2inds = build_word2inds(dataprep,min_token_instances=30)

print('encoding data...')
# encode descDict to inds, return wnid2ind, wordinds matrix.
encoded_data,lengths = encodeTexts(dataprep, word2inds, maxLen)

# save word Dict
print('save encoded data...')
ratio = 0.95
numData = len(dataprep)
assert(len(dataprep)==len(label))
numTrain = int(numData*ratio)
numVal = numData - numTrain
print(' - numTrain: '+str(numTrain))
print(' - numVal: '+str(numVal))

dataEnc_train = {}
dataEnc_train['label'] = label[:numTrain]
dataEnc_train['data'] = encoded_data[:numTrain]
dataEnc_train['lengths'] = lengths[:numTrain]
with open(os.path.join(datapath, 'train.pkl'),'wb') as f:
	pickle.dump(dataEnc_train,f)

dataEnc_val = {}
dataEnc_val['label'] = label[numTrain:]
dataEnc_val['data'] = encoded_data[numTrain:]
dataEnc_val['lengths'] = lengths[numTrain:]
with open(os.path.join(datapath, 'val.pkl'),'wb') as f:
	pickle.dump(dataEnc_val,f)


# save inds2embeddings
print('start word2vec...')
model = gensim.models.KeyedVectors.load_word2vec_format('~/Downloads/GoogleNews-vectors-negative300.bin', binary=True)

numwords = len(word2inds)

word2inds['<START>'] = numwords+1
word2inds['<END>'] = numwords+2
word2inds['<PAD>'] = 0

word2vec = np.zeros([numwords+3,300])

wastedCnt = 0
for word,idx in word2inds.items():
	if word in model.wv:
		word2vec[idx] = model.wv[word]
	else:
		wastedCnt += 1
		word2vec[idx] = np.random.uniform(-1,1,300)

print('wasted words: '+str(wastedCnt))

VocabData = {}
VocabData['word_dict'] = word2inds
VocabData['word_embs'] = word2vec

with open(os.path.join(datapath, 'vocabs.pkl'), 'wb') as f:
	pickle.dump(VocabData,f)
