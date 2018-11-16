import bcolz
import numpy as np
import pickle
import os

# words = []
# idx = 0
# word2idx = {}
# vectors = bcolz.carray(np.zeros(1), rootdir=f'/Users/CenIII/Downloads/glove.twitter.27B/glove.twitter.27B.100d.dat', mode='w')

# with open(f'/Users/CenIII/Downloads/glove.twitter.27B/glove.twitter.27B.100d.txt', 'rb') as f:
#     for l in f:
#         line = l.decode().split()
#         if(len(line[1:])==100):
# 	        word = line[0]
# 	        words.append(word)
# 	        word2idx[word] = idx
# 	        idx += 1
#         	vect = np.array(line[1:]).astype(np.float)
#         	vectors.append(vect)
#         # if idx%100==0:
#         # 	print(len(vectors))
    
# vectors = bcolz.carray(vectors[1:].reshape((1193513, 100)), rootdir=f'/Users/CenIII/Downloads/glove.twitter.27B/glove.twitter.27B.100d.dat', mode='w')
# vectors.flush()
# pickle.dump(words, open(f'/Users/CenIII/Downloads/glove.twitter.27B/glove.twitter.27B.100d_words.pkl', 'wb'))
# pickle.dump(word2idx, open(f'/Users/CenIII/Downloads/glove.twitter.27B/glove.twitter.27B.100d_idx.pkl', 'wb'))


vectors = bcolz.open(f'/Users/CenIII/Downloads/glove.twitter.27B/glove.twitter.27B.100d.dat')[:]
words = pickle.load(open(f'/Users/CenIII/Downloads/glove.twitter.27B/glove.twitter.27B.100d_words.pkl', 'rb'))
word2idx = pickle.load(open(f'/Users/CenIII/Downloads/glove.twitter.27B/glove.twitter.27B.100d_idx.pkl', 'rb'))

glove = {w: vectors[word2idx[w]] for w in words}

wordlist = []

for dataset in ['yelp/']:
	filelist = os.listdir('../../Data/'+dataset)
	for file in filelist:
		with open('../../Data/'+dataset+file,'r') as f:
			line = f.readline()
			while line:
				# suggestions = sym_spell.lookup_compound(line, max_edit_distance_lookup)
				wordlist += line.split(' ')
				line = f.readline()

wordlist.append('<unk>')
wordlist.append('@@START@@')
wordlist.append('@@END@@')
wordlist.append('<m_start>')
wordlist.append('<m_end>')
vocabs = set(wordlist)

print(len(vocabs))

wordDict = {}
word2vec = []
cnt=0
wastewords = []
for word in vocabs:
	if word in glove:
		word2vec.append(glove[word])
		wordDict[word] = cnt
		cnt += 1
	else:
		wastewords.append(word)
		word2vec.append(np.random.uniform(-1,1,100))
		wordDict[word] = cnt
		cnt += 1
print(len(wastewords))

word2vec = np.array(word2vec)
# with open('./word2vec', "wb") as fp:   #Pickling
np.save('glovevec.npy',word2vec)
with open('./gloveDict', "wb") as fp:   #Pickling
	pickle.dump(wordDict, fp)