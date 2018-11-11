import numpy as np
import torch

def subset(array):
    result = []
    n = len(array)
    for k in range(1, n):
        for i in range(n-k+1):
            result.append(array[i:i+k])
    return result

def seq_collate(batch):
	# print('>>>>>>>batch: '+str(batch))
	batchSize = len(batch)
	def extract(ind):
		maxLen = 0
		lengths = []
		for seq in batch:
			seqLen = len(seq[ind])
			lengths.append(seqLen)
			if seqLen > maxLen:
				maxLen = seqLen
		packed = np.zeros([batchSize, maxLen])
		for i in range(batchSize):
			packed[i][:lengths[i]] = batch[i][ind]
		lengths = np.array(lengths)
		inds = np.argsort(lengths)[::-1]
		return torch.LongTensor(packed[inds]), torch.tensor(lengths[inds])
	brk_sentence, seqLengths = extract(0)
	marker, mkLengths = extract(1) 
	sent, stLengths = extract(2)
	return {'brk_sentence': brk_sentence,
			'bs_inp_lengths':seqLengths,
			'marker': marker,
			'mk_inp_lengths':mkLengths,
			'sentence':sent,
			'st_inp_lengths':stLengths }