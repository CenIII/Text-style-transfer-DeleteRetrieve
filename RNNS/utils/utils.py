import numpy as np
import torch
import os

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
		# inds = np.argsort(lengths)[::-1]
		return torch.LongTensor(packed), torch.tensor(lengths)

	def extract_marker_lengths(ind):
		lengths = []
		maxlen = 0
		for seq in batch:
			numMk = len(seq[ind])
			maxlen = max([maxlen,numMk])
		lengths = np.zeros([batchSize,maxlen])
		k=0
		for seq in batch:
			numMk = len(seq[ind])
			tmp = [len(seq[ind][i]) for i in range(numMk)]
			lengths[k,:numMk] = np.array(tmp)
			k += 1

		# 	tmp = [len(seq[ind][0])]# for i in range(numMk)]
		# 	tmp += [0] if numMk==1 else [len(seq[ind][1])]
		# 	lengths.append(tmp)
		# lengths = np.array(lengths)
		return torch.tensor(lengths)

	brk_sentence, seqLengths = extract(0)
	style, styleLengths = extract(1) 
	sent, stLengths = extract(2)
	# marker, mkLengths = extract(3)
	mkLengths = extract_marker_lengths(3)
	return {'brk_sentence': brk_sentence,
			'bs_inp_lengths':seqLengths,
			'style': style,
			'sentence':sent,
			'st_inp_lengths':stLengths,
			# 'marker':marker,
			'mk_inp_lengths':mkLengths }

def reloadModel(model,config):
	checkpoint = os.path.join(config['contPath'], config['opt'].resume_file)
	print("=> Reloading checkpoint '{}': model".format(checkpoint))
	checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
	# model.load_state_dict(self.checkpoint['state_dict'])
	model_dict = model.state_dict()
	# 1. filter out unnecessary keys
	pretrained_dict = {}
	for k, v in checkpoint['state_dict'].items():
		if(k in model_dict):
			pretrained_dict[k] = v
	# 2. overwrite entries in the existing state dict
	model_dict.update(pretrained_dict)
	# 3. load the new state dict
	model.load_state_dict(model_dict)
	return model

def checkPath(config,style):
	
	checkpoint = os.path.join(config['neg_model']+'.pth.tar' if style==0 else config['pos_model']+'.pth.tar')
	print("=> Reloading checkpoint '{}': model".format(checkpoint))


def reloadLM(model=None,config=None,style=None):
	checkpoint = os.path.join(config['neg_model']+'.pth.tar' if style==0 else config['pos_model']+'.pth.tar')
	print("=> Reloading checkpoint '{}': model".format(checkpoint))

	checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
	model_dict = model.state_dict()
	# 1. filter out unnecessary keys
	pretrained_dict = {}
	for k, v in checkpoint['state_dict'].items():
		if(k in model_dict):
			pretrained_dict[k] = v
	# 2. overwrite entries in the existing state dict
	model_dict.update(pretrained_dict)
	# 3. load the new state dict
	model.load_state_dict(model_dict)
	
	# Freeze model
	for param in model.parameters():
		param.requires_grad = False
	return model

def makeInp(inputs):
	if torch.cuda.is_available():
		for key in inputs:
			inputs[key] = inputs[key].cuda()
	return inputs
