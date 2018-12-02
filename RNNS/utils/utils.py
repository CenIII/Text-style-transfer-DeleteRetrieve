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
	"""Pack a series of samples into a batch. Each Sample is a tuple (brkSentence, [style], sentence, marker).

	The default collate_fn of Pytorch use torch.stack() assuming every sample has the same size.
	For this task, the length of sentences may vary, so do the sample generated.
	See https://jdhao.github.io/2017/10/23/pytorch-load-data-and-make-batch/ for more information.

	Returns:
		A dict of list with all its member being a list of length batch_size
	"""
	# TODO: Some work might be done in __getitem__().
	# print('>>>>>>>batch: '+str(batch))
	batchSize = len(batch)

	def extract(ind):
		maxLen = 0
		lengths = []
		for seq in batch:
			seqLen = len(seq[ind])
			lengths.append(seqLen)
			# The maxLen is the largest length in this batch
			if seqLen > maxLen:
				maxLen = seqLen
		packed = np.zeros([batchSize, maxLen])
		# Put each sentence on a row, filled with zero.
		for i in range(batchSize):
			packed[i][:lengths[i]] = batch[i][ind]
		lengths = np.array(lengths)
		# inds = np.argsort(lengths)[::-1]
		return torch.LongTensor(packed), torch.tensor(lengths) # LongTensor is used for better precision

	brk_sentence, seqLengths = extract(0)
	style, styleLengths = extract(1) 
	sent, stLengths = extract(2)
	marker, mkLengths = extract(3)
	return {'brk_sentence': brk_sentence,
			'bs_inp_lengths':seqLengths,
			'style': style,
			'sentence':sent,
			'st_inp_lengths':stLengths,
			'marker':marker,
			'mk_inp_lengths':mkLengths }

def reloadModel(model,config):
	"""Load checkpoint of the model using torch.load()

	Args:
		model: which pytorch model to load the parameters.
		config: information of loader. the checkpoint is at contPath/resume_file
	Returns:
		model: the model with parameters loaded.
	"""
	# 
	checkpoint = os.path.join(config['contPath'], config['opt'].resume_file)
	print("=> Reloading checkpoint '{}': model".format(checkpoint))
	# store all the tensors on The CPU. See https://pytorch.org/docs/master/torch.html?highlight=load#torch.load
	checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
	# model.load_state_dict(self.checkpoint['state_dict'])
	model_dict = model.state_dict()
	# TODO: The loading part contains unnecessary steps
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

def makeInp(inputs):
	"""Move tensors onto GPU if available.

	Args:
		inputs: A dict with a batch of word-indexed data from DataLoader. Contains
			['brk_sentence', 'bs_inp_lengths', 'style', 'sentence', 'st_inp_lengths', 'marker', 'mk_inp_lengths']
	Returns:
		inputs: The dict with same structure but stored on GPU.
	"""
	if torch.cuda.is_available():
		for key in inputs:
			inputs[key] = inputs[key].cuda()
	return inputs
