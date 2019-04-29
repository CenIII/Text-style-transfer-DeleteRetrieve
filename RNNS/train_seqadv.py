import os
import torch
from torch.utils.data import DataLoader
from loader import YelpDataset
from model import Seq2att, AdvClassifier, Criterion
import torch.nn.functional as F
import tqdm
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def makeInp(*inps):
	ret = []
	for inp in inps:
		ret.append(inp.to(device))
	return ret

def train(loader, net, advclss, crit):
	print('start to train...')
	optimizer = torch.optim.Adam(list(filter(lambda p: p.requires_grad, net.parameters()))+list(advclss.parameters()), 0.0001)
	# train
	def lstr(lss):
		return str(np.round(np.float(lss),3))

	epoch = 0

	while True:
		print('epoch: '+str(epoch))
		net.train()
		ld = iter(loader)
		numIters = len(ld)
		qdar = tqdm.tqdm(range(numIters),total= numIters,ascii=True)
		for itr in qdar: 
			sents, labels, lengths = makeInp(*next(ld))
			with torch.set_grad_enabled(True):
				seq2att_outs = net(sents, labels, lengths)  # outputs: score array, out lengths, att matrix, enc out left over
				advclss_outs = advclss(seq2att_outs['left_over']) # outputs: score array
				loss1,loss2,loss3 = crit(seq2att_outs, advclss_outs, labels)
			loss = loss1+loss2+loss3
			max_out_len = max(seq2att_outs['length'])
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			qdar.set_postfix(loss1=lstr(loss1),loss2=lstr(loss2),loss3=lstr(loss3),max_out_len=max_out_len)

		epoch += 1


# init loader
trainData = YelpDataset()
ldTrain = DataLoader(trainData, batch_size=16, shuffle=True, num_workers=2, collate_fn=trainData.collate_fn)

# init seq2att net
seq2att = Seq2att(hidden_size=2048, style_size=100, input_dropout_p=0, 
					max_len=200, dropout_p=0, n_layers=1, bidirectional=False, 
					rnn_cell='lstm', decode_function=F.sigmoid).to(device)
# init adv classifier
advclss = AdvClassifier(16,2048,512,1,n_classes=1).to(device)

# init crit 
crit = Criterion().to(device)

# start train
train(ldTrain, seq2att, advclss, crit)