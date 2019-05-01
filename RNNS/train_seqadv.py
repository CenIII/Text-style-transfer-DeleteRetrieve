import os
import torch
from torch.utils.data import DataLoader
from loader import YelpDataset
from model import Seq2att, AdvClassifier, DecCriterion, AdvCriterion
import torch.nn.functional as F
import tqdm
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def makeInp(*inps):
	ret = []
	for inp in inps:
		ret.append(inp.to(device))
	return ret

def train(loader, net, advclss, crit1, crit2):
	print('start to train...')
	optim_seqdec = torch.optim.Adam(list(filter(lambda p: p.requires_grad, net.decoder.parameters())), 0.0001)
	optim_adv = torch.optim.Adam(list(filter(lambda p: p.requires_grad, net.encoder.parameters()))+list(net.linear_first.parameters())+list(advclss.parameters()), 0.0001)
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
				enc_outs, dec_outs = net(sents, labels, lengths, advclss)  # outputs: score array, out lengths, att matrix, enc out left over
				# crit 1: binary cross entropy on carried steps
				loss1 = crit1(dec_outs['score'],labels)
				# backward, optim_seqdec.step()
				net.zero_grad()
				advclss.zero_grad()
				loss1.backward()
				optim_seqdec.step()
				# net.zero_grad, advclss.zero_grad()
				adv_orig_outs = advclss(key=enc_outs['enc_outputs_key'],hiddens=enc_outs['enc_outputs']) 
				adv_left_outs = advclss(key=enc_outs['left_over_key'],hiddens=enc_outs['enc_outputs']) 
				# crit 2: binary cross entropy on adv results
				loss2 = crit2([adv_orig_outs,adv_left_outs], labels)
				# backward, optim_adv
				net.zero_grad()
				advclss.zero_grad()
				loss2.backward()
				optim_adv.step()

			max_out_len = max(dec_outs['length'])

			qdar.set_postfix(loss1=lstr(loss1),loss2=lstr(loss2),max_out_len=max_out_len)

		epoch += 1


# init loader
trainData = YelpDataset()
ldTrain = DataLoader(trainData, batch_size=16, shuffle=True, num_workers=2, collate_fn=trainData.collate_fn)

# init seq2att net
seq2att = Seq2att(hidden_size=2048, style_size=100, input_dropout_p=0, 
					max_len=200, dropout_p=0, n_layers=1, bidirectional=False, 
					rnn_cell='lstm', decode_function=F.sigmoid).to(device)
# init adv classifier
advclss = AdvClassifier(16,2048,1024,1,n_classes=1).to(device)

# init crit 
crit1 = DecCriterion().to(device)
crit2 = AdvCriterion().to(device)

# start train
train(ldTrain, seq2att, advclss, crit1, crit2)