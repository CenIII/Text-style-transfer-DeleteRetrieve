import os
import torch
from loader import YelpDataset
from model import Seq2att, AdvClassifier, Criterion

def train(loader, net, advclss, crit):
	print('start to train...')
	optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), self.lr)
	# train
	epoch = 0
	while True:
		print('epoch: '+str(epoch))
		net.train()
		ld = iter(loader)
		numIters = len(ld)
		qdar = tqdm.tqdm(range(numIters),total= numIters,ascii=True)
		for itr in qdar: 
			sents, labels, lengths = next(ld)
			with torch.set_grad_enabled(True):
				seq2att_outs = net(sents, labels, lengths)  # outputs: score array, out lengths, att matrix, enc out left over
				advclss_outs = advclss(seq2att_outs, labels, lengths) # outputs: score array
				loss = crit(seq2att_outs, advclss_outs, labels)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			qdar.set_postfix(loss=str(np.round(loss.cpu().detach().numpy(),3)))

		epoch += 1


# init loader
trainData = YelpDataset()
ldTrain = DataLoader(trainData, batch_size=128, shuffle=True, num_workers=2, collate_fn=trainData.collate_fn)

# init seq2att net
seq2att = Seq2att(hidden_size=2048, style_size=100, input_dropout_p=0, 
					max_len=200, dropout_p=0, n_layers=1, bidirectional=False, 
					rnn_cell='lstm', decode_function=F.sigmoid)
# init adv classifier
advclss = AdvClassifier(128,2048,512,1,n_classes=1)

# init crit 
crit = Criterion()

# start train
train(ldTrain, seq2att, advclss, crit)