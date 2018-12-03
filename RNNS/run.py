import torch
from loader import LoaderHandler
from evaluator import Evaluator
from trainer import Trainer
from model import Seq2seq, Criterion
from utils import ConfigParser, utils
import fileinput

#todo: 
# 1. add word ind trans tools as a class to utils? can be used by evaluator and loader.
# 2. make delete and retrieve a class? encapsulate into loader. 

# reorganize the code to 
# folder:
	# AuxDATA
	# Data
	# Model
	# Tools
		# trainer
		# loader
		# delete&retrieve
		# evaluator
		# metrics
		# utils
			# wordindvec
	# config
	# run.py

def runTrain(config):
	loader = LoaderHandler(config)
	net = Seq2seq(**config['model'])
	if config['opt'].continue_exp:
		net = utils.reloadModel(net, config)
	crit = Criterion(config['crit'])
	if torch.cuda.is_available():
		net = net.cuda()
		crit = crit.cuda()
	trainer = Trainer(config['trainer'],config['expPath'])
	evaluator = Evaluator(config['evaluator'],config['expPath'], config)
	trainer.train(loader, net, crit, evaluator, config)

def runVal(config):
	loader = LoaderHandler(config)
	net = Seq2seq(**config['model'])
	if torch.cuda.is_available():
		net = net.cuda()
	net = utils.reloadModel(net,config)
	evaluator = Evaluator(config['evaluator'],config['expPath'], config)
	bleu, acc = evaluator.evaluate(loader, net)
	print(bleu, acc)
	return 

def runOnline(config):
	loader = LoaderHandler(config)
	net = Seq2seq(**config['model'])
	if torch.cuda.is_available():
		net = net.cuda()
	net = utils.reloadModel(net,config)
	evaluator = Evaluator(config['evaluator'],config['expPath'],config)
	print("Enter your sentence and its style: (e.g.: 0 the chicken was horrible)")
	while True:
		line = input("#: ")
		line = line.split(' ')
		style = int(line[0])
		line = line[1:]
		pred = evaluator.predictLine(loader.ldDevEval, net, line, style)
		print(pred)


def main():
	config = ConfigParser.parse_config()
	mode = config['opt'].mode
	if mode == 'train':
		runTrain(config)
	elif mode == 'val':
		runVal(config)
	elif mode == 'test':
		runVal(config)
	elif mode == 'online':
		runOnline(config)
	else:
		pass
	
if __name__ == '__main__':
	main() 
