from loader import LoaderHandler
from evaluator import Evaluator
from trainer import Trainer
from model import Seq2seq, Criterion
from utils import ConfigParser, utils

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
	loader = LoaderHandler(config['loader'])
	net = Seq2seq(**config['model'])
	if config['opt'].continue_exp:
		net = utils.reloadModel(net, config)
	crit = Criterion(config['crit'])
	if torch.cuda.is_available():
		net = net.cuda()
		crit = crit.cuda()
	trainer = Trainer(config['trainer'],config['expPath'])
	evaluator = Evaluator(config['evaluator'],config['expPath'])
	trainer.train(loader, net, crit, evaluator)

def runVal(config):
	loader = LoaderHandler(config['loader'])
	net = Seq2seq(**config['model'])
	net = utils.reloadModel(net,config)
	evaluator = Evaluator(config['evaluator'],config['expPath'])
	evaluator.predict(loader,net)

def main():
	config = ConfigParser.parse_config()
	if config['opt'].mode == 'train':
		runTrain(config)
	else:
		runVal(config)
	
if __name__ == '__main__':
	main() 