from loader import LoaderHandler
# from evaluator import Evaluator
from trainer import Trainer
from model import Seq2seq, Criterion
from utils import ConfigParser, utils

def runTrain(config):
	loader = LoaderHandler(config['loader'])
	net = Seq2seq(**config['model'])
	if config['opt'].continue_exp:
		net = utils.reloadModel(net, config)
	crit = Criterion(config['crit'])
	trainer = Trainer(config['trainer'],config['expPath'])
	evaluator = None#Evaluator(config['evaluator'],config['expPath'])
	trainer.train(loader, net, crit, evaluator)

def runVal():
	loader = LoaderHandler(config['loader'])
	net = Seq2seq(**config['model'])
	net = utils.reloadModel(net,config)
	evaluator = Evaluator(config['evaluator'],config['expPath'])

def main():
	config = ConfigParser.parse_config()
	if config['opt'].mode == 'train':
		runTrain(config)
	else:
		runVal()
	
if __name__ == '__main__':
	main() 