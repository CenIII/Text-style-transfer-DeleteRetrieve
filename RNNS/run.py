from loader import LoaderHandler
# from evaluator import Evaluator
from trainer import Trainer
# from model import Seq2seq, Criterion
from utils import ConfigParser, Tools

def runTrain(config):
	print('loader...')
	loader = LoaderHandler(config['loader'])
	if config['opt'].continue_exp:
		net = Tools.reloadModel(config)
	print('net...')
	net = Seq2seq(**config['model'])
	print('crit...')
	crit = None#Criterion(config['crit'])
	print('trainer...')
	trainer = Trainer(config['trainer'],config['expPath'])
	print('evaluator...')
	evaluator = None#Evaluator(config['evaluator'],config['expPath'])
	print('start to train...')
	trainer.train(loader, net, crit, evaluator)

def runVal():
	loader = LoaderHandler(config['loader'])
	net = Tools.reloadModel(config)
	evaluator = Evaluator(config['evaluator'],config['expPath'])

def main():
	config = ConfigParser.parse_config()
	if config['opt'].mode == 'train':
		runTrain(config)
	else:
		runVal()
	
if __name__ == '__main__':
	main() 