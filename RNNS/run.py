from loader import LoadHandler
from evaluator import Evaluator
from trainer import Trainer
from model import Seq2seq, Criterion
from utils import ConfigParser, Tools

def runTrain(config):
	loader = LoadHandler(config['loader'])
	if config['opt'].continue_exp:
		net = Tools.reloadModel(config)
	net = Seq2seq(config['model'])
	crit = Criterion(config['crit'])
	trainer = Trainer(config['trainer'],config['expPath'])
	evaluator = Evaluator(config['evaluator'],config['expPath'])
	trainer.train(loader, net, crit, evaluator)

def runVal():
	loader = LoadHandler(config['loader'])
	net = Tools.reloadModel(config)
	evaluator = Evaluator(config['evaluator'],config['expPath'])

def main():
	config = ConfigParser.parse_config()
	if config['opt'].mode == 'train':
		run_train()
	else
		run_val()
	
if __name__ == '__main__':
	main() 