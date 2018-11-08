from loader import LoadHandler
from evaluator import Evaluator
from trainer import Trainer
from model import Seq2seq, Criterion

def runTrain():
	loader = LoadHandler()
	net = Seq2seq()
	crit = Criterion()
	trainer = Trainer()
	evaluator = Evaluator()
	trainer.train(loader, net, crit, evaluator)

def runVal():
	loader = 
	net = 
	evaluator = 
	# get and save prediction

	# metric evaluation



def main():
	if config['opt'].mode == 'train':
		run_train()
	else
		run_val()
	
if __name__ == '__main__':
	main() 