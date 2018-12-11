import os
import torch
from loader import LoaderHandler
from evaluator import Evaluator
from trainer import Trainer, LangTrainer
from model import Seq2seq, Criterion, languageModel
from utils import ConfigParser, utils
import fileinput

def runTrain(config):
	"""Train the main network"""
	loader = LoaderHandler(config)
	net = Seq2seq(**config['model'])
	if config['opt'].continue_exp:
		net = utils.reloadModel(net, config)
	crit = Criterion(config)
	if torch.cuda.is_available():
		net = net.cuda()
		crit = crit.cuda()
	trainer = Trainer(config['trainer'],config['expPath'])
	evaluator = Evaluator(config['evaluator'],config['expPath'], config)
	trainer.train(loader, net, crit, evaluator, config)

def runVal(config):
	"""Evaluate model on valid data"""
	loader = LoaderHandler(config)
	net = Seq2seq(**config['model'])
	if torch.cuda.is_available():
		net = net.cuda()
	net = utils.reloadModel(net,config)
	evaluator = Evaluator(config['evaluator'],config['expPath'], config)
	bleu, acc, lang_loss= evaluator.evaluate(loader, net)
	print(bleu, acc,lang_loss)
	return 

def runOnline(config):
	"""Online style tranfer for input sentence"""
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

def runPreTrain(config):
	"""Train the language model"""
	# utils.reloadLM(config=config)
	loader = LoaderHandler(config)
	# TODO: modify config.json 
	lm_pos = languageModel(**config['lang_model'])
	lm_neg = languageModel(**config['lang_model'])
	utils.checkPath(config,0)
	utils.checkPath(config,1)
	# import pdb;pdb.set_trace()
	if torch.cuda.is_available():
		lm_pos = lm_pos.cuda()
		lm_neg = lm_neg.cuda()

	lang_trainer = LangTrainer(config['trainer'],config['LmPath'])
	# TODO: fill in the parameters
	lang_trainer.train(loader,lm_pos,config,isStyle=1)
	lang_trainer.train(loader,lm_neg,config,isStyle=0)

def main():
	config = ConfigParser.parse_config() # Load parameters from command line and json file
	mode = config['opt'].mode
	if mode == 'train':
		runTrain(config)
	elif mode =='pretrain':
		runPreTrain(config)
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
