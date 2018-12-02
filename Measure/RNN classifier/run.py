import torch
from loader import LoaderHandler
from evaluator import Evaluator
from trainer import Trainer
from model import Classifier, Criterion
from utils import ConfigParser, utils
import fileinput
from evaluator import Metrics

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
	net = Classifier(**config['model'])
	if config['opt'].continue_exp:
		net = utils.reloadModel(net, config)
	crit = Criterion(config['crit'])
	if torch.cuda.is_available():
		net = net.cuda()
		crit = crit.cuda()
	trainer = Trainer(config['trainer'],config['expPath'])
	evaluator = Evaluator(config['evaluator'],config['expPath'])
	trainer.train(loader, net, crit, evaluator, config)

def runVal(config):
	def isValidSentence(sentence):
		if(sentence == [] or 
			sentence == 'Positive' or 
			sentence == 'Negative'):
			return False
		return True

	def loadData(lines,label):
		key = 'positive' if label == 1 else 'negative'
		for line in lines:
			sentence = line.split(' ')[:-1]
			if isValidSentence(sentence):
				preds[key].append(sentence)
	net = Classifier(**config['model'])
	if torch.cuda.is_available():
		net = net.cuda()
	metric = Metrics('exp/pose', '../../Data/yelp/reference',net)
	preds = {'positive':[],'negative':[]}
	with open('../../Data/yelp/sentiment.test.0','r') as f:
		loadData(f.readlines(), 0)
	with open('../../Data/yelp/sentiment.test.1','r') as f:
		loadData(f.readlines(), 1)
	acc = metric.classifierMetrics(preds)
	print(acc)

	score = metric.bleuMetrics(preds)
	print(score)

def extractClassifierMarker(config):
	def isValidSentence(sentence):
		if(sentence == [] or 
			sentence == 'Positive' or 
			sentence == 'Negative'):
			return False
		return True

	def loadData(lines,label):
		key = 'positive' if label == 1 else 'negative'
		for line in lines:
			sentence = line.split(' ')[:-1]
			if isValidSentence(sentence):
				preds[key].append(sentence)
	net = Classifier(**config['model'])
	if torch.cuda.is_available():
		net = net.cuda()
	metric = Metrics('exp/pose', '../../Data/yelp/reference',net)
	preds = {'positive':[],'negative':[]}
	with open('../../Data/yelp/sentiment.train.0','r') as f:
		loadData(f.readlines(), 0)
	with open('../../Data/yelp/sentiment.train.1','r') as f:
		loadData(f.readlines(), 1)
	metric.extractMarkerWord(preds)

def runOnline(config):
	loader = LoaderHandler(config)
	net = Seq2seq(**config['model'])
	if torch.cuda.is_available():
		net = net.cuda()
	net = utils.reloadModel(net,config)
	evaluator = Evaluator(config['evaluator'],config['expPath'])
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
	elif mode == 'online':
		runOnline(config)
	elif mode == 'extract':
		extractClassifierMarker(config)
	else:
		pass
	
if __name__ == '__main__':
	main() 
