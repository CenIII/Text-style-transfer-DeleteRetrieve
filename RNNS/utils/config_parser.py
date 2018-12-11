import argparse
import json
import os

class ConfigParser: 

	def _create_command_line_parser():
		"""	Specify the expected arguments
		Returns:
			parser: An ArgumentParser object from standard library argparse.
		"""
		parser = argparse.ArgumentParser()
		parser.add_argument('-m', '--mode', type=str, choices=['train','val','test','online','pretrain'], default='train', help='Running mode: [pretrain | train | val | test | online]' )
		parser.add_argument('-c', '--continue_exp', type=str, help='The exp folder name from which you want to continue. This option is valid for train, val and test modes.')
		parser.add_argument('-e', '--exp', type=str, default='pose', help='The exp folder name to which you want to save your checkpoint or outputs, e.g. try01. This is valid for train, val and test modes.')
		# parser.add_argument('-v', '--evaluate', action='store_true', help='load predicted results')
		parser.add_argument('-f', '--resume_file', type=str, default='checkpoint.pth.tar' ,help='The checkpoint file name.')
		parser.add_argument('-p', '--epoch', type=str, default='0', help='Epoch num you want to start from')
		parser.add_argument('-t', '--trans_style', type=int, default=0, help='Want to transfer to opposite style or just reconstruction? Can not be 1 when training!')
		return parser
	# Command line parser
	def _parse_command_line():
		"""Finish command line parsing
		Returns:
			args: argparse.Namespace object 
		"""
		parser = ConfigParser._create_command_line_parser()
		args = parser.parse_args()
		print('options: ')
		print(json.dumps(vars(args), indent = 4))
		if args.mode=='train':
			assert(args.trans_style==0)
		return args

	# Config file parser
	def _parse_config_file(opt):
		"""Read JSON formatted configuration file.
		Args:
			opt: A Namespace object formed by parsing command line.
		Returns:
			config: A dict contains information of loader,trainer, model, evaluator, crit, expPath and opt(parsed command line)
		"""
		with open('./config.json', 'r') as f:
			config = json.load(f)
		if opt.continue_exp:
			config['contPath'] = os.path.join(config['expPath'], opt.continue_exp)
		opt.epoch = float(opt.epoch)
		config['expPath'] = os.path.join(config['expPath'], opt.exp)
		config['loader']['isTrans'] = int(opt.trans_style)
		print('config: ')
		print(json.dumps(config, indent = 4))
		config['opt'] = opt
		return config

	def parse_config():
		"""The top level function of parsing."""
		# parse command line
		opt = ConfigParser._parse_command_line()
		# parse config file, combine opt and config into config. 
		config = ConfigParser._parse_config_file(opt)
		return config
