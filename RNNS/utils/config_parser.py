import argparse
import json
import os

class ConfigParser: 

	def _create_command_line_parser():
		parser = argparse.ArgumentParser()
		parser.add_argument('-m', '--mode', type=str, choices=['train','val','test'], default='train', help='running mode: [train | val | test]' )
		parser.add_argument('-c', '--continue_exp', type=str, help='continue exp')
		parser.add_argument('-e', '--exp', type=str, default='pose', help='experiments name: [anystring]')
		# parser.add_argument('-v', '--evaluate', action='store_true', help='load predicted results')
		parser.add_argument('-f', '--resume_file', type=str, help='resume_file_name')
		return parser
	# Command line parser
	def _parse_command_line():
		parser = ConfigParser._create_command_line_parser()
		args = parser.parse_args()
		print('options: ')
		print(json.dumps(vars(args), indent = 4))
		return args

	# Config file parser
	def _parse_config_file(opt):
		with open('./config.json', 'r') as f:
			config = json.load(f)
		print('config: ')
		print(json.dumps(config, indent = 4))
		config['opt'] = opt
		config['expPath'] = os.path.join(config['expPath'], opt.exp)
		return config

	def parse_config():
		# parse command line
		opt = ConfigParser._parse_command_line()
		# parse config file, combine opt and config into config. 
		config = ConfigParser._parse_config_file(opt)
		return config
