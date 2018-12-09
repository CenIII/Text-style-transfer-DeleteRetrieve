# Text style transfer - DeleteRetrieve
The task is sentiment transfer of text using non-parallel data. Our model first delete the phrases in a sentence with the most of style information, and use the broken sentence together with trained target style embedding to reconstruct the transferred sentence.

## Getting Started

These instructions will provide information about how to use our code.

### Prerequisites

Several packages are needed to run the code. To install them, run
```
pip install -r requirement.txt
```
Or install whatever packages that are missing manually, for example
```
pip install torch
```

### Installing

The checkpoint of trained model are needed for evaluating and online transferring. 
<!-- You can download [checkpoint](https://drive.google.com/file/d/1dCYjVoylK4BOgiKlGDolpJEU2TQq3LtK/view?usp=sharing) here. After downloading, unzip to target folder

```
unzip exp.zip
mv exp path_of_this_git/RNNS
```
-->


You can download the [word2vec embedding](https://code.google.com/archive/p/word2vec/) pre-trained on Google New Corpus. 

The Yelp dataset used can be find [here](https://github.com/shentianxiao/language-style-transfer), put the `data` folder into this git 



## Running the experiment
To get help about setting parameters with command line arguments, run
```
python run.py -h
```
All the hyper-parameters have their default value set already. But if you want to modify them, you can input them either through command line or in `config.json`. All following operation assume the current working directory to be `RNNS` folder.

### Pre-train language model

Train the language models for two sentiments

```
python run.py -m pretrain
```

### Training model

To train the whole model
```
python run.py -m train
```
The checkpoint will be saved at `RNNS/exp` folder.

### Evaluate model
Provide metrics on the test set, including BLEU, accuracy and perplexity. Assume the experiment result is in `RNNS/exp/exp0`
```
python3 run.py -m test -c exp0 -e exp0 -f bestmodel.pth.tar
```
### Online transferring
Entering the source sentiment and sentence, the transferred sentence will be displayed.
```
python3 run.py -m online -c exp0 -e exp0 -f bestmodel.pth.tar
```



## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

