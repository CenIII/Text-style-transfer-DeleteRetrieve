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

**All data and files are included and you can now skip to the "Running the experiment" section.** In case you lose any data or checkpoint to test the code, below are some download links.

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
First be sure that you are at "./RNNS" working directory. 
```
cd RNNS
```

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
python run.py -m train -e try01
```
The checkpoint will be saved at `RNNS/exp/try01` folder.

### Evaluate model
Provide metrics on the test set, including BLEU, accuracy and perplexity. Assume the experiment result is in `RNNS/exp/exp0`
```
python3 run.py -m test -c bestexp -e bestexp -f bestmodel.pth.tar -t 1
```
The "bestexp" folder exists and contains the pretrained checkpoint for you to do testing. If it not exist, checkout the "Installing" section above to download a checkpoint for testing. 

Here the "-t" option indicates whether you want to transfer the sentiment or just do reconstruction. The default value is 0, so during test time you need to set it to 1 so that the model will transfer the inputs to the opposite sentiment. 

### Online demo
Entering the source sentiment and sentence, the transferred sentence will be displayed.
```
python3 run.py -m online -c bestexp -e bestexp -f bestmodel.pth.tar -t 1
```


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

