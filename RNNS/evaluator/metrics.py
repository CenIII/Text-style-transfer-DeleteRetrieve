import os
import pickle
import torch
import tqdm
import numpy as np
import nltk.translate.bleu_score
from utils import makeInp


class Metrics:
    def __init__(self, model_path, bleu_reference_path, net, word_dict_path):
        self.model_path = model_path
        self.bleu_reference_path = bleu_reference_path
        self.net = net
        self.wordDict = word_dict_path

    def reloadClassifierModel(self, model, model_path):

        print("=> Reloading checkpoint '{}': model".format(model_path + '/selfatt.pt'))
        checkpoint = torch.load(model_path + '/selfatt.pt')
        # model.load_state_dict(self.checkpoint['state_dict'])
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {}
        for k, v in checkpoint.items():
            if(k in model_dict):
                pretrained_dict[k] = v
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
        return model
    
    def classifierMetrics(self, preds):
        net = self.reloadClassifierModel(self.net, self.model_path)
        correct = 0
        total = len(preds['positive']) + len(preds['negative'])
        def test(net, label):
            correct = 0
            if label == 0:
                key = 'negative'
            else:
                key = 'positive'
            for sentence in preds[key]:
                output,_ = net(makeInp(self.wrap(sentence)))
                if output[0][0] > output[0][1]:
                    output_label = 0
                else:
                    output_label = 1
                if output_label == label:
                    correct += 1
            return correct
        correct += test(net, 0)
        correct += test(net, 1)
        
        acc = correct / total

        return acc

    # def extractMarkerWord(self, train):
    #     net = self.reloadClassifierModel(self.net, self.model_path)
    #     def extract(net, label):
    #         if label == 0:
    #             key = 'negative'
    #         else:
    #             key = 'positive'
    #         for sentence in train[key]:
    #             output,_ = net(makeInp(self.wrap(sentence)))
    #             marker_words = []
    #             for i in range(len(sentence)):
    #                 if i<len(sentence)-1:
    #                     part_sentence = sentence[:i] + sentence[i+1:]
    #                 else:
    #                     part_sentence = sentence[:i]
    #                 part_output,_ = net(makeInp(self.wrap(part_sentence)))
    #     # there are three types of words:
    #     # 1. change the sentence to a totally different style (like not)
    #     # 2. make the classifier unable to decide the type of the sentence (what we will mark as marker words)
    #     # 3. has little to do with the style 
    #                 if output[0][0] > output[0][1]:
    #                     if part_output[0][0]<min(0.6,output[0][0]) and part_output[0][0]>max(0.4, output[0][1]):
    #                         marker_words.append(sentenc'../../AuxData/wordDict'e[i])
    #                 else:'../../AuxData/wordDict'
    #                     if part_output[0][0]<min(0.6,ou'../../AuxData/wordDict'tput[0][1]) and part_output[0][0]>max(0.4, output[0][0]):
    #                         marker_words.append(sentenc'../../AuxData/wordDict'e[i])
    #             print(sentence)'../../AuxData/wordDict'
    #             print('marker words are', marker_words)'../../AuxData/wordDict'
    #             print()
    #     extract(net, 0)
    #     extract(net, 1)


    def wrap(self, sentence):
        indArr = self.word2index(sentence)
        return indArr

    def word2index(self, sentence):
        with open(self.wordDict,"rb") as fp:
            wordDict = pickle.load(fp)
        sos_id = wordDict['@@START@@']
        resList = []
        indArr = []
        indArr.append(sos_id)
        for i in range(len(sentence)):
            word = sentence[i]
            if word in wordDict:
                indArr.append(wordDict[word])
        # indArr.append(self.eos_id) 
        indArr = torch.LongTensor(np.array(indArr))
        return indArr.unsqueeze(0)

    def bleuMetrics(self, preds):
        references = self.loadReferences()
        hypothesis = preds['positive'] + preds['negative']
        score = nltk.translate.bleu_score.corpus_bleu(references, preds['positive'] + preds['negative'])
        score2 = 0
        for i in range(len(references)):            
            score2 += nltk.translate.bleu_score.modified_precision(references[i], hypothesis[i], n = 2)

        score2 /= len(references)
        return float(score2)
       

    def loadReferences(self):
        #pos and neg
        references = []
        for i in range(2):
            with open(self.bleu_reference_path + '.' + str(i), 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                words = line.split('\t')[1].strip().split()
                references.append([words])

        return references