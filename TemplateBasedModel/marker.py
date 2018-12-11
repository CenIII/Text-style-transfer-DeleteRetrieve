
# coding: utf-8

# In[ ]:
import pickle

def get_dict(sen_array):
        tmp_dict = {}
        for i in sen_array:
                sens = i.split()
                for n in range(1, 5):
                        for l in range(0, len(sens)-n+1):
                                tmp = ' '.join(sens[l:l+n])
                                tmp_dict[tmp] = tmp_dict.get(tmp, 0) + 1
        return tmp_dict

def file_get_dict(filename):
    f = open(filename)
    res = get_dict(f.readlines())
    f.close()
    return res

def subset(array):
    result = []
    n = len(array)
    for k in range(1, n):
        for i in range(n-k+1):
            result.append(array[i:i+k])
    return result

def getMarker(sentence, pos_style_dict, neg_style_dict, sentiment, gamma = 2):
    res = []
    result = []
    words = sentence.split()
    if sentiment == "positive":
        style_count = pos_style_dict
    elif sentiment == "negative":
        style_count = neg_style_dict
    for n in range(1, 5):
        state = False
        for l in range(0, len(words)-n+1):
            tmp = ' '.join(words[l:l+n])
            if style_count[tmp] > gamma and (all(i in res for i in subset(words[l:l+n])) or all(i not in res for i in subset(words[l:l+n])) or n == 1):
                print('word:'+tmp+'score:'+str(style_count[tmp]))
                res.append(words[l:l+n])
                result.append(tmp)
                for j in subset(words[l:l+n]):
                    temp = ' '.join(j)
                    if temp in result:
                        result.remove(temp)
                state = True
        if not state:
            break
    return result

f = open("../Data/yelp/sentiment.train.1")
dat = f.readlines()
# pos_dict = file_get_dict("../Data/yelp/sentiment.train.1")
# neg_dict = file_get_dict("../Data/yelp/sentiment.train.0")
# with open('./pos_dict', "wb") as fp:   #Pickling
#     pickle.dump(pos_dict, fp)
# with open('./neg_dict', "wb") as fp:   #Pickling
#     pickle.dump(neg_dict, fp)
with open('./pos_dict', "rb") as fp:   #Pickling
    pos_dict = pickle.load(fp)
with open('./neg_dict', "rb") as fp:   #Pickling
    neg_dict = pickle.load(fp)
pos_style_count = {}
neg_style_count = {}
for i in pos_dict:
    pos_style_count[i] = (pos_dict[i]+1)/(neg_dict.get(i, 0)+1)
for i in neg_dict:
    neg_style_count[i] = (neg_dict[i]+1)/(pos_dict.get(i, 0)+1)


with open('./pos_style_count', "wb") as fp:   #Pickling
    pickle.dump(pos_style_count, fp)
with open('./neg_style_count', "wb") as fp:   #Pickling
    pickle.dump(neg_style_count, fp)

# [[getMarker(data, pos_style_count, neg_style_count, "positive") for data in dat]]


