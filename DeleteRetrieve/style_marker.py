from scipy import stats
from scipy.special import psi
from multiprocessing import Process
from sklearn import feature_extraction    
from sklearn.feature_extraction.text import TfidfTransformer    
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import cosine
import numpy as np

# Build words dictionary from a corpus
def get_dict(sen_array):
    tmp_dict = {}
    for i in sen_array:
        sens = i.split()
        for n in range(1, 3):
            for l in range(0, len(sens)-n+1):
                if all(t!='.' for t in sens[l:l+n]):
                    tmp = ' '.join(sens[l:l+n])
                    tmp_dict[tmp] = tmp_dict.get(tmp, 0) + 1
                    tmp_dict[n] = tmp_dict.get(n, 0) + 1
    return tmp_dict

# Build words dictionary from a corpus file
def file_get_dict(filename):
    f = open(filename)
    res = get_dict(f.readlines())
    f.close()
    return res

## Bayes marker extractor
def getMarker(sentence, pos_dict, neg_dict, sentiment, max_marker_num=2, max_marker_len = 2, alpha = 1):
    """
    argument:
        sentence: the sentence you want to extract marker from
        pos_dict: the positive words dictionary
        neg_dict: the negative words dictonary
        sentiment: the sentiment of marker
        max_marker_num: maximum number of markers
        max_marker_len: maximum length of markers
        alpha: minimum sentiment score of markers
    return:
        prev_sen: a string of content words sequence without marker, <unk> indicate the blank of markers
        prev_res: list of markers
    """
    words = sentence.split()
    # max_marker_len should not be too long compared with the total lenght
    max_marker_len = min((len(words)-1)//(max_marker_num), max_marker_len)
    if max_marker_len < 2:
        zzr = max_marker_len + 1
    else:
        zzr = 3
    scores = {}
    token = []
    # define target corpus and origin corpus
    if sentiment == "positive":
        style_count = pos_dict
        target_count = neg_dict
    elif sentiment == "negative":
        style_count = neg_dict
        target_count = pos_dict
    # calculate sentiment scores
    for n in range(1, zzr):
        for l in range(0, len(words)-n+1):
            if all(t!='.' for t in words[l:l+n]):
                tmp = ' '.join(words[l:l+n])
                a = style_count.get(tmp, 0)
                total_a = style_count[0]
                b = target_count.get(tmp, 0)
                total_b = style_count[0]
                scores[(l,l+n)] = tmp, (psi(a+1) - psi(total_a-a+1)- psi(b+1) + psi(total_b-b+1))
    for loc in scores:
        if scores[loc][1] > alpha:
            token.append([loc,scores[loc][1]])
    prev = 0
    prev_res = ''
    prev_sen = ''
    total = 1
    if len(token) == 0:
        return sentence, ['']
    # find the markers and generate formated results
    while total < len(token)+1:
        newtoken = sorted(token, key = lambda x:x[1], reverse=True)[:total]
        newtoken = [i[0] for i in newtoken]
        newtoken = sorted(newtoken, key = lambda x:x[0])
        result = [list(newtoken[0])]
        m = len(newtoken)
        for i in range(1,m):
            if newtoken[i][0]>=result[-1][1]:
                result.append(list(newtoken[i]))
            else:
                result[-1][1] = max(result[-1][1], newtoken[i][1])
        res_t = [' '.join(words[i[0]:i[1]]) for i in result]
        pp = 0
        sen = ''
        for interval in result:
            if pp != interval[0]:
                sen += ' '.join(words[pp:interval[0]]) + ' <unk> '
            else:
                sen += '<unk> '
            pp = interval[1]
        sen += ' '.join(words[pp:])
        if (len(res_t) > max_marker_num) or any(len(i.split())>max_marker_len for i in res_t):
            return prev_sen, prev_res
        total += 1
        prev = len(res_t)
        prev_res = res_t
        prev_sen = sen
    return prev_sen, prev_res



# Search the nearest sentence in the target corpus
def retrieve(origin, sentiment, pos_dat, neg_dat):
    """
    argument:
        origin: the original sentence
        sentiment: the target sentiment of corpus you want to search the nearest sentence from
        pos_dat: positive corpus
        neg_dat: negative corpus
    return:
        res: a nearest sentence in the target corpus
    """
    marker = origin[1]
    res = []
    # Find nearest sentences for each separated content of each marker
    for j in range(len(marker)):
        sentence = origin[0]
        count = 0
        words = sentence.split()
        for i in range(len(words)):
            if words[i] == '<unk>' and count != j:
                words[i] = marker[count]
                count += 1
            elif words[i] == '<unk>' and count == j:
                count += 1
        sentence = ' '.join(words)
        if sentiment == "positive":
            origin_corpus = neg_dat
            target_corpus = pos_dat
        elif sentiment == "negative":
            origin_corpus = pos_dat
            target_corpus = neg_dat
        # transform the sentence into tf-idf vector
        corpus = origin_corpus.copy()
        corpus.extend(target_corpus)
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(corpus)
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(X)
        sen_arr = vectorizer.transform([sentence]).toarray()
        sen_tfidf = transformer.transform(sen_arr).toarray()[0]
        # search the sentence with the smallest distance
        ind = np.argmin([cosine(sen_tfidf, tfidf[i,:].toarray()[0]) if np.max(tfidf[i,:].toarray()[0]) != 0 else 1 for i in range(len(origin_corpus), len(corpus))])
        res.append(target_corpus[ind])
    return res



# Full template based model
def templatebased(sentence, pos_dict, neg_dict, pos_dat, neg_dat, sentiment, max_marker_num=2, max_marker_len = 2, alpha = 1):
    """
    argument:
        sentence: the sentence you want to extract marker from
        pos_dict: the positive words dictionary
        neg_dict: the negative words dictonary
        pos_dat: positive corpus
        neg_dat: negative corpus
        sentiment: the sentiment of marker
        max_marker_num: maximum number of markers
        max_marker_len: maximum length of markers
        alpha: minimum sentiment score of markers
    return:
        a nearest sentence in the target corpus
    """
    if sentiment == 'positive':
        target_sen = 'negative'
    elif sentiment == 'negative':
        target_sen = 'positive'
    tmp = getMarker(sentence, pos_dict, neg_dict, sentiment, max_marker_num, max_marker_len, alpha)
    tttt = retrieve(tmp, target_sen, pos_dat, neg_dat)
    origin = tmp[0].split()
    count = 0
    origin = tmp[0].split()
    i = 0
    # reconstruct the sentence
    while i < len(origin):
        if origin[i] == '<unk>':
            start = i - 1
            end = i + 1
            ret_sen = tttt[count].split()
            if start == -1:
                status0 = True
                for j in range(len(ret_sen)):
                    if ret_sen[j] == origin[end]:
                        origin = ret_sen[:j] + origin[end:]
                        i += j
                        status0 = False
                        break
                if status0:
                    addin = getMarker(tttt[count], pos_dict, neg_dict, target_sen, 1)[1][0].split()
                    origin = addin + origin[end:]
                    i = len(addin)
                    status = True
            else:
                status1 = True
                for j in range(len(ret_sen)):
                    if ret_sen[j] == origin[start]:
                        start1 = j
                        status1 = False
                        break
                status2 = True
                if not status1:
                    for k in range(start1, len(ret_sen)):
                        if end >= len(origin) or ret_sen[k] == origin[end]:
                            end1 = k
                            status2 = False
                            break
                if any([status1, status2]):
                    addin = getMarker(tttt[count], pos_dict, neg_dict, target_sen, 1)[1][0].split()
                    origin = origin[:start+1] + addin + origin[end:]
                    i = start + 1 + len(addin)
                else:
                    origin = origin[:start+1] + ret_sen[start1+1: end1] + origin[end:]
                    i = start + end1 - start1
            count += 1
        else:
            i += 1
    return ' '.join(origin)

# multiple processing for generating positive test result
def job_pos(sentences, pos_dict, neg_dict, pos_dat, neg_dat):
    f = open('test_pos_result', 'a')
    for sentence in sentences:
        tmp = templatebased(sentence, pos_dict, neg_dict, pos_dat, neg_dat, 'positive')
        f.write('train: '+ sentence + '\n')
        f.write('pred: ' + tmp + '\n')
        f.write('\n')
    f.close()
    return

# multiple processing for generating negative test result
def job_neg(sentences, pos_dict, neg_dict, pos_dat, neg_dat):
    f = open('test_neg_result', 'a')
    for sentence in sentences:
        tmp = templatebased(sentence, pos_dict, neg_dict, pos_dat, neg_dat, 'negative')
        f.write('train: '+ sentence + '\n')
        f.write('pred: ' + tmp + '\n')
        f.write('\n')
    f.close()
    return

if __name__ == '__main__':
    # read the file to generate corpus and words dictionary
    f = open("sentiment.train.1.txt")
    pos_dat = f.readlines()
    f.close()
    newf = open("sentiment.train.0.txt")
    neg_dat = newf.readlines()
    newf.close()
    pos_dict = file_get_dict("sentiment.train.1.txt")
    neg_dict = file_get_dict("sentiment.train.0.txt")
    pos_dict[0] = sum(pos_dict.values())
    neg_dict[0] = sum(neg_dict.values())
    testf_pos = open('sentiment.test.1.txt')
    test_pos = testf_pos.readlines()
    test_pos = [i.strip() for i in test_pos]
    testf_pos.close()
    testf_neg = open('sentiment.test.0.txt')
    test_neg = testf_neg.readlines()
    test_neg = [i.strip() for i in test_neg]
    testf_neg.close()
    # multiple processing
    p1 = Process(target=job_pos,args=(test_pos[:2], pos_dict, neg_dict, pos_dat, neg_dat))
    p2 = Process(target=job_pos,args=(test_pos[62:125], pos_dict, neg_dict, pos_dat, neg_dat))
    p3 = Process(target=job_pos,args=(test_pos[125:125+62], pos_dict, neg_dict, pos_dat, neg_dat))
    p4 = Process(target=job_pos,args=(test_pos[125+62:250], pos_dict, neg_dict, pos_dat, neg_dat))
    p5 = Process(target=job_pos,args=(test_pos[250:250+62], pos_dict, neg_dict, pos_dat, neg_dat))
    p6 = Process(target=job_pos,args=(test_pos[250+62:375], pos_dict, neg_dict, pos_dat, neg_dat))
    p7 = Process(target=job_pos,args=(test_pos[375:375+62], pos_dict, neg_dict, pos_dat, neg_dat))
    p8 = Process(target=job_pos,args=(test_pos[375+62:500], pos_dict, neg_dict, pos_dat, neg_dat))
    q1 = Process(target=job_neg,args=(test_neg[:2], pos_dict, neg_dict, pos_dat, neg_dat))
    q2 = Process(target=job_neg,args=(test_neg[62:125], pos_dict, neg_dict, pos_dat, neg_dat))
    q3 = Process(target=job_neg,args=(test_neg[125:125+62], pos_dict, neg_dict, pos_dat, neg_dat))
    q4 = Process(target=job_neg,args=(test_neg[125+62:250], pos_dict, neg_dict, pos_dat, neg_dat))
    q5 = Process(target=job_neg,args=(test_neg[250:250+62], pos_dict, neg_dict, pos_dat, neg_dat))
    q6 = Process(target=job_neg,args=(test_neg[250+62:375], pos_dict, neg_dict, pos_dat, neg_dat))
    q7 = Process(target=job_neg,args=(test_neg[375:375+62], pos_dict, neg_dict, pos_dat, neg_dat))
    q8 = Process(target=job_neg,args=(test_neg[375+62:500], pos_dict, neg_dict, pos_dat, neg_dat))
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    p7.start()
    p8.start()
    q1.start()
    q2.start()
    q3.start()
    q4.start()
    q5.start()
    q6.start()
    q7.start()
    q8.start()
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    p7.join()
    p8.join()
    q1.join()
    q2.join()
    q3.join()
    q4.join()
    q5.join()
    q6.join()
    q7.join()
    q8.join()

