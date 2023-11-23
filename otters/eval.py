from nltk.translate import meteor
import nltk
import pyrouge
import logging
#from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.cider.cider import Cider
import os
from sacrebleu.metrics import BLEU
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import numpy as np
import spacy
import pytextrank
import string

def rouge(reference, candidate, log_path):
    """
    compute the rouge score
    :param reference: reference
    :param candidate: candidate
    :param log_path: path to log
    :return: rouge-2 score
    """
    # check if of equal amount.
    assert len(reference) == len(candidate)
    # directory for saving sentences
    ref_dir = log_path + 'reference/'
    cand_dir = log_path + 'candidate/'
    # check if there are directories for reference and candidate
    if not os.path.exists(ref_dir):
        os.mkdir(ref_dir)
    if not os.path.exists(cand_dir):
        os.mkdir(cand_dir)

    # write files
    for i in range(len(reference)):
        with open(ref_dir+"%06d_reference.txt" % i, 'w', encoding='utf-8') as f:
            f.write(reference[i] + '\n')
        with open(cand_dir+"%06d_candidate.txt" % i, 'w', encoding='utf-8') as f:
            f.write(candidate[i] + '\n')

    # use pyrouge and ROUGE155
    r = pyrouge.Rouge155(log_level=logging.CRITICAL)
    r.model_filename_pattern = '#ID#_reference.txt'
    r.system_filename_pattern = '(\d+)_candidate.txt'
    r.model_dir = ref_dir
    r.system_dir = cand_dir
    # compute the scores
    rouge_results = r.convert_and_evaluate()
    scores = r.output_to_dict(rouge_results)
    # recall
    recall = [round(scores["rouge_1_recall"] * 100, 2),
              round(scores["rouge_2_recall"] * 100, 2),
              round(scores["rouge_l_recall"] * 100, 2)]
    # precision
    precision = [round(scores["rouge_1_precision"] * 100, 2),
                 round(scores["rouge_2_precision"] * 100, 2),
                 round(scores["rouge_l_precision"] * 100, 2)]
    # f score
    f_score = [round(scores["rouge_1_f_score"] * 100, 2),
               round(scores["rouge_2_f_score"] * 100, 2),
               round(scores["rouge_l_f_score"] * 100, 2)]

    return f_score[:], recall[:], precision[:]

def my_lcs(string, sub):
    """
    Calculates longest common subsequence for a pair of tokenized strings
    :param string : list of str : tokens from a string split using whitespace
    :param sub : list of str : shorter string, also split using whitespace
    :returns: length (list of int): length of the longest common subsequence between the two strings

    Note: my_lcs only gives length of the longest common subsequence, not the actual LCS
    """
    if(len(string)< len(sub)):
        sub, string = string, sub

    lengths = [[0 for i in range(0,len(sub)+1)] for j in range(0,len(string)+1)]

    for j in range(1,len(sub)+1):
        for i in range(1,len(string)+1):
            if(string[i-1] == sub[j-1]):
                lengths[i][j] = lengths[i-1][j-1] + 1
            else:
                lengths[i][j] = max(lengths[i-1][j] , lengths[i][j-1])

    return lengths[len(string)][len(sub)]

class Rouge():
    '''
    Class for computing ROUGE-L score for a set of candidate sentences for the MS COCO test set

    '''
    def __init__(self):
        # vrama91: updated the value below based on discussion with Hovey
        self.beta = 1.2

    def calc_score(self, candidate, refs):
        """
        Compute ROUGE-L score given one candidate and references for an image
        :param candidate: str : candidate sentence to be evaluated
        :param refs: list of str : COCO reference sentences for the particular image to be evaluated
        :returns score: int (ROUGE-L score for the candidate evaluated against references)
        """
        assert(len(candidate)==1)	
        assert(len(refs)>0)         
        prec = []
        rec = []

        # split into tokens
        token_c = candidate[0].split(" ")
    	
        for reference in refs:
            # split into tokens
            token_r = reference.split(" ")
            # compute the longest common subsequence
            lcs = my_lcs(token_r, token_c)
            prec.append(lcs/float(len(token_c)))
            rec.append(lcs/float(len(token_r)))

        prec_max = max(prec)
        rec_max = max(rec)

        if(prec_max!=0 and rec_max !=0):
            score = ((1 + self.beta**2)*prec_max*rec_max)/float(rec_max + self.beta**2*prec_max)
        else:
            score = 0.0
        return score

    def compute_score(self, gts, res):
        """
        Computes Rouge-L score given a set of reference and candidate sentences for the dataset
        Invoked by evaluate_captions.py 
        :param hypo_for_image: dict : candidate / test sentences with "image name" key and "tokenized sentences" as values 
        :param ref_for_image: dict : reference MS-COCO sentences with "image name" key and "tokenized sentences" as values
        :returns: average_score: float (mean ROUGE-L score computed by averaging scores for all the images)
        """
        assert(gts.keys() == res.keys())
        imgIds = gts.keys()

        score = []
        for id in imgIds:
            hypo = res[id]
            ref  = gts[id]

            score.append(self.calc_score(hypo, ref))

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) > 0)

        average_score = np.mean(np.array(score))
        return average_score, np.array(score)

    def method(self):
        return "Rouge"


def compute_cider_score(ref_caps, gen_caps):
    cider_scorer = Cider()
    score, _ = cider_scorer.compute_score(ref_caps, gen_caps)
    return score


def tokenize(sent):
    words = nltk.word_tokenize(sent)

    words=[word.lower() for word in words if word not in string.punctuation]
    return ' '.join(words)

def nlg_eval(refs, gens):
    bleu_score = BLEU()
    #rouge_l = rouge(refs, gens, 'tmp/')[0][2]
    rouge_score = Rouge()

    meteor_sum, count = 0, 0
    cider_refs = {}
    cider_gens = {}
    for ref, gen in zip(refs, gens):
        meteor_sum += meteor([x.split(' ') for x in ref], gen.split(' '))
        cider_refs['%s'%str(count)] = ref
        cider_gens['%s'%str(count)] = [gen]
        count += 1
    bleu = bleu_score.corpus_score(gens, refs).score
    cider = compute_cider_score(cider_refs, cider_gens)
    rouge_l, _ = rouge_score.compute_score(cider_refs, cider_gens)

    mtr = meteor_sum / count
    return [bleu, mtr, rouge_l, cider]

def hits_eval(labels, gens, targets):
    porter_stemmer  = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe("textrank", last=True)
    h1 = 0
    h3 = 0
    count = 0
    for label, gen, target in zip(labels, gens, targets):
        doc = nlp(gen)
        #gen_kw1 = [lemmatizer.lemmatize(p.text) for p in doc._.phrases[:1]]
        gen_kw1 = [p.text for p in doc._.phrases[:1]]
        #gen_kw3 = [lemmatizer.lemmatize(p.text) for p in doc._.phrases[:3]]
        gen_kw3 = [p.text for p in doc._.phrases[:3]]
        #label = [porter_stemmer.stem(w) for w in label]
        #label = [lemmatizer.lemmatize(w) for w in label if w not in target]
        label = [w for w in label]
        if len(list(set(label)&set(gen_kw1))) > 0:
            h1 += 1
        if len(list(set(label)&set(gen_kw3))) > 0:
            h3 += 1
        count += 1
    return [float(h1)/count, float(h3)/count]

def evaluate(output_file, target_file):
    targets = []
    with open(target_file, 'r') as fin:
        for line in fin:
            targets.append(eval(line.strip('\n')))
    
    outputs = []
    with open(output_file, 'r') as fin:
        for line in fin:
            outputs.append(eval(line.strip('\n')))
    
    ### Evaluate response generation
    references = []
    zs_resp_gents = []
    fs_resp_gents = []
    zs_gents = []
    fs_gents = []
    zs_pcot_gents = []
    fs_pcot_gents = []
    for target, output in zip(targets, outputs):
        references.append([tokenize(x.lower()) for x in target['response']])
        zs_resp_gents.append(tokenize(output['zs_resp'].lower()))
        if 'The conversation history' in output['fs_resp']:
            fs_resp = output['fs_resp'].split('The conversation history')[0]
        else:
            fs_resp = output['fs_resp']
        fs_resp_gents.append(tokenize(fs_resp.lower()))
        if 'response is' in output['zs']:
            zs_gent = output['zs'].split('response is')[1]
        elif 'response could be' in output['zs']:
            zs_gent = output['zs'].split('response could be')[1]
        else:
            zs_gent = output['zs']
            #print(zs_gent)
        zs_gents.append(tokenize(zs_gent.lower()))
        if 'response is' in output['fs']:
            fs_gent = output['fs'].split('response is')[1]
        elif 'response could be' in output['fs']:
            fs_gent = output['fs'].split('response could be')[1]
        else:
            fs_gent = output['fs']
            #print(fs_gent)
        if 'The conversation history' in fs_gent:
            fs_gent = fs_gent.split('The conversation history')[0]
        fs_gents.append(tokenize(fs_gent.lower()))
        if 'response is' in output['zs-pcot']:
            zs_pcot_gent = output['zs-pcot'].split('response is')[1]
        elif 'response could be' in output['zs-pcot']:
            zs_pcot_gent = output['zs-pcot'].split('response could be')[1]
        else:
            zs_pcot_gent = output['zs-pcot']
            #print(zs_pcot_gent)
        zs_pcot_gents.append(tokenize(zs_pcot_gent.lower()))
        if 'response is' in output['fs-pcot']:
            fs_pcot_gent = output['fs-pcot'].split('response is')[1]
        elif 'response could be' in output['fs-pcot']:
            fs_pcot_gent = output['fs-pcot'].split('response could be')[1]
        else:
            fs_pcot_gent = output['fs-pcot']
            #print(fs_pcot_gent)
        if 'The conversation history' in fs_pcot_gent:
            fs_pcot_gent = fs_pcot_gent.split('The conversation history')[0]
        fs_pcot_gents.append(tokenize(fs_pcot_gent.lower()))

    print("Standard zero-shot: " + str(float(sum(len(x.split(' ')) for x in zs_resp_gents))/len(references)))
    print("Standard few-shot: " + str(float(sum(len(x.split(' ')) for x in fs_resp_gents))/len(references)))
    print("Proactive zero-shot: " + str(float(sum(len(x.split(' ')) for x in zs_gents))/len(references)))
    print("Proactive few-shot: " + str(float(sum(len(x.split(' ')) for x in fs_gents))/len(references)))
    print("ProCot zero-shot: " + str(float(sum(len(x.split(' ')) for x in zs_pcot_gents))/len(references)))
    print("ProCot few-shot: " + str(float(sum(len(x.split(' ')) for x in fs_pcot_gents))/len(references)))

    with open('output.txt', 'w') as outfile:
        for a,b,c,d,e,f,g in zip(zs_resp_gents,fs_resp_gents,zs_gents,fs_gents,zs_pcot_gents,fs_pcot_gents,references):
            outfile.write("%s\n" % str([a,b,c,d,e,f,g]))


    print("Standard zero-shot: " + str(nlg_eval(references, zs_resp_gents)))
    print("Standard few-shot: " + str(nlg_eval(references, fs_resp_gents)))
    print("Proactive zero-shot: " + str(nlg_eval(references, zs_gents)))
    print("Proactive few-shot: " + str(nlg_eval(references, fs_gents)))
    print("ProCot zero-shot: " + str(nlg_eval(references, zs_pcot_gents)))
    print("ProCot few-shot: " + str(nlg_eval(references, fs_pcot_gents)))


    ### Evaluate topic prediction
    labels = [x['next_topics'] for x in targets]
    ts = [x['targets'] for x in targets]
    print("Standard zero-shot: " + str(hits_eval(labels, zs_resp_gents, ts)))
    print("Standard few-shot: " + str(hits_eval(labels, fs_resp_gents, ts)))
    print("Proactive zero-shot: " + str(hits_eval(labels, zs_gents, ts)))
    print("Proactive few-shot: " + str(hits_eval(labels, fs_gents, ts)))
    print("ProCot zero-shot: " + str(hits_eval(labels, zs_pcot_gents, ts)))
    print("ProCot few-shot: " + str(hits_eval(labels, fs_pcot_gents, ts)))


if __name__ == "__main__":
    evaluate('otters-vicuna-13B.txt', '../data/otters-target.txt')
    evaluate('otters-chatgpt.txt', '../data/otters-target.txt')