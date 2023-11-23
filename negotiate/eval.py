import os
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from bert_score import BERTScorer
import string
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
import pickle

def tokenize(sent):
    words = nltk.word_tokenize(sent)

    words=[word.lower() for word in words if word not in string.punctuation]
    return ' '.join(words)


def nlg_eval(refs, gens):
    bs = []
    sm = SmoothingFunction()
    for ref, gen in zip(refs, gens):
        ref_words = ref[0].split(" ")
        ngram_weights = [0.25] * min(4, len(ref_words))
        bleu_score = sentence_bleu([ref_words], gen.split(" "), weights=ngram_weights, smoothing_function=sm.method3)
        bs.append(bleu_score)
    bleu = np.mean(bs)
    
    bert_scorer  = BERTScorer(lang="en", rescale_with_baseline=True, idf = False)
    
    bert_p, bert_r, bert_f1 = bert_scorer.score(gens, [x[0] for x in refs])
    with open('bert_score.txt', 'w') as outfile:
        for gen, ref, f1 in zip(gens, refs, bert_f1.tolist()):
            outfile.write('%s\t%s\t%s\n' % (gen, str(ref), str(f1)))

    return [bleu, bert_p.mean().item(), bert_r.mean().item(), bert_f1.mean().item()]

def first_item_in_string(string, list_items):
    # Create a list to store the indexes of items found in the string
    indexes = []

    # Loop over each item in the list
    for item in list_items:
        # Check if the item is in the string
        if item in string:
            # Get the index of the item in the string
            index = string.index(item)
            # Add the index and item to the list
            indexes.append((index, item))

    # Sort the list by index
    indexes.sort()

    # Return the first item in the list, if it's not empty
    if indexes:
        return indexes[0][1]
    else:
        return None
    
def evaluate(output_file, target_file, response_file):
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
        if target['dial_act'] in ['accept','reject','offer','quit']:
            continue
        references.append([tokenize(target['response'].lower())])
        zs_resp_gents.append(tokenize(output['zs_resp'].lower()))
        if 'The item description' in output['fs_resp']:
            fs_resp = output['fs_resp'].split('The item description')[0]
        else:
            fs_resp = output['fs_resp']
        fs_resp_gents.append(tokenize(fs_resp.lower()))
        if 'response is' in output['zs']:
            zs_gent = output['zs'].split('response is')[1]
        else:
            zs_gent = 'hello'
            #print(zs_gent)
        zs_gents.append(tokenize(zs_gent.lower()))
        if 'response is' in output['fs']:
            fs_gent = output['fs'].split('response is')[1]
        else:
            fs_gent = 'hello'
            #print(fs_gent)
        if 'The item description' in fs_gent:
            fs_gent = fs_gent.split('The item description')[0]
        fs_gents.append(tokenize(fs_gent.lower()))
        if 'response is' in output['zs_pcot']:
            zs_pcot_gent = output['zs_pcot'].split('response is')[1]
        else:
            zs_pcot_gent = 'hello'
            #print(zs_pcot_gent)
        zs_pcot_gents.append(tokenize(zs_pcot_gent.lower()))
        if 'response is' in output['fs_pcot']:
            fs_pcot_gent = output['fs_pcot'].split('response is')[1]
        else:
            fs_pcot_gent = 'hello'
            #print(fs_pcot_gent)
        if 'The item description' in fs_pcot_gent:
            fs_pcot_gent = fs_pcot_gent.split('The item description')[0]
        fs_pcot_gents.append(tokenize(fs_pcot_gent.lower()))


    print("Standard zero-shot: " + str(float(sum(len(x.split(' ')) for x in zs_resp_gents))/len(references)))
    print("Standard few-shot: " + str(float(sum(len(x.split(' ')) for x in fs_resp_gents))/len(references)))
    print("Proactive zero-shot: " + str(float(sum(len(x.split(' ')) for x in zs_gents))/len(references)))
    print("Proactive few-shot: " + str(float(sum(len(x.split(' ')) for x in fs_gents))/len(references)))
    print("ProCot zero-shot: " + str(float(sum(len(x.split(' ')) for x in zs_pcot_gents))/len(references)))
    print("ProCot few-shot: " + str(float(sum(len(x.split(' ')) for x in fs_pcot_gents))/len(references)))


    with open(response_file, 'w') as outfile:
        for i in range(len(fs_pcot_gents)):
            resp = {#'zs_resp': zs_resp_gents[i],
                    'fs_resp': fs_resp_gents[i],
                    'zs': zs_gents[i],
                    'fs': fs_gents[i],
                    'zs_pcot': zs_pcot_gents[i],
                    'ps_pcot': fs_pcot_gents[i]}
            outfile.write('%s\n' % resp)
        
    act_map = {'inform':"Answer a question", 'agree':"Agree with the proposal", 'intro':"Greetings", 'reject':"Reject the offer", 'counter-price':"Proposing a counter price", 'vague-price':"Using comparatives with existing price", 'inquiry':"Ask a question", 'unknown':"Unknown", 'insist':"Insist on an offer", 'accept':"Accept the offer", 'offer':"Offer the price", 'disagree':"Disagree with a proposal", 'init-price':"Propose the first price", 'quit':"Quit the session"}
    strat_map = {'assertive_count':"Use assertive words", 'third_person_singular':"Use third person singular", 'hedge_count':"Use hedge words", 'neg_sentiment':"Show negative sentiment", 'trade_in':"Trade in", 'liwc_informal':"Talk informally", 'personal_concern':"Show personal concerns", 'friend':"Build rapport as a friend", 'liwc_certainty':"Use certainty words", 'first_person_singular_count':"Use first person singular", 'third_person_plural':"Use third person plural", 'politeness_greet':"Polite greetings", 'number_of_diff_dic_neg':"Use positive words", 'factive_count':"Use factive verbs", 'politeness_gratitude':"Polite gratitude", 'number_of_diff_dic_pos':"Use negative words", 'family':"Build rapport as a family", 'politeness_please':"Polite request", 'pos_sentiment':"Show positive sentiment", 'propose':"Propose a price", 'first_person_plural_count':"Use first person plural"}
    
    with open('negotiate_data.pkl', 'rb') as infile:
        data = pickle.load(infile)
        tmp = data['strategies2colid']
        del(tmp['<start>'])
        del(tmp['agent_id'])
        tmp['first_person_plural_count'] = 0
        strat2id = {}
        for key in tmp:
            strat2id[strat_map[key]] = tmp[key]
        print(strat2id)
        tmp_dict = data['dialacts2id']
        tmp_dict['reject'] = tmp_dict.pop('<reject>')
        tmp_dict['accept'] = tmp_dict.pop('<accept>')
        tmp_dict['offer'] = tmp_dict.pop('<offer>')
        tmp_dict['quit'] = tmp_dict.pop('<quit>')
        del(tmp_dict['<start>'])
        dialact2id = {}
        for key in tmp_dict:
            dialact2id[act_map[key]] = tmp_dict[key]
        print(dialact2id)

    ### Evaluate strategy prediction
    labels = []
    zs_strat = []
    fs_strat = []
    zs = []
    fs = []
    zs_pcot_strat = []
    fs_pcot_strat = []
    for target, output in zip(targets, outputs):
        label = [0] * len(strat2id)
        for s in target['strategies']:
            if 'seller' in s.split('_'):
                s = '_'.join([x for x in s.split('_') if x != 'seller'])
            elif s.startswith('who_'):
                s = s[4:]
            label[strat2id[s]] = 1
        if sum(label) == 0:
            continue
        labels.append(label)

        ### Zero-shot standard
        
        tmp = [0] * len(strat2id)
        for s in strat2id:
            if s in output['zs_strat']:
                tmp[strat2id[s]] = 1
        zs_strat.append(tmp)

        ### Few-shot standard
        
        tmp = [0] * len(strat2id)
        for s in strat2id:
            if s in output['fs_strat']:
                tmp[strat2id[s]] = 1
        fs_strat.append(tmp)
        
        ### Zero-shot proactive
        tmp = [0] * len(strat2id)
        for s in strat2id:
            if "dialogue act is" in output['zs']:
                output['zs'] = output['zs'].split("dialogue act is")[0]
            if s in output['zs']:
                tmp[strat2id[s]] = 1
        zs.append(tmp)

        ### Few-shot proactive
        tmp = [0] * len(strat2id)
        for s in strat2id:
            if "dialogue act is" in output['fs']:
                output['fs'] = output['fs'].split("dialogue act is")[0]
            if s in output['fs']:
                tmp[strat2id[s]] = 1
        fs.append(tmp)

        ### Zero-shot proactive cot
        tmp = [0] * len(strat2id)
        for s in strat2id:
            if "dialogue act is" in output['zs_pcot']:
                output['zs_pcot'] = output['zs_pcot'].split("dialogue act is")[0]
            if s in output['zs_pcot']:
                tmp[strat2id[s]] = 1
        zs_pcot_strat.append(tmp)

        ### Few-shot proactive cot
        tmp = [0] * len(strat2id)
        for s in strat2id:
            if "dialogue act is" in output['fs_pcot']:
                output['fs_pcot'] = output['fs_pcot'].split("dialogue act is")[0]
            if s in output['fs_pcot']:
                tmp[strat2id[s]] = 1
        fs_pcot_strat.append(tmp)

    print("Zero-shot standard strat: ")
    print(f1_score(labels,zs_strat,average='macro'),f1_score(labels,zs_strat,average='micro'),f1_score(labels,zs_strat,average='weighted'),roc_auc_score(labels,zs_strat,average='macro'),roc_auc_score(labels,zs_strat,average='micro'),roc_auc_score(labels,zs_strat,average='weighted'))
    print("Few-shot standard strat: ")
    print(f1_score(labels,fs_strat,average='macro'),f1_score(labels,fs_strat,average='micro'),f1_score(labels,fs_strat,average='weighted'),roc_auc_score(labels,fs_strat,average='macro'),roc_auc_score(labels,fs_strat,average='micro'),roc_auc_score(labels,fs_strat,average='weighted'))
    print("Zero-shot proactive strat: ")
    print(f1_score(labels,zs,average='macro'),f1_score(labels,zs,average='micro'),f1_score(labels,zs,average='weighted'),roc_auc_score(labels,zs,average='macro'),roc_auc_score(labels,zs,average='micro'),roc_auc_score(labels,zs,average='weighted'))
    print("Few-shot proactive strat: ")
    print(f1_score(labels,fs,average='macro'),f1_score(labels,fs,average='micro'),f1_score(labels,fs,average='weighted'),roc_auc_score(labels,fs,average='macro'),roc_auc_score(labels,fs,average='micro'),roc_auc_score(labels,fs,average='weighted'))
    print("Zero-shot pcot strat: ")
    print(f1_score(labels,zs_pcot_strat,average='macro'),f1_score(labels,zs_pcot_strat,average='micro'),f1_score(labels,zs_pcot_strat,average='weighted'),roc_auc_score(labels,zs_pcot_strat,average='macro'),roc_auc_score(labels,zs_pcot_strat,average='micro'),roc_auc_score(labels,zs_pcot_strat,average='weighted'))
    print("Few-shot pcot strat: ")
    print(f1_score(labels,fs_pcot_strat,average='macro'),f1_score(labels,fs_pcot_strat,average='micro'),f1_score(labels,fs_pcot_strat,average='weighted'),roc_auc_score(labels,fs_pcot_strat,average='macro'),roc_auc_score(labels,fs_pcot_strat,average='micro'),roc_auc_score(labels,fs_pcot_strat,average='weighted'))

    ### Evaluate dialogue act prediction

    outputs = []
    with open(output_file, 'r') as fin:
        for line in fin:
            outputs.append(eval(line.strip('\n')))

    labels = []
    zs_act = []
    fs_act = []
    zs = []
    fs = []
    zs_pcot_act = []
    fs_pcot_act = []
    for target, output in zip(targets, outputs):
        label = [0] * len(dialact2id)
        label[dialact2id[target['dial_act']]] = 1
        labels.append(label)
    
    ### Zero-shot standard
        
        tmp = [0] * len(dialact2id)
        output['zs_act'] = output['zs_act'].split(',')[0]
        s = first_item_in_string(output['zs_act'], dialact2id.keys())
        if s:
            tmp[dialact2id[s]] = 1
        else:
            tmp[7] = 1
        assert sum(tmp) == 1
        zs_act.append(tmp)

        ### Few-shot standard
        
        tmp = [0] * len(dialact2id)
        output['fs_act'] = output['fs_act'].split(',')[0]
        s = first_item_in_string(output['fs_act'], dialact2id.keys())
        if s:
            tmp[dialact2id[s]] = 1
        else:
            tmp[7] = 1
        assert sum(tmp) == 1
        fs_act.append(tmp)

        ### Zero-shot proactive
        tmp = [0] * len(dialact2id)
        if "dialogue act is" in output['zs']:
            output['zs'] = output['zs'].split("dialogue act is")[1]
        if "response is" in output['zs']:
            output['zs'] = output['zs'].split("response is")[0]
        output['zs'] = output['zs'].split(',')[0]
        s = first_item_in_string(output['zs'], dialact2id.keys())
        if s:
            tmp[dialact2id[s]] = 1
        else:
            tmp[7] = 1
        assert sum(tmp) == 1
        zs.append(tmp)

        ### Few-shot proactive
        tmp = [0] * len(dialact2id)
        if "dialogue act is" in output['fs']:
            output['fs'] = output['fs'].split("dialogue act is")[1]
        if "response is" in output['fs']:
            output['fs'] = output['fs'].split("response is")[0]
        output['fs'] = output['fs'].split(',')[0]
        s = first_item_in_string(output['fs'], dialact2id.keys())
        if s:
            tmp[dialact2id[s]] = 1
        else:
            tmp[7] = 1
        assert sum(tmp) == 1
        fs.append(tmp)

        ### Zero-shot proactive cot
        tmp = [0] * len(dialact2id)
        if "dialogue act is" in output['zs_pcot']:
            output['zs_pcot'] = output['zs_pcot'].split("dialogue act is")[1]
        if "response is" in output['zs_pcot']:
            output['zs_pcot'] = output['zs_pcot'].split("response is")[0]
        output['zs_pcot'] = output['zs_pcot'].split(',')[0]
        s = first_item_in_string(output['zs_pcot'], dialact2id.keys())
        if s:
            tmp[dialact2id[s]] = 1
        else:
            tmp[7] = 1
        assert sum(tmp) == 1
        zs_pcot_act.append(tmp)

        ### Few-shot proactive cot
        tmp = [0] * len(dialact2id)
        if "dialogue act is" in output['fs_pcot']:
            output['fs_pcot'] = output['fs_pcot'].split("dialogue act is")[1]
        if "response is" in output['fs_pcot']:
            output['fs_pcot'] = output['fs_pcot'].split("response is")[0]
        output['fs_pcot'] = output['fs_pcot'].split(',')[0]
        s = first_item_in_string(output['fs_pcot'], dialact2id.keys())
        if s:
            tmp[dialact2id[s]] = 1
        else:
            tmp[7] = 1
        assert sum(tmp) == 1
        fs_pcot_act.append(tmp)

    print("Zero-shot standard act: ")
    print(f1_score(labels,zs_act,average='macro'),f1_score(labels,zs_act,average='micro'),f1_score(labels,zs_act,average='weighted'),roc_auc_score(labels,zs_act,average='macro'),roc_auc_score(labels,zs_act,average='weighted'))
    print("Few-shot standard act: ")
    print(f1_score(labels,fs_act,average='macro'),f1_score(labels,fs_act,average='micro'),f1_score(labels,fs_act,average='weighted'),roc_auc_score(labels,fs_act,average='macro'),roc_auc_score(labels,fs_act,average='weighted'))
    print("Zero-shot proactive act: ")
    print(f1_score(labels,zs,average='macro'),f1_score(labels,zs,average='micro'),f1_score(labels,zs,average='weighted'),roc_auc_score(labels,zs,average='macro'),roc_auc_score(labels,zs,average='weighted'))
    print("Few-shot proactive act: ")
    print(f1_score(labels,fs,average='macro'),f1_score(labels,fs,average='micro'),f1_score(labels,fs,average='weighted'),roc_auc_score(labels,fs,average='macro'),roc_auc_score(labels,fs,average='weighted'))
    print("Zero-shot pcot act: ")
    print(f1_score(labels,zs_pcot_act,average='macro'),f1_score(labels,zs_pcot_act,average='micro'),f1_score(labels,zs_pcot_act,average='weighted'),roc_auc_score(labels,zs_pcot_act,average='macro'),roc_auc_score(labels,zs_pcot_act,average='weighted'))
    print("Few-shot pcot act: ")
    print(f1_score(labels,fs_pcot_act,average='macro'),f1_score(labels,fs_pcot_act,average='micro'),f1_score(labels,fs_pcot_act,average='weighted'),roc_auc_score(labels,fs_pcot_act,average='macro'),roc_auc_score(labels,fs_pcot_act,average='weighted'))


    

if __name__ == "__main__":
    evaluate('negotiate-vicuna-13B.txt', 'negotiate-target.txt', 'response-vicuna-13B.txt')
    evaluate('negotiate-chatgpt.txt', 'negotiate-target.txt', 'response-chatgpt.txt')