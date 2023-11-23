from sklearn.metrics import f1_score, precision_score, recall_score
import pyrouge
import nltk
import os
import logging
import string

def cnp_eval(labels, preds):
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)

    auto_scores = [precision, recall, f1]
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    auto_scores.extend([precision, recall, f1])
    return auto_scores


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



def tokenize(sent):
    words = nltk.word_tokenize(sent)

    words=[word.lower() for word in words if word not in string.punctuation]
    return ' '.join(words)

def evaluate(output_file, target_file):
    targets = []
    with open(target_file, 'r') as fin:
        for line in fin:
            targets.append(eval(line.strip('\n')))
    
    outputs = []
    with open(output_file, 'r') as fin:
        for line in fin:
            outputs.append(eval(line.strip('\n')))

    ### Evaluating CNP 
    labels = [1 if x['ambiguity'] == 'ambiguous' else 0 for x in targets]
    zs_preds = []
    fs_preds = []
    zs_pcot_preds = []
    fs_pcot_preds = []
    for output in outputs:
        #print(output)
        ### Zero-shot
        if "the answer is" in output['zs'].lower(): 
            zs_preds.append(0)
        elif "the clarifying question is" in output['zs'].lower():
            zs_preds.append(1)
        else:
            zs_preds.append(0)
            #print('zero_shot error: ' + output['zs'])
        ### Few-shot
        if "the answer is" in output['fs'].lower(): 
            fs_preds.append(0)
        elif "the clarifying question is" in output['fs'].lower():
            fs_preds.append(1)
        else:
            fs_preds.append(0)
            #print('few_shot error: ' + output['fs'])
        ### Zero-shot PCoT
        if "not ambiguous" in output['zs-pcot'].lower(): 
            zs_pcot_preds.append(0)
        elif "is ambiguous" in output['zs-pcot'].lower():
            zs_pcot_preds.append(1)
        else:
            zs_pcot_preds.append(0)
            #print('zero-shot pcot error: ' + output['zs-pcot'])
        ### Few-shot PCoT
        if "not ambiguous" in output['fs-pcot'].lower(): 
            fs_pcot_preds.append(0)
        elif "is ambiguous" in output['fs-pcot'].lower():
            fs_pcot_preds.append(1)
        else:
            fs_pcot_preds.append(0)
            #print('few-shot pcot error: ' + output['fs-pcot'])


    print('zero_shot CNP: ' + str(cnp_eval(labels,zs_preds)))
    print(sum(zs_preds))
    print('few_shot CNP: ' + str(cnp_eval(labels,fs_preds)))
    print(sum(fs_preds))
    print('zero_shot pcot CNP: ' + str(cnp_eval(labels,zs_pcot_preds)))
    print(sum(zs_pcot_preds))
    print('few_shot pcot CNP: ' + str(cnp_eval(labels,fs_pcot_preds)))
    print(sum(fs_pcot_preds))


    ### Evaluating CQG
    references = []
    zs_gents = []
    fs_gents = []
    zs_pcot_gents = []
    fs_pcot_gents = []
    for target, output in zip(targets, outputs):
        if target['ambiguity'] == 'ambiguous':
            references.append([tokenize(x.lower()) for x in target['answer']][0])
            ### Zero-shot
            if "the clarifying question is" in output['cqg'].lower():
                tmp = output['cqg'].lower().replace("the clarifying question is", "")
                zs_gents.append(tokenize(tmp.strip()))
            else:
                zs_gents.append(tokenize(output['cqg'].lower().strip()))
            ### Few-shot
            if 'The given document' in output['fs_cqg']:
                output['fs_cqg'] = output['fs_cqg'].split('The given document')[0]
            if "the clarifying question is" in output['fs_cqg'].lower():
                tmp = output['fs_cqg'].lower().replace("the clarifying question is", "")
                fs_gents.append(tokenize(tmp.strip()))
            else:
                fs_gents.append(tokenize(output['fs_cqg'].lower().strip()))
            ### Zero-shot PCoT
            if "the clarifying question is" in output['zs_pcot_cqg'].lower():
                tmp = output['zs_pcot_cqg'].lower().split("the clarifying question is")
                zs_pcot_gents.append(tokenize(tmp[1].strip()))
            else:
                zs_pcot_gents.append(tokenize(output['zs_pcot_cqg'].lower().strip()))
            ### Few-shot PCoT
            if 'The given document' in output['fs_pcot_cqg']:
                output['fs_pcot_cqg'] = output['fs_pcot_cqg'].split('The given document')[0]
            if "the clarifying question is" in output['fs_pcot_cqg'].lower():
                tmp = output['fs_pcot_cqg'].lower().split("the clarifying question is")
                fs_pcot_gents.append(tokenize(tmp[1].strip()))
            else:
                fs_pcot_gents.append(tokenize(output['fs_pcot_cqg'].lower().strip()))


    print('zero_shot CQG: ' + str(rouge(references, zs_gents, 'tmp/')))
    print('few_shot CQG: ' + str(rouge(references, fs_gents, 'tmp/')))
    print('zero_shot pcot CQG: ' + str(rouge(references, zs_pcot_gents, 'tmp/')))
    print('few_shot pcot CQG: ' + str(rouge(references, fs_pcot_gents, 'tmp/')))


def human_evaluate(output_file, target_file, score_file):
    targets = []
    with open(target_file, 'r') as fin:
        for line in fin:
            targets.append(eval(line.strip('\n')))
    
    outputs = []
    with open(output_file, 'r') as fin:
        for line in fin:
            outputs.append(eval(line.strip('\n')))
    
    existing_scores = []
    if os.path.exists(score_file):
        with open(score_file, 'r') as fin:
            for line in fin:
                existing_scores.append(line)

    with open(score_file, 'w') as fout:
        scores = []
        count = 0
        for target, output in zip(targets, outputs):
            if target['ambiguity'] == 'ambiguous':
                if count < len(existing_scores):
                    fout.write(existing_scores[count])
                    scores.append({'score':eval(existing_scores[count].strip('\n'))['score']})
                    count += 1
                    continue
                score = {}
                score['target'] = target['answer']
                print(target['answer'])
                ### Zero-shot
                if "the clarifying question is" in output['cqg'].lower():
                    tmp = output['cqg'].lower().replace("the clarifying question is", "")
                else:
                    tmp = output['cqg'].lower()
                score['cqg'] = tmp
                print(tmp)
                ### Few-shot
                if 'The given document' in output['fs_cqg']:
                    output['fs_cqg'] = output['fs_cqg'].split('The given document')[0]
                if "the clarifying question is" in output['fs_cqg'].lower():
                    tmp = output['fs_cqg'].lower().replace("the clarifying question is", "")
                else:
                    tmp = output['fs_cqg'].lower()
                score['fs_cqg'] = tmp
                print(tmp)
                ### Zero-shot PCoT
                if "the clarifying question is" in output['zs_pcot_cqg'].lower():
                    tmp = output['zs_pcot_cqg'].lower().split("the clarifying question is")[1]
                else:
                    tmp = output['zs_pcot_cqg'].lower()
                score['zs_pcot_cqg'] = tmp
                print(tmp)
                ### Few-shot PCoT
                if 'The given document' in output['fs_pcot_cqg']:
                    output['fs_pcot_cqg'] = output['fs_pcot_cqg'].split('The given document')[0]
                if "the clarifying question is" in output['fs_pcot_cqg'].lower():
                    tmp = output['fs_pcot_cqg'].lower().split("the clarifying question is")[1]
                else:
                    tmp = output['fs_pcot_cqg'].lower()
                score['fs_pcot_cqg'] = tmp
                print(tmp)

                tmp = input('score:').split(' ')
                score['score'] = [int(x) for x in tmp]
                scores.append(score)

                fout.write('%s\n' % str(score))
    print(scores)
    sum_0 = sum([x['score'][0] for x in scores])
    sum_1 = sum([x['score'][1] for x in scores])
    sum_2 = sum([x['score'][2] for x in scores])
    sum_3 = sum([x['score'][3] for x in scores])
    print(sum_0,sum_1,sum_2,sum_3)
    print(float(sum_0)/len(scores),float(sum_1)/len(scores),float(sum_2)/len(scores),float(sum_3)/len(scores))


if __name__ == "__main__":
    evaluate('pacific-vicuna-13B.txt', 'pacific-target.txt')
    evaluate('pacific-chatgpt.txt', 'pacific-target.txt')