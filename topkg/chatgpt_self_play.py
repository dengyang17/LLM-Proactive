import argparse
import nltk
import string
from tqdm import tqdm
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

import openai
import time
import os

from nltk.stem import WordNetLemmatizer

API_KEY = YOUR_KEY

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_dialog_similarity(all_data):
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2',cache_dir='/storage_fast/ydeng/plm')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2',cache_dir='/storage_fast/ydeng/plm').to('cuda')
    model.eval()
    all_cos = []
    with torch.no_grad():
        for dialog in tqdm(all_data):
            # Tokenize input texts
            encoded_input = tokenizer(dialog, padding=True, truncation=True, return_tensors="pt").to('cuda')
            model_output = model(**encoded_input, output_hidden_states=True, return_dict=True)
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            sentence_embeddings = torch.functional.F.normalize(sentence_embeddings, p=2, dim=1)
            dialog_cos = []
            window = 1
            for i in range(window, len(sentence_embeddings), 2):
                cos = torch.cosine_similarity(sentence_embeddings[i], sentence_embeddings[i-window], dim=0).item()
                dialog_cos.append(cos)
            all_cos.extend(dialog_cos)

    return torch.tensor(all_cos).mean()

def tokenize(sent):
    words = nltk.word_tokenize(sent)

    words=[word.lower() for word in words if word not in string.punctuation]
    return words

def query_openai_model(api_key: str, prompt: str, model: str = "gpt-3.5-turbo-0301", max_tokens: int = 128, temperature: float = 0):
    openai.api_key = api_key

    completions = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=temperature,
    )
    output = completions.choices[0].message.content.strip()
    return output

def self_play(instruct, start_utt, target, max_tokens):
    api_key = API_KEY

    lemmatizer = WordNetLemmatizer()
    dialog = [start_utt]
    sys_instruct = instruct + ' The target topic is "%s". </s>' % target
    user_instruct = 'Given the conversational history, generate a response. </s>'

    sys_conv = ['"user": %s' % start_utt]
    user_conv = ['"system": %s' % start_utt]

    flag = True
    count = 0
    while flag and count < 8:
        if count % 2 == 0: 
            prompt = sys_instruct + ' The conversation history is [' + ', '.join(sys_conv) + '] Please reply: '
            max_new_tokens = max_tokens
        else:
            prompt = user_instruct + ' The conversation history is [' + ', '.join(user_conv) + '] Please reply: '
            max_new_tokens = 40


        chatgpt_flag = True
        while chatgpt_flag:
            try:
                output = query_openai_model(api_key, prompt, max_tokens=max_new_tokens)
                chatgpt_flag = False
            except openai.error.OpenAIError as e:
                print("Some error happened here.")
                time.sleep(1)
        #print(output)
        if 'he response is' in output:
            output = output.split('response is')[1]
        elif 'he response could be' in output:
            output = output.split('response could be')[1]
        if 'The conversation history' in output:
            output = output.split('The conversation history')[0]
        if 'target topic' in output:
            print(output)
            sents = nltk.sent_tokenize(output)
            output = ' '.join([sent for sent in sents if 'target topic' not in sent])
            print(output)
        output = output.strip()

        #print(prompt)
        if count % 2 == 0: 
            user_conv.append('"user": %s' % output)
            sys_conv.append('"system": %s' % output)
            #print("System: %s" % output)
        else:
            user_conv.append('"system": %s' % output)
            sys_conv.append('"user": %s' % output)
            #print("User: %s" % output)
        count += 1
        dialog.append(output)

        if lemmatizer.lemmatize(target) in [lemmatizer.lemmatize(x) for x in tokenize(output)]:
            flag = False
    return flag, count, dialog

def main(dialog_file, instruct, method, mode, max_tokens=128):
    instruct = instruct[method]

    output_path = 'output/%s_%s_%s.json' % (method, mode, 'chatgpt')

    existing_outputs = []
    if os.path.exists(output_path):
        with open(output_path, 'r') as fin:
            for line in fin:
                existing_outputs.append(line)


    with open(dialog_file, 'r') as infile,\
        open(output_path, 'w') as outfile:
        turns = []
        succ = []
        dialogs = []
        count = 0
        for line in tqdm(infile):
            if count < len(existing_outputs):
                output = eval(existing_outputs[count].strip('\n'))
                if "target topic" in output['dialog'][-1]:
                    flag, turn, dialog = self_play(instruct, output['dialog'][0], output['target'], max_tokens)
                    output['succ'] = flag
                    output['turns'] = turn + 1
                    output['dialog'] = dialog
                if output['succ']:
                    succ.append(0)
                else:
                    succ.append(1)
                    turns.append(output['turns'])
                dialogs.append(output['dialog'])
                count += 1
                outfile.write('%s\n' % output)
                continue

            sample = eval(line.strip('\n'))
            start_utt = sample['dialog'][0]
            target = sample[mode]
            
            flag, count, dialog = self_play(instruct, start_utt, target, max_tokens)
            if flag:
                succ.append(0)
                #print("Failed!")
            else:
                succ.append(1)
                turns.append(count+1)
                #print("Success at turn %d!" % (count+1))
            dialogs.append(dialog)
            outfile.write('%s\n' % {'dialog':dialog, 'target':target, 'succ':flag, 'turns':count+1})
            
    coh = get_dialog_similarity(dialogs)
    print("coherence: %s" % str(coh))
    print("success: %s" % str(float(sum(succ))/len(succ)))
    print("turns: %s" % str(float(sum(turns))/len(turns)))
    return [float(sum(succ))/len(succ), float(sum(turns))/len(turns), coh]


if __name__ == "__main__":

    with open('prompt.txt', 'r') as infile:
        resp = infile.readline().strip('\n')
        instruct = infile.readline().strip('\n')
        pcot = infile.readline().strip('\n')
        resp_demo = infile.readline().strip('\n')
        demo = infile.readline().strip('\n')
        pcot_demo = infile.readline().strip('\n')
    prompt = {}
    prompt['zs_resp'] = ' '.join([resp])
    prompt['fs_resp'] = ' '.join([resp, resp_demo])
    prompt['zs'] = ' '.join([instruct])
    prompt['fs'] = ' '.join([instruct, demo])
    prompt['zs-pcot'] = ' '.join([pcot])
    prompt['fs-pcot'] = ' '.join([pcot, pcot_demo])

    scores = []
    
    max_new_tokens = 40
    scores.append(main('topkg-test.json', prompt, 'zs_resp', 'hard_target', max_new_tokens))
    scores.append(main('topkg-test.json', prompt, 'zs_resp', 'easy_target', max_new_tokens))
    scores.append(main('topkg-test.json', prompt, 'fs_resp', 'hard_target', max_new_tokens))
    scores.append(main('topkg-test.json', prompt, 'fs_resp', 'easy_target', max_new_tokens))

    max_new_tokens = 80
    scores.append(main('topkg-test.json', prompt, 'zs', 'hard_target', max_new_tokens))
    scores.append(main('topkg-test.json', prompt, 'zs', 'easy_target', max_new_tokens))
    scores.append(main('topkg-test.json', prompt, 'fs', 'hard_target', max_new_tokens))
    scores.append(main('topkg-test.json', prompt, 'fs', 'easy_target', max_new_tokens))
    

    scores.append(main('topkg-test.json', prompt, 'zs-pcot', 'hard_target'))
    scores.append(main('topkg-test.json', prompt, 'zs-pcot', 'easy_target'))
    scores.append(main('topkg-test.json', prompt, 'fs-pcot', 'hard_target'))
    scores.append(main('topkg-test.json', prompt, 'fs-pcot', 'easy_target'))

    print(scores)