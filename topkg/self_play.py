import argparse
import nltk
import string
from tqdm import tqdm
import torch

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

from fastchat.model import load_model, get_conversation_template, add_model_args

from nltk.stem import WordNetLemmatizer

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_dialog_similarity(all_data):
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2',cache_dir='/storage_fast/ydeng/plm')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2',cache_dir='your_path_to/plm').to('cuda')
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

@torch.inference_mode()
def main(args, dialog_file, instruct, method, mode):
    model, tokenizer = load_model(
        args.model_path,
        args.device,
        args.num_gpus,
        args.max_gpu_memory,
        args.load_8bit,
        args.cpu_offloading,
        debug=args.debug,
    )

    lemmatizer = WordNetLemmatizer()
    instruct = instruct[method]

    output_path = 'output/%s_%s_%s.json' % (method, mode, args.model_path.split('/')[-1])

    with open(dialog_file, 'r') as infile,\
        open(output_path, 'w') as outfile:
        turns = []
        succ = []
        dialogs = []
        for line in tqdm(infile):
            sample = eval(line.strip('\n'))
            start_utt = sample['dialog'][0]
            dialog = [start_utt]
            target = sample[mode]
            sys_instruct = instruct + ' The target topic is "%s". </s>' % target
            user_instruct = 'Given the conversational history, generate a response. </s>'

            sys_conv = ['"user": %s' % start_utt]
            user_conv = ['"system": %s' % start_utt]

            #print('The target topic is "%s".' % target)
            #print('User: %s' % start_utt)

            flag = True
            count = 0
            while flag and count < 8:
                if count % 2 == 0: 
                    prompt = sys_instruct + ' The conversation history is [' + ', '.join(sys_conv) + '] Please reply: '
                    max_new_tokens = args.max_new_tokens
                else:
                    prompt = user_instruct + ' The conversation history is [' + ', '.join(user_conv) + '] Please reply: '
                    max_new_tokens = 40


                input_ids = tokenizer([prompt]).input_ids
                output_ids = model.generate(
                    torch.as_tensor(input_ids).cuda(),
                    temperature=args.temperature,
                    max_new_tokens=max_new_tokens,
                    early_stopping=True
                )
                output_ids = output_ids[0][len(input_ids[0]) :]
                output = tokenizer.decode(
                    output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
                )
                if 'he response is' in output:
                    output = output.split('he response is')[1]
                elif 'he response could be' in output:
                    output = output.split('he response could be')[1]
                if method in ['fs_resp', 'fs', 'fs-pcot']:
                    if 'The conversation history' in output:
                        output = output.split('The conversation history')[0]
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
            
            if flag:
                succ.append(0)
                #print("Failed!")
            else:
                succ.append(1)
                turns.append(count+1)
                #print("Success at turn %d!" % (count+1))
            dialogs.append(dialog)
            outfile.write('%s\n' % {'dialog':dialog, 'target':target})
            
    del(model)
    coh = get_dialog_similarity(dialogs)
    print("coherence: %s" % str(coh))
    print("success: %s" % str(float(sum(succ))/len(succ)))
    print("turns: %s" % str(float(sum(turns))/len(turns)))
    return [float(sum(succ))/len(succ), float(sum(turns))/len(turns), coh]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

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
    args.model_path = "your_path_to/vicuna_hf/13B"
    
    scores.append(main(args, 'topkg-test.json', prompt, 'zs_resp', 'hard_target'))
    scores.append(main(args, 'topkg-test.json', prompt, 'zs_resp', 'easy_target'))
    scores.append(main(args, 'topkg-test.json', prompt, 'fs_resp', 'hard_target'))
    scores.append(main(args, 'topkg-test.json', prompt, 'fs_resp', 'easy_target'))

    args.max_new_tokens = 80
    scores.append(main(args, 'topkg-test.json', prompt, 'zs', 'hard_target'))
    scores.append(main(args, 'topkg-test.json', prompt, 'zs', 'easy_target'))
    scores.append(main(args, 'topkg-test.json', prompt, 'fs', 'hard_target'))
    scores.append(main(args, 'topkg-test.json', prompt, 'fs', 'easy_target'))

    args.max_new_tokens = 128
    scores.append(main(args, 'topkg-test.json', prompt, 'zs-pcot', 'hard_target'))
    scores.append(main(args, 'topkg-test.json', prompt, 'zs-pcot', 'easy_target'))
    scores.append(main(args, 'topkg-test.json', prompt, 'fs-pcot', 'hard_target'))
    scores.append(main(args, 'topkg-test.json', prompt, 'fs-pcot', 'easy_target'))

    print(scores)