from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

import torch
from tqdm import tqdm

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

def eval_topkg(output_file):

    turns = []
    succ = []
    dialogs = []
    with open(output_file, 'r') as fin:
        for line in fin:
            output = eval(line.strip('\n'))
            if "the target topic" in output['dialog'][-1]:
                continue
            if output['succ']:
                succ.append(0)
            else:
                succ.append(1)
                turns.append(output['turns'])
            dialogs.append(output['dialog'])
    coh = get_dialog_similarity(dialogs)
    print(len(succ))
    return [float(sum(succ))/len(succ), float(sum(turns))/len(turns), coh]



if __name__ == "__main__":
    print('zs_easy: ' + str(eval_topkg('zs_easy_target_chatgpt.json')))
    print('zs_hard: ' + str(eval_topkg('zs_hard_target_chatgpt.json')))
    print('fs_easy: ' + str(eval_topkg('fs_easy_target_chatgpt.json')))
    print('fs_hard: ' + str(eval_topkg('fs_hard_target_chatgpt.json')))
    print('zs_pcot_easy: ' + str(eval_topkg('zs-pcot_easy_target_chatgpt.json')))
    print('zs_pcot_hard: ' + str(eval_topkg('zs-pcot_hard_target_chatgpt.json')))

        