import argparse
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from fastchat.model import load_model, add_model_args

import os

@torch.inference_mode()
def infer(args, infile, outfile):
    print(args.model_path)
    model, tokenizer = load_model(
        args.model_path,
        args.device,
        args.num_gpus,
        args.max_gpu_memory,
        args.load_8bit,
        args.cpu_offloading,
        debug=args.debug,
    )

    existing_outputs = []
    if os.path.exists(outfile):
        with open(outfile, 'r') as fin:
            for line in fin:
                existing_outputs.append(line)

    with open(infile, 'r') as fin,\
        open(outfile, 'w') as fout:
        count = 0
        
        for line in fin:
            prompts = eval(line.strip('\n'))
            if count < len(existing_outputs):
                outputs = eval(existing_outputs[count].strip('\n'))
                for key in prompts:
                    if key not in outputs:
                        max_new_tokens=args.max_new_tokens
                        prompt = prompts[key]
                        input_ids = tokenizer([prompt]).input_ids
                        output_ids = model.generate(
                            torch.as_tensor(input_ids).cuda(),
                            max_new_tokens=max_new_tokens,
                            temperature = 0,
                            early_stopping=True
                        )
                        output_ids = output_ids[0][len(input_ids[0]):]
                        output = tokenizer.decode(output_ids, skip_special_tokens=True,
                                                spaces_between_special_tokens=False)
                        outputs[key] = output
                fout.write('%s\n' % outputs)
                print(outputs)
                count += 1
                continue

            outputs = {}
            for key in prompts:
                prompt = prompts[key]
                input_ids = tokenizer([prompt]).input_ids
                max_new_tokens=args.max_new_tokens
                output_ids = model.generate(
                    torch.as_tensor(input_ids).cuda(),
                    max_new_tokens=max_new_tokens,
                    temperature = 0,
                    early_stopping=True
                )
                output_ids = output_ids[0][len(input_ids[0]):]
                output = tokenizer.decode(output_ids, skip_special_tokens=True,
                                        spaces_between_special_tokens=False)
                outputs[key] = output
            
            fout.write('%s\n' % outputs)
            

            #print(prompt)
            print(outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    args.model_path = "your_path_to/vicuna_hf/13B"
    infer(args, 'abgcoqa-source.txt', 'abgcoqa-vicuna-13B.txt')

    

    

    