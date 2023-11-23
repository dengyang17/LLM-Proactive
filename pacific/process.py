import json

def pacific():
    with open('prompt.txt', 'r') as infile:
        standard = infile.readline().strip('\n')
        instruct = infile.readline().strip('\n')
        pcot = infile.readline().strip('\n')
        standard_demo = infile.readline().strip('\n')
        demo = infile.readline().strip('\n')
        demo_cot = infile.readline().strip('\n')
        cqg = infile.readline().strip('\n')
        pcot_cqg = infile.readline().strip('\n')
        cqg_demo = infile.readline().strip('\n')
        cqg_demo_cot = infile.readline().strip('\n')

    with open('validation.json', 'r') as infile,\
        open('pacific-source.txt', 'w') as outfile1,\
        open('pacific-target.txt', 'w') as outfile2:
        data = json.load(infile)
        for sample in data:
            table = sample['table']['table']
            paras = sample['paragraphs']
            dial = sample['questions']

            table_text = []
            for row in table:
                table_text.append("{} : ".format(row[0]) + " | ".join(row[1:]))
            para_text = []
            for para in paras:
                text = para['text'].replace('\n', ' ')
                para_text.append(text)

            conv = []
            for turn in dial:
                source = {}
                target = {}
                prompt = 'The given document is "[Table] ' + ' \n '.join(table_text) + ' [Paragraph] ' + ' '.join(para_text) + '" '
                if len(prompt.split(' ')) > 512:
                    print(len(prompt.split(' ')))
                    prompt = ' '.join(prompt.split(' ')[-512:])
                prompt += 'The conversation history is [' + ', '.join(conv) + '] '
                prompt += 'The question is "' +  turn['question'] + '" Please generate the response: '
                target['answer'] = turn['answer']
                target['ambiguity'] = 'non_ambiguous'
                if turn['req_clari']:
                    target['ambiguity'] = 'ambiguous'
                    source['cqg'] = ' '.join([cqg, prompt])
                    source['fs_cqg'] = ' '.join([cqg, cqg_demo, prompt])
                    source['zs_pcot_cqg'] = ' '.join([pcot_cqg, prompt])
                    source['fs_pcot_cqg'] = ' '.join([pcot_cqg, cqg_demo_cot, prompt])
                
                source['zs_standard'] = ' '.join([standard, prompt])
                source['fs_standard'] = ' '.join([standard, standard_demo, prompt])
                source['zs'] = ' '.join([instruct, prompt])
                source['fs'] = ' '.join([instruct, demo, prompt])
                source['zs-pcot'] = ' '.join([pcot, prompt])
                source['fs-pcot'] = ' '.join([pcot, demo_cot, prompt])

                conv.append('"user": "%s", "system": "%s"' % (turn['question'], turn['answer']))


                outfile1.write('%s\n' % source)
                outfile2.write('%s\n' % target)




pacific()