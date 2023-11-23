import json
def abgcoqa():
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

    with open('abg-coqa-test.txt', 'r') as infile,\
        open('abgcoqa-source.txt', 'w') as outfile1,\
        open('abgcoqa-target.txt', 'w') as outfile2:
        data = json.load(infile)
        for sample in data['data']:
            source = {}
            target = {}
            conv = []
            for turn in sample['history_turns']:
                conv.append("\"user\": \"%s\", \"system\": \"%s\"" % (turn['question'], turn['answer']))
            prompt = "The given document is \"" + sample['story'] + "\" The given conversation history is [" + ', '.join(conv) +'] '
            prompt += "The question is \"" + sample['target_turn']['question'] + "\" Please generate the response: "
            if len(prompt.split(' ')) > 512:
                print(len(prompt.split(' ')))
                prompt = ' '.join(prompt.split(' ')[-512:])
            target['ambiguity'] = sample['ambiguity']
            if target['ambiguity'] == 'ambiguous':
                target['answer'] = [sample['clarification_turn']['question'], sample['clarification_turn_2']['question']]
                source['cqg'] = ' '.join([cqg, prompt])
                source['fs_cqg'] = ' '.join([cqg, cqg_demo, prompt])
                source['zs_pcot_cqg'] = ' '.join([pcot_cqg, prompt])
                source['fs_pcot_cqg'] = ' '.join([pcot_cqg, cqg_demo_cot, prompt])
            else:
                target['answer'] = sample['target_turn']['answer']

            source['zs_standard'] = ' '.join([standard, prompt])
            source['fs_standard'] = ' '.join([standard, standard_demo, prompt])
            source['zs'] = ' '.join([instruct, prompt])
            source['fs'] = ' '.join([instruct, demo, prompt])
            source['zs-pcot'] = ' '.join([pcot, prompt])
            source['fs-pcot'] = ' '.join([pcot, demo_cot, prompt])

            outfile1.write('%s\n' % source)
            outfile2.write('%s\n' % target)




abgcoqa()