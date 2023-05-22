def otters():
    with open('prompt.txt', 'r') as infile:
        resp = infile.readline().strip('\n')
        instruct = infile.readline().strip('\n')
        pcot = infile.readline().strip('\n')
        resp_demo = infile.readline().strip('\n')
        demo = infile.readline().strip('\n')
        pcot_demo = infile.readline().strip('\n')

    with open('otters-test.json', 'r') as infile,\
        open('otters-source.txt', 'w') as outfile1,\
        open('otters-target.txt', 'w') as outfile2:
        test_ref = {} 
        for line in infile:
            sample = eval(line.strip())
            ref = sample['dialog'][1]
            s_t = ','.join(sample['s_c'] + sample['t_c'])
            if s_t in test_ref:
                test_ref[s_t]['ref'].append(ref)
                test_ref[s_t]['b_c'].extend(sample['b_c'])
            else:
                test_ref[s_t] = {
                    'idx': len(test_ref.keys()),
                    's_c': sample['s_c'],
                    't_c': sample['t_c'],
                    'b_c': sample['b_c'].copy(),
                    'dialog': sample['dialog'],
                    'ref': [ref],
                }

        for s_t in test_ref:
            sample = test_ref[s_t]
            source = {}
            target = {}
            prompt = "The conversation history is " + str(sample['dialog'][:1]) + ". The target topic is %s" % str(sample['t_c']) + ". Please reply:"

            target['next_topics'] = sample['b_c']
            target['response'] = sample['ref']
            target['targets'] = sample['t_c']

            source['zs_resp'] = ' '.join([resp, prompt])
            source['fs_resp'] = ' '.join([resp, resp_demo, prompt])
            source['zs'] = ' '.join([instruct, prompt])
            source['fs'] = ' '.join([instruct, demo, prompt])
            source['zs-pcot'] = ' '.join([pcot, prompt])
            source['fs-pcot'] = ' '.join([pcot, pcot_demo, prompt])

            outfile1.write('%s\n' % source)
            outfile2.write('%s\n' % target)




otters()