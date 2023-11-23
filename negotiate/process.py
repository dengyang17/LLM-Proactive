import pickle


def negotiate():
    with open('prompt.txt', 'r') as infile:
        resp_inst = infile.readline().strip('\n')
        inst = infile.readline().strip('\n')
        pcot = infile.readline().strip('\n')
        strat_list = infile.readline().strip('\n')
        act_list = infile.readline().strip('\n')
        resp_demo = infile.readline().strip('\n')
        demo = infile.readline().strip('\n')
        pcot_demo = infile.readline().strip('\n')
    
    with open('negotiate_data.pkl','rb') as infile,\
        open('negotiate-source.txt', 'w') as outfile1,\
        open('negotiate-target.txt', 'w') as outfile2:
        data = pickle.load(infile)['test']
        print(len(data))
        for sample in data:
            source = {}
            target = {}
            assert sample['raw_data']['scenario']['kbs'][0]['personal']['Role'] == 'buyer'
            assert sample['raw_data']['scenario']['kbs'][1]['personal']['Role'] == 'seller'
            target_price = sample['raw_data']['scenario']['kbs'][0]['personal']['Target']
            selling_price = sample['raw_data']['scenario']['kbs'][1]['personal']['Target']
            conv = []
            for turn, act in zip(sample['raw_data']['events'], sample['dial_acts'][1:]):
                #print(turn,act)
                if act in ['<accept>','<offer>','<reject>','<quit>']:
                    act = act[1:-1]
                if turn['data'] == None:
                    turn['data'] = turn['action']
                if type(turn['data']) is not str:
                    turn['data'] = 'I offer the price %s.' % turn['data']['price']
                if turn['agent'] == 0:
                    conv.append("'buyer': '%s'" % turn['data'])
                    continue
                target['response'] = turn['data']
                target['dial_act'] = act
                target['strategies'] = turn['strategies']

                prompt = "The item description is '" + sample['scene'] + "'. The target selling price is %s. " % (selling_price)
                prompt += "The conversation history is [" + ', '.join(conv) + "] Please generate the response: "

                source['zs_resp'] = ' '.join([resp_inst, prompt])
                source['fs_resp'] = ' '.join([resp_inst, resp_demo, prompt])
                source['zs'] = ' '.join([inst, strat_list, act_list, prompt])
                source['fs'] = ' '.join([inst, strat_list, act_list, demo, prompt])
                source['zs_pcot'] = ' '.join([pcot, strat_list, act_list, prompt])
                source['fs_pcot'] = ' '.join([pcot, strat_list, act_list, pcot_demo, prompt])



                outfile1.write('%s\n' % source)
                outfile2.write('%s\n' % target)

                conv.append("'seller': '%s'" % turn['data'])


negotiate()