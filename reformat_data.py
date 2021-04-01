with open('pre_train_data/LibriSpeech/data.json') as f:
    line = f.readline()

line = line.replace('[', '').replace(']', '').replace('}, ', '}\n')
with open('pre_train_data/LibriSpeech/data2.json', 'w') as f:
    f.write(line)

line = ''.join(line.split('\n')[:1000]).replace('}', '}\n')
with open('pre_train_data/LibriSpeech/data_debug.json', 'w') as f:
    f.write(line)
