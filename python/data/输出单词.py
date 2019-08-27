# coding:utf-8

import sys
import json

file_name = 'enhanced-jsons/friends_season_04.json'

with open(file_name, encoding='gbk', errors='ignore') as infile:
    texts = infile.read()

outfile_name = file_name.rsplit(u'/')[-1]
outfile_name = outfile_name.rsplit(u'.')[0]
outfile = open('word_%s.txt' % outfile_name, 'wb')

json_str = json.loads(texts)
print(json_str['season_id'])
for episode in json_str['episodes']:
    print(episode['episode_id'])
    for scene in episode['scenes']:
        print(scene['scene_id'])
        for utterance in scene['utterances']:
            print(utterance['utterance_id'])
            new_sen_li = []
            if utterance['tokens_with_note']:
                for word_li in utterance['tokens_with_note']:
                    new_word_li = []
                    for word in word_li:
                        out_str = u'%s\n' % word
                        outfile.write(out_str.encode('gbk', 'ignore'))

outfile.close()
