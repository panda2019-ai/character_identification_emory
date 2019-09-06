# coding:utf-8

import sys
import json

file_name = 'enhanced-jsons/friends_season_04.json'


outfile = open('word_friends.txt', 'wb')

file_name_li = ['enhanced-jsons/friends_season_01.json', 
                'enhanced-jsons/friends_season_02.json', 
                'enhanced-jsons/friends_season_03.json', 
                'enhanced-jsons/friends_season_04.json']

for file_name in file_name_li:
    with open(file_name, encoding='utf-8', errors='ignore') as infile:
        texts = infile.read()
        json_str = json.loads(texts)
        print(json_str['season_id'])
        for episode in json_str['episodes']:
            print(episode['episode_id'])
            for scene in episode['scenes']:
                print(scene['scene_id'])
                for utterance in scene['utterances']:
                    print(utterance['utterance_id'], end=', ', flush=True)
                    new_sen_li = []
                    if utterance['tokens_with_note']:
                        for word_li in utterance['tokens_with_note']:
                            new_word_li = []
                            for word in word_li:
                                out_str = u'%s\n' % word
                                outfile.write(out_str.encode('utf-8', 'ignore'))

outfile.close()
