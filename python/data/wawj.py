# coding:utf-8
"""
将我爱我家语料转成Emory语料格式
"""

import codecs
import re
import json
from pyhanlp import *


outfile = open('enhanced-jsons_wawj/wawj_season_01.json', 'wb')

# 分句分词、词性标注、命名实体识别
def lexical_analysis(text):
    word_li = []
    pos_li = []
    name_entity_li = []
    character_entities_li = []
    for index, word_item in enumerate(HanLP.segment(text)):
        word = word_item.word
        pos = str(word_item.nature)
        word_li.append(word)
        pos_li.append(pos)
        if pos == u"nr":
            name_entity_li.append("U-PERSON")
        else:
            name_entity_li.append("O")
        if word == "你" or word == "他" or word == "她":
            character_entities_li.append([index, index+1, ""])
    
    return word_li, pos_li, name_entity_li, character_entities_li


with codecs.open('wawjdata.txt', 'rb', 'utf-8', 'ignore') as infile:
    text = infile.read()
    season_dict = dict()
    # 构建季结构
    print("season_id:", "s01")
    season_dict["season_id"] = "s01"
    season_dict["episodes"] = []
    # 构建剧集结构
    episode_li = re.split('\n\n\n', text)
    for episode_id, episode in enumerate(episode_li) :
        if episode_id >= 4:
            break
        episode_dict = dict()
        print('episode_id:', "s01_e%02d" % (episode_id + 1))
        episode_dict["episode_id"] = "s01_e%02d" % (episode_id + 1)
        episode_dict["scenes"] = []
        # 构建场景结构
        scene_li = re.split('\n\n', episode)
        for scene_id, scene in enumerate(scene_li):
            scene_dict = dict()
            print('scene_id:', "s01_e%02d_c%02d" % (episode_id+1, scene_id+1))
            scene_dict["scene_id"] = "s01_e%02d_c%02d" % (episode_id+1, scene_id+1)
            scene_dict["utterances"] = []
            utterance_li = re.split('\n', scene)
            for utterance_id, utterance in enumerate(utterance_li):
                # 构建发言结构
                speakers = utterance.split(u':')[0]
                transcript = u':'.join(utterance.split(u':')[1:])
                speakers = speakers.split(u'\t')[-1]
                utterance_dict = dict()
                print('utterance_id: ', "s01_e%02d_c%02d_u%03d"%(episode_id+1, scene_id+1, utterance_id+1))
                utterance_dict["utterance_id"] = "s01_e%02d_c%02d_u%03d"%(episode_id+1, scene_id+1, utterance_id+1)
                # 说话者
                utterance_dict["speakers"] = [speakers]
                # 说话内容
                utterance_dict["transcript"] = transcript
                # 词法分析
                tokens, part_of_speech_tags, named_entity_tags, character_entities_li = lexical_analysis(transcript)
                # 分句并分词
                utterance_dict["tokens"] = [tokens]
                # 词性标注
                utterance_dict["part_of_speech_tags"] = [part_of_speech_tags]
                # 依存句法标注
                utterance_dict["dependency_tags"] = None
                utterance_dict["dependency_heads"] = None
                # 命名实体标注
                utterance_dict["named_entity_tags"] = [named_entity_tags]
                # 角色标注
                utterance_dict["character_entities"] = [character_entities_li]
                # 添加1个发言数据
                # print(utterance_dict)
                # input()
                scene_dict["utterances"].append(utterance_dict)
            # 添加1个场景数据
            # print(scene_dict)
            # input()
            episode_dict["scenes"].append(scene_dict)
        # 添加1个剧集数据
        # print(episode_dict)
        # input()
        season_dict["episodes"].append(episode_dict)
    # 将1个季的数据输出到文件
    out_str = json.dumps(season_dict, ensure_ascii=False, indent="\t")
    outfile.write(out_str.encode('utf-8', 'ignore'))

    
outfile.close()
