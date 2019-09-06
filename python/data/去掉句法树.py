# coding:utf-8
"""

"""
import json
import sys

# 读取角色识别英文语料并将英文单词翻译为中文
file_name = sys.argv[1]  # 输入文件名
with open(file_name, encoding='gbk', errors='ignore') as infile:
    texts = infile.read()
    json_str = json.loads(texts)
    print(json_str['season_id'])
    for episode in json_str['episodes']:
        print(episode['episode_id'])
        for scene in episode['scenes']:
            print(scene['scene_id'])
            for utterance in scene['utterances']:
                print(utterance['utterance_id'])
                new_sen_li = []
                utterance['dependency_tags'] = None
                utterance['dependency_heads'] = None

# 输出翻译后的结果文件
outfile_name = u'%s' % file_name.rsplit(u'/')[-1]
with open(outfile_name, 'wb') as outfile:
    out_str = json.dumps(json_str, indent="\t", ensure_ascii=False)
    outfile.write(out_str.encode('utf-8', 'ignore'))
