# coding:utf-8
"""
将英文语料中的分词后的单词翻译为中文词语
"""
import json
import sys


# 读取英文-中文对照词典
eng_ch_dict = dict()
with open('translate_result/word_translation.txt', encoding='utf-8', errors='ignore') as infile:
    for line in infile:
        line = line.strip()
        if line:
            eng_word, ch_word = line.split(u'\t')
            eng_ch_dict[eng_word] = ch_word

# 读取翻译后的中文句子
ch_sen_dict = dict()
with open('translate_result/sentence_translation.txt', encoding='utf-8', errors='ignore') as infile:
    for line in infile:
        line = line.strip()
        if line:
            items_li = line.split()
            u_id = items_li[0]
            zh_text = u''.join(items_li[1::])
            ch_sen_dict[u_id] = zh_text

# 读取角色识别英文语料并将英文单词翻译为中文
file_name = sys.argv[1]  # 输入文件名
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
                # 输出语料需要满足的条件
                if utterance['tokens'] and utterance['transcript'] and (utterance['utterance_id'] in ch_sen_dict):
                    # 翻译词语序列
                    new_sen_li = []
                    for word_li in utterance['tokens']:
                        new_word_li = []
                        for word in word_li:
                            if word in eng_ch_dict:
                                chinese_word = eng_ch_dict[word]
                            else:
                                chinese_word = word
                            new_word_li.append(chinese_word)
                        new_sen_li.append(new_word_li)
                    utterance['tokens'] = new_sen_li
                    # 翻译句子
                    utterance['transcript'] = ch_sen_dict[utterance['utterance_id']]
                    # 去掉transcript_with_note
                    utterance['transcript_with_note'] = None
                    # 去掉依存句法分析信息
                    utterance['dependency_tags'] = None
                    utterance['dependency_heads'] = None

# 输出翻译后的结果文件
file_name = sys.argv[1]

outfile_name = u'ch_%s' % file_name.rsplit(u'/')[-1]
with open(outfile_name, 'wb') as outfile:
    out_str = json.dumps(json_str, indent="\t", ensure_ascii=False)
    outfile.write(out_str.encode('utf-8', 'ignore'))
