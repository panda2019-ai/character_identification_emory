import json
from pydash import flatten

from experiments.latest.tools.mention import PluralMentionNode
from structure import Utterance, Scene, Episode, TokenNode
from util import idutils
import codecs


class SpliceReader:
    def __init__(self):
        self.mid = 0

    # 抽取1个季的数据中的所有mention，同时构建以集为单位的剧集数据集
    def read_season_json(self, json_path):
        season_mentions = []

        with codecs.open(json_path, "rb", "utf-8", "ignore") as fin:
            season_json = json.load(fin)
            episode_jsons = season_json["episodes"]
            # 对每1集数据抽取mentions并添加到season_mentions
            episodes = [self.read_episode_json(episode_json, season_mentions)
                        for episode_json in episode_jsons]

            for i in range(len(episodes) - 1):
                episodes[i + 1]._previous = episodes[i]
                episodes[i]._next = episodes[i + 1]

            self.assign_metadata(episodes)

        return episodes, season_mentions

    # episode_json表示1集的所有数据，season_mentions表示1个季的所有mentions。
    # 该方法从episode_json抽取出所有mentions后，将这些mention添加到season_mentions。
    def read_episode_json(self, episode_json, season_mentions):
        # 剧集id
        episode_id = episode_json["episode_id"]
        # 剧集的集序号
        episode_num = idutils.parse_episode_id(episode_id)[-1]

        scene_jsons = episode_json["scenes"]
        # 读取并存储每个场景的数据到scenes列表，同时抽取出每个场景中的mentions添加到season_mentions
        scenes = [self.read_scene_json(scene_json, season_mentions)
                  for scene_json in scene_jsons]

        # scenes列表中的每个场景都有指向其前一个场景的指针和指向后一个场景的指针
        for i in range(len(scenes) - 1):
            scenes[i + 1]._previous = scenes[i]
            scenes[i]._next = scenes[i + 1]

        return Episode(episode_num, scenes)

    # scene_json表示1个场景的所有数据，season_mentions表示1个季的所有mentions。
    # 该方法从scene_json抽取出所有mentions后，将这些mention添加到season_mentions。
    def read_scene_json(self, scene_json, season_mentions):
        scene_id = scene_json["scene_id"]
        scene_num = idutils.parse_scene_id(scene_id)[-1]

        utterance_jsons = scene_json["utterances"]
        # [(发言信息，发言中的所有mention的信息),...]
        utterance_mention_pairs = [self.read_utterance_json(utterance_json) for utterance_json in utterance_jsons]
        # 1个场景中的所有发言信息
        utterances = [pair[0] for pair in utterance_mention_pairs]
        # 1个场景中的所有mention信息
        scene_mentions = flatten([pair[1] for pair in utterance_mention_pairs])

        # remove any entities which do not have sing. mentions references but have pl. mention references
        # 1个场景中的所有单数mention对应的角色
        sing_labels = set([m.gold_refs[0] for m in scene_mentions if not m.plural])
        for m in scene_mentions:
            if m.plural:  # 如果mention为复数mention
                # if entity does not exist in the singular labels, then replace it with "#other#" label
                m.gold_refs = [gref
                               if gref in sing_labels or gref == "#other#" or gref == "#general#"
                               else "#other#"
                               for gref in m.gold_refs]

                # remove any duplicate labels
                m.gold_refs = list(set(m.gold_refs))

            # 如果训练语料中mention没有对应的角色标记，则为它分配角色标记为"#other#"
            if len(m.gold_refs) == 0:
                m.gold_refs = ["#other#"]
        # 更新1个季的数据的mentions序列列表
        season_mentions.extend(scene_mentions)

        for i in range(len(utterances) - 1):
            utterances[i + 1]._previous = utterances[i]
            utterances[i]._next = utterances[i + 1]

        return Scene(scene_num, utterances)

    # utterance_json表示1个发言的信息
    def read_utterance_json(self, utterance_json):
        # 说话者
        speakers = utterance_json["speakers"]
        # 分词后的词语序列
        word_forms = utterance_json["tokens"]
        # 词性标注序列
        pos_tags = utterance_json["part_of_speech_tags"]
        # 依存关系标记序列
        dep_tags = utterance_json["dependency_tags"]
        # 依存关系标记序列
        dep_heads = utterance_json["dependency_heads"]
        # 命名实体标注序列
        ner_tags = utterance_json["named_entity_tags"]
        # 角色标注序列
        ref_tags = utterance_json["character_entities"]

        # 构建1个utterence的所有TokenNode实例序列，TokenNode存储了1个词语的词形、词性、实体标记等信息
        tokens_all = self.parse_token_nodes(word_forms, pos_tags, dep_tags, dep_heads, ner_tags)
        # 由tokens_all和角色信息ref_tags构建1个utterance中mentions的PluralMentionNode序列
        # 1个PluralMentionNode存储了1个mention的序号、在TokenNode序列中的起始/终止位置、mention对应的角色、
        # 指示mention是否为复数mention。
        utterance_mentions = self.parse_mention_nodes(tokens_all, ref_tags)

        # 返回(Utterance实例, 输入utterance中包含的mentions列表)
        return Utterance(speakers, statements=tokens_all), utterance_mentions

    # 解析1个utterence中的所有词语，构建1个utterence的所有TokenNode实例序列，TokenNode存储了1个词语的词形、词性、实体标记等信息
    def parse_token_nodes(self, word_forms, pos_tags, dep_tags, dep_heads, ner_tags):
        tokens_all = []
        for word_s, pos_s, ner_s in zip(word_forms, pos_tags, ner_tags):
            tokens = []
            for idx, word, pos, ner in zip(range(len(word_s)), word_s, pos_s, ner_s):
                # 构建TokenNode实例，它的信息有，词语在句子中的序号，词形，词性，实体标记
                token = TokenNode(idx, word, pos, ner)
                # 将TokenNode实例添加到列表tokens
                tokens.append(token)
            # 将1个句子的tokens列表添加到1个utterence列表tokens_all中
            tokens_all.append(tokens)
        # 返回该utterence的TokenNode实例序列
        return tokens_all

    # tokens表示TokenNode实例列表，referents表示tokens中各mention对应的角色信息，
    def parse_mention_nodes(self, tokens, referents):
        mentions = []
        for token_s, ref_s in zip(tokens, referents):
            # condensed referent
            for ref_cond in ref_s:
                # ref_cond = [<start index>, <end index>, <label 1>, <label 2>, ...]
                start_idx, end_idx = ref_cond[0], ref_cond[1]
                refs = list(set(ref_cond[2:]))

                if refs == ["Non-Entity"]:
                    continue

                is_plural = True if len(refs) > 1 else False

                # remove general label from plural mentions
                if len(refs) > 1:
                    refs = list(set([ref if ref != "#GENERAL#" else "#OTHER#" for ref in refs]))
                # 初始化1个PluralMentionNode实例，该实例中存储了
                mention = PluralMentionNode(self.mid,  # mention_id
                                            token_s[start_idx:end_idx],  # 构成mention的TokenNode列表
                                            refs,  # 该mention对应的角色列表
                                            plural=is_plural)  # 是否为复数mention也就是这个mention对应多个角色
                # 添加PluralMentionNode实例到mentions
                mentions.append(mention)
        # 返回输入tokens序列中含有的所有实例化后的PluralMentionNode列表
        return mentions

    def assign_metadata(self, episodes):
        for episode in episodes:
            for scene in episode.scenes:
                scene._episode = episode

                for utterance in scene.utterances:
                    utterance._scene = scene

                    for sentence in utterance.statements:
                        for token in sentence:
                            token._episode = episode
                            token._scene = scene
                            token._utterance = utterance


class StateWriter(object):
    def __init__(self):
        self.fout = None

    def open_file(self, file_path):
        self.fout = open(file_path, "w")

    def write_states(self, states):
        self.fout.write("Mention/Gold/System\n\n")

        for s in states:
            self.write_state(s)
        self.fout.close()

    def write_state(self, state):
        for m in state:
            result = "%s - %s / %s\n" % (str(m), str(m.gold_refs), str(m.auto_refs))
            self.fout.write(result)
