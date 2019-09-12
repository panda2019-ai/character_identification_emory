# 角色识别

<!-- TOC -->

- [角色识别](#角色识别)
    - [1. 中文语料准备](#1-中文语料准备)
        - [1.1 语料统计](#11-语料统计)
        - [1.2 EmoryNLP对语料标注](#12-emorynlp对语料标注)
        - [1.3 将EmroyNLP语料翻译为中文](#13-将emroynlp语料翻译为中文)
    - [引用文献](#引用文献)
    - [参考文献](#参考文献)
    - [共享任务](#共享任务)

<!-- /TOC -->

角色识别是实体关系识别的一个子任务。它的目标是在多人对话中找到mention的全局的真实角色。
这些mention可以是人名实体（贾志国、和平等），代词（她、他等），或物主代词（他的）等。
下边的例子为情景喜剧《我爱我家》中的一段对话，在这里我们标出了所有mentions。

## 1. 中文语料准备

这里我们借用Emory NLP的角色识别系统Demo语料来构建中文角色识别系统语料。

Emory NLP角色识别系统可以参考：

* Latest release: [v2.0](https://github.com/emorynlp/character-identification/archive/character-identification-2.0.tar.gz).
* [Release notes](https://github.com/emorynlp/character-identification/releases).

### 1.1 语料统计

对于每一季《好友记》数据，1-19集（episodes）用于训练，20-21集用于验证，22集以后用于测试。

| Dataset | Episodes | Scenes | Utterances |  Tokens | Speakers | Mentions | Entities |
|:-------:|---------:|-------:|-----------:|--------:|---------:|---------:|---------:|
| TRN   | 76 | 987   | 18,789 | 262,650 | 265 | 36,385 | 628 |
| DEV   | 8  | 122   | 2142   | 28523   | 48  | 3932   | 102 |
| TST   | 13 | 192   | 3,597  | 50,232  | 91  | 7,050  | 165 |
| Total | 97 | 1,301 | 24,528 | 341,405 | 331 | 47,367 | 781 |

### 1.2 EmoryNLP对语料标注

主要标注的是所有mentions所指代的角色，EmroyNLP通过人工对4个季的好友记数据进行了标记。标注后的语料数据示例如下所示：

```json
{
  "utterance_id": "s01_e01_c01_u039",
  "speakers": ["Ross Geller"],
  "transcript": "I told mom and dad last night, they seemed to take it pretty well.",
  "tokens": [
    ["I", "told", "mom", "and", "dad", "last", "night", ",", "they", "seemed", "to", "take", "it", "pretty", "well", "."]
  ],
  "character_entities": [
    [[0, 1, "Ross Geller"], [2, 3, "Judy Geller"], [4, 5, "Jack Geller"], [8, 9, "Jack Geller", "Judy Geller"]]
  ]
}
```

### 1.3 将EmroyNLP语料翻译为中文

## 引用文献

* [They Exist! Introducing Plural Mentions to Coreference Resolution and Entity Linking](http://aclweb.org/anthology/C18-1003). Ethan Zhou and Jinho D. Choi. In Proceedings of the 27th International Conference on Computational Linguistics, COLING'18, 2018 ([slides](https://www.slideshare.net/jchoi7s/they-exist-introducing-plural-mentions-to-coreference-resolution-and-entity-linking)). 

## 参考文献

* [Robust Coreference Resolution and Entity Linking on Dialogues: Character Identification on TV Show Transcripts](http://www.aclweb.org/anthology/K17-1023), Henry Y. Chen, Ethan Zhou, and Jinho D. Choi. Proceedings of the 21st Conference on Computational Natural Language Learning, CoNLL'17, 2017 ([slides](https://www.slideshare.net/jchoi7s/robust-coreference-resolution-and-entity-linking-on-dialogues-character-identification-on-tv-show-transcripts)).
* [Character Identification on Multiparty Conversation: Identifying Mentions of Characters in TV Shows](http://www.aclweb.org/anthology/W16-3612), Henry Y. Chen and Jinho D. Choi. Proceedings of the 17th Annual SIGdial Meeting on Discourse and Dialogue, SIGDIAL'16, 2016 ([poster](https://www.slideshare.net/jchoi7s/character-identification-on-multiparty-conversation-identifying-mentions-of-characters-in-tv-shows)).

## 共享任务

* [SemEval 2018 Task 4: Character Identification on Multiparty Dialogues](../../../semeval-2018-task4).
