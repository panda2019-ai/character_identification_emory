from util import *


def init_vector_map(keys, dim, init='random'):
    def random_vectors(keys):
        d = dict([(k, np.random.rand(dim)) for k in keys])
        d[''] = np.zeros(dim)
        return d

    return {'random': random_vectors}[init](keys)


def index_of(nodes, *args):
    indices = [nodes.index(n) if n in nodes else -1 for n in args]
    return indices if len(indices) > 1 else indices[0]


def contain_all(iterable, *args):
    return all([e in iterable for e in args])


def mention_meta(mention):
    def head_token(mention):
        tids = set([t.id for t in mention.tokens])
        for t in mention.tokens:
            if t.dep_head and t.dep_head not in tids:
                return t
        return mention.tokens[0]

    ht, ft, lt = head_token(mention), mention.tokens[0], mention.tokens[-1]
    e, s, u = ft.parent_episode(), ft.parent_scene(), ft.parent_utterance()
    uid = index_of(s.utterances, u)

    u_ns = sum([st for st in u.statements], [])
    u_stid, u_etid = index_of(u_ns, ft, lt)

    st_idx = st_htid = st_stid = st_etid = -1
    for idx, stat in enumerate(u.statements):
        if contain_all(stat, ht, ft, lt):
            st_idx = idx
            st_htid, st_stid, st_etid = index_of(stat, ht, ft, lt)

    return e, s, u, uid, u_ns, u_stid, u_etid, st_idx, st_htid, ht, st_stid, ft, st_etid, lt


def utterance_span(curr, noff, poff):
    noff = abs(noff)
    l = [None] * noff + [curr] + [None] * poff

    for idx in range(noff, noff+poff):
        l[idx+1] = l[idx].next_utterance() if l[idx] else None
    for idx in range(noff, 0, -1):
        l[idx-1] = l[idx].previous_utterance() if l[idx] else None
    return l


def padded_span(iterable, start, end):
    llen = len(iterable)
    bp = [None] * (abs(start) if start < 0 else 0)
    ap = [None] * (max(0, end - llen))
    return bp + iterable[max(start, 0):min(llen, end)] + ap


def anc_str(t):
    p = t.dep_head
    gp = p.dep_head if p else None

    if not p and not gp:
        return '|'.join([t.word_form, 'root'])
    elif p and not gp:
        d = 'R' if p.id < t.id else 'L'
        return '|'.join([t.word_form, d, p.pos_tag, 'root'])
    else:
        d1, d2 = 'R' if p.id < t.id else 'L', 'R' if gp.id < p.id else 'L'
        return '|'.join([t.word_form, d1, p.pos_tag, d2, gp.pos_tag])


class MentionFeatureExtractor(object):
    def __init__(self, w2v, w2g, spks, poss, ners, deps, ani, ina, spk_dim=5, pos_dim=5, ner_dim=5, dep_dim=5, anc_dim=5):
        self.w2v, self.w2v_d, self.w2g, self.w2g_d = w2v, w2v.get_dimension(), w2g, len(list(w2g.values())[0])
        self.none_wvec, self.none_gvec = np.zeros(self.w2v_d).astype('float32'), np.zeros(self.w2g_d).astype('float32')

        self.ani, self.ina = ani, ina
        self.spk_d, self.pos_d, self.ner_d, self.dep_d, self.anc_d = spk_dim, pos_dim, ner_dim, dep_dim, anc_dim
        self.spk, self.pos, self.ner, self.dep = \
            map(init_vector_map, [spks, poss, ners, deps], [spk_dim, pos_dim, ner_dim, dep_dim])
        self.anc = dict()

    def __getstate__(self):
        return self.w2v_d, self.w2g, self.w2g_d, \
               self.none_wvec, self.none_gvec, \
               self.ani, self.ina, \
               self.spk_d, self.pos_d, self.ner_d, self.dep_d, self.anc_d, \
               self.spk, self.pos, self.ner, self.dep, \
               self.anc

    def __setstate__(self, state):
        self.w2v_d, self.w2g, self.w2g_d, \
            self.none_wvec, self.none_gvec, \
            self.ani, self.ina, \
            self.spk_d, self.pos_d, self.ner_d, self.dep_d, self.anc_d, \
            self.spk, self.pos, self.ner, self.dep, \
            self.anc = state

    def extract_mention(self, mention):
        e, s, u, uid, u_ns, u_stid, u_etid, st_idx, \
        st_htid, ht, st_stid, ft, st_etid, lt = mention_meta(mention)

        # Group 1 embedding (Mention tokens, up to 4 tokens)
        emb_ft1 = self.wvecs(padded_span(mention.tokens, 0, 3), False)

        # Group 2 embedding (+-# token sequence)
        emb_ft2 = []
        emb_ft2.extend(self.wvecs(padded_span(u_ns, u_stid-4, u_stid-0), False))
        emb_ft2.append(self.wvecs(mention.tokens, True))
        emb_ft2.extend(self.wvecs(padded_span(u_ns, u_etid+1, u_etid+4), False))

        # Group 3 embedding (Sentence vector)
        emb_ft3, sts = [], padded_span(u.statements, st_idx-3, st_idx+2)
        emb_ft3.extend([self.wvecs(st, True) for st in sts])

        # Group 4 embedding (Utterance vector)
        emb_ft4, utts = [], utterance_span(u, -3, 1)
        emb_ft4.extend([self.uvec(u) for u in utts])

        # Discrete mention features
        men_fts, utts = [], utterance_span(u, -2, 1)
        men_fts.append(self.gvecs(mention.tokens, True))
        men_fts.append(self.word_animacy(mention.tokens, True))
        men_fts.extend([self.spk_vec(u.speakers if u else None) for u in utts])

        # Result features
        emb_fts = list(map(np.array, [emb_ft1, emb_ft2, emb_ft3, emb_ft4]))
        men_fts = np.concatenate(men_fts)

        return emb_fts, men_fts

    def extract_pairwise(self, m1, m2):
        (e1, s1, u1, uid1, u_ns1, u_stid1, u_etid1, st_idx1, st_htid1, ht1, st_stid1, ft1, st_etid1, lt1), \
        (e2, s2, u2, uid2, u_ns2, u_stid2, u_etid2, st_idx2, st_htid2, ht2, st_stid2, ft2, st_etid2, lt2)  \
            = mention_meta(m1), mention_meta(m2)
        pfts, spk1, spk2 = [], u1.speakers, u2.speakers

        w1, w2 = str(m1), str(m2)
        smatch = float(len(StringUtils.lcs(str(m1), str(m2))))
        r1 = smatch / len(w1) if w1 else 0.0
        r2 = smatch / len(w2) if w2 else 0.0
        pfts.append([r1, r2])
        pfts.append([1.0 if str(ht1) == str(ht2) else 0.0])
        pfts.append([1.0 if len(set(spk1).difference(set(spk2))) == 0 else 0.0])
        pfts.append([m2.id-m1.id, st_idx2-st_idx1])

        return np.concatenate(pfts).astype('float32')

    def wvecs(self, tokens, avg=False):
        if tokens:
            words = [t.word_form if t else '' for t in tokens]
            w2v, w2v_d = self.w2v, self.w2v_d
            vecs = [np.array(w2v[w]) for w in words]
            return np.sum(vecs, axis=0) / len([w for w in words if w]) \
                if avg else vecs
        return np.zeros(self.w2v_d)

    def uvec(self, utterance):
        if utterance:
            tokens = sum(utterance.statements, [])
            return self.wvecs(tokens, True)
        return np.zeros(self.w2v_d)

    def gvecs(self, tokens, avg=False):
        if tokens:
            words = [t.word_form if t else '' for t in tokens]
            w2g, w2g_d = self.w2g, self.w2g_d
            for w in [w for w in words if w not in w2g]:
                w2g[w] = np.random.rand(w2g_d).astype('float32')
            vecs = [w2g[w] if w else self.none_gvec for w in words]
            return np.sum(vecs, axis=0) / len([w for w in words if w]) \
                if avg else vecs
        return np.zeros(self.w2g_d)

    def spk_vec(self, speakers):
        if speakers:
            for speaker in speakers:
                if speaker and speaker not in self.spk:
                    self.spk[speaker] = np.random.rand(self.spk_d).astype('float32')
        else:
            return np.zeros(self.spk_d)

        return np.mean([self.spk[spkr] for spkr in speakers], axis=0).astype('float32') if speakers else np.zeros(self.spk_d)

    def word_animacy(self, tokens, avg=False):
        words = [t.word_form if t else None for t in tokens]
        anis = [1 if w in self.ani else 0 for w in words]
        inas = [1 if w in self.ina else 0 for w in words]
        return np.array(list(map(np.mean, [anis, inas]))).astype('float32') if avg \
            else np.concatenate([anis, inas]).astype('float32')
