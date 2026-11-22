"""Mock of the notebook namespace, used only to test revision_experiments.py
end-to-end before it is pasted into Kaggle.  Synthetic data: 1426 utterances
over 41 sessions, 30-best lists whose scores are mildly informative."""
import re, sys, types, numpy as np
import jiwer

rng = np.random.RandomState(0)
VOCAB = "the a to of and i you it is that was he for on are with as his they be at one have this from".split()


def normalize_text(sentence):
    sentence = re.sub(r'[^a-zA-Z\- \']', '', sentence)
    sentence = sentence.replace('- ', ' ').lower().replace('--', '').replace(" '", "'").strip()
    return ' '.join(w for w in sentence.split() if w)


def corpus_wer(refs, hyps):
    return jiwer.wer([normalize_text(r) for r in refs], [normalize_text(h) for h in hyps])


def corpus_cer(refs, hyps):
    return jiwer.cer([normalize_text(r) for r in refs], [normalize_text(h) for h in hyps])


class Hyp:
    def __init__(self, words, score):
        self.words, self.score = words, score


def normalized_ngram_score(h):
    return h.score / max(len(h.words), 1)


def top1_margin(r):
    if len(r) < 2:
        return float('inf')
    return normalized_ngram_score(r[0]) - normalized_ngram_score(r[1])


def _standardize(v):
    v = np.asarray(v, dtype=np.float64); sd = v.std()
    return np.zeros_like(v) if sd < 1e-8 else (v - v.mean()) / sd


N = 1426
val_refs, val_days, _truth = [], [], []
for i in range(N):
    n = rng.randint(3, 10)
    val_refs.append(" ".join(rng.choice(VOCAB, n)))
    val_days.append(int(rng.randint(0, 41)))
val_emissions = list(range(N))
val_lengths = list(range(N))
DECODE_BATCH = 16


def _make_nbest(i):
    ref = val_refs[i].split()
    out = []
    for k in range(30):
        w = list(ref)
        nerr = 0 if k == 0 and rng.rand() < 0.6 else rng.randint(0, 3)
        for _ in range(nerr):
            w[rng.randint(len(w))] = rng.choice(VOCAB)
        sc = -len(w) * (3 + 0.4 * k + rng.rand())
        out.append(Hyp(w, sc))
    out.sort(key=lambda h: -normalized_ngram_score(h))
    if rng.rand() < 0.01:
        return []
    return out


_cache = {}


def build_decoder(lm_weight=4.5):
    def dec(em, le):
        return [_cache.setdefault(i, _make_nbest(i)) for i in em]
    return dec


def compute_llm_scores(hyps, batch_size=16, prefix=None):
    # LLM is mildly better than chance at preferring the reference
    out = []
    for h in hyps:
        out.append(-rng.rand() - 0.5 * (h not in val_refs))
    return out


CONFIG = {'checkpoint_dir': sys.argv[1] if len(sys.argv) > 1 else '.'}

ns = dict(globals())
code = open(sys.argv[2]).read()
exec(compile(code, 'revision_experiments.py', 'exec'), ns)
