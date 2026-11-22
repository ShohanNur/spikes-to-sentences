# ============================================================================
# BLOCK 7 · REVISION EXPERIMENTS  (paste as ONE new cell at the END of the
# notebook and run it after Blocks 1-5 have run; Block 5.6 sweep NOT needed)
# ============================================================================
# What it does, all on the SAME N-best lists so every comparison is paired:
#   1. Decodes validation at the OP-B operating point (alpha = 5.5, beam 250,
#      30-best) and scores EVERY non-empty N-best list with the LLM once.
#   2. Compares six rescoring rules (baseline, ungated fusion, one-sided gate,
#      hard two-sided gate, adaptive weight without gate, ours).
#   3. Repeats the comparison with session-disjoint 2-fold calibration: gate
#      thresholds AND lambda_0 are fitted on one half of the sessions and
#      applied to the other half, so no statistic touches the scored data.
#   4. Paired bootstrap (utterance-level and session-level) and an exact sign
#      test for ours vs baseline and ours vs the strongest competitor.
#   5. Latency per utterance for the decoder and for one LLM scoring pass.
#   6. Writes two Kaggle test submissions with thresholds frozen from
#      validation at alpha = 5.5: baseline (lambda_0 = 0) and OP-B.
#   7. Writes revision_results.tex (LaTeX macros read by the paper) and
#      revision_results.json.  Download both from the output folder.
# Expected cost on a 16 GB GPU: ~45 min (LLM pass on all 1,426 lists).
# ============================================================================
import time
import math
import json
import os
import re
import numpy as np

REV_ALPHA = 5.5                 # OP-B n-gram weight
REV_LAMBDA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
REV_RAW_GRID = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
REV_Q_LOW, REV_Q_MARGIN = 1, 75  # gate percentiles (paper values)
REV_N_BOOT = int(os.environ.get("REV_N_BOOT", 10000))
REV_SEED = 20261118
REV_OUT = CONFIG.get('checkpoint_dir', '.') if 'CONFIG' in globals() else '.'


# ----------------------------------------------------------------------------
# Edit distances (own implementation, so the counts behind every bootstrap
# resample are explicit; checked against jiwer below)
# ----------------------------------------------------------------------------
def _lev(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


try:
    import editdistance as _ed          # C implementation, same result as _lev
    _dist = _ed.eval
except ImportError:
    _dist = _lev

_COUNT_CACHE = {}


def _counts(ref, hyp):
    key = (ref, hyp)
    if key not in _COUNT_CACHE:
        r, h = normalize_text(ref), normalize_text(hyp)
        rw, hw = r.split(), h.split()
        _COUNT_CACHE[key] = (_dist(rw, hw), len(rw), _dist(list(r), list(h)), len(r))
    return _COUNT_CACHE[key]


# ----------------------------------------------------------------------------
# Rescoring rules.  Every rule maps (nbest, llm_scores, params) -> index.
# Thresholds are passed explicitly; no global state is read or written.
# ----------------------------------------------------------------------------
def _dec_scores(res):
    return [normalized_ngram_score(h) for h in res]


def _margin(res):
    return top1_margin(res)


def _z_fuse(res, llm, lam):
    if lam == 0 or llm is None:
        return 0
    return int(np.argmax(_standardize(_dec_scores(res)) + lam * _standardize(llm)))


def _raw_fuse(res, llm, w):
    # Classic N-best log-linear interpolation of raw scores (no standardization)
    if w == 0 or llm is None:
        return 0
    return int(np.argmax(np.asarray(_dec_scores(res)) + w * np.asarray(llm)))


def _adaptive_lam(res, lam0, t_m):
    m = _margin(res)
    if not np.isfinite(m) or t_m <= 0:
        return 0.0
    return lam0 * (1.0 - min(m / t_m, 1.0))


RULES = {
    # name: (pick(res, llm, lam, t_low, t_m), uses_llm(res, t_low, t_m), grid)
    'baseline':  (lambda r, s, lam, tl, tm: 0,
                  lambda r, tl, tm: False, [0.0]),
    'ungated_raw': (lambda r, s, w, tl, tm: _raw_fuse(r, s, w),
                    lambda r, tl, tm: True, REV_RAW_GRID),
    'ungated_z': (lambda r, s, lam, tl, tm: _z_fuse(r, s, lam),
                  lambda r, tl, tm: True, REV_LAMBDA_GRID),
    'one_sided': (lambda r, s, lam, tl, tm:
                  0 if _dec_scores(r)[0] < tl else _z_fuse(r, s, lam),
                  lambda r, tl, tm: _dec_scores(r)[0] >= tl, REV_LAMBDA_GRID),
    'hard_two_sided': (lambda r, s, lam, tl, tm:
                       0 if (_dec_scores(r)[0] < tl or _margin(r) > tm) else _z_fuse(r, s, lam),
                       lambda r, tl, tm: not (_dec_scores(r)[0] < tl or _margin(r) > tm),
                       REV_LAMBDA_GRID),
    'adaptive_nogate': (lambda r, s, lam, tl, tm: _z_fuse(r, s, _adaptive_lam(r, lam, tm)),
                        lambda r, tl, tm: _adaptive_lam(r, 1.0, tm) > 0, REV_LAMBDA_GRID),
    'ours':      (lambda r, s, lam, tl, tm:
                  0 if (_dec_scores(r)[0] < tl or _margin(r) > tm)
                  else _z_fuse(r, s, _adaptive_lam(r, lam, tm)),
                  lambda r, tl, tm: not (_dec_scores(r)[0] < tl or _margin(r) > tm)
                  and _adaptive_lam(r, 1.0, tm) > 0,
                  REV_LAMBDA_GRID),
}


def _calibrate(results):
    top1 = [normalized_ngram_score(r[0]) for r in results if r]
    ms = [m for m in (_margin(r) for r in results if r) if np.isfinite(m)]
    return float(np.percentile(top1, REV_Q_LOW)), float(np.percentile(ms, REV_Q_MARGIN))


def _apply(rule, idxs, lam, tl, tm):
    pick, uses, _ = RULES[rule]
    out, calls = {}, 0
    for i in idxs:
        res, hyps, llm = REV_RES[i], REV_HYPS[i], REV_LLM[i]
        if not res or not any(hyps):
            out[i] = ""
            continue
        if lam != 0 and uses(res, tl, tm):
            calls += 1
        out[i] = hyps[pick(res, llm, lam, tl, tm)]
    return out, calls


def _corpus(idxs, preds):
    ew = nw = ec = nc = 0
    for i in idxs:
        a, b, c, d = _counts(REV_REFS[i], preds[i])
        ew += a; nw += b; ec += c; nc += d
    return ew / max(nw, 1), ec / max(nc, 1)


def _tune(rule, idxs, tl, tm):
    best = None
    for lam in RULES[rule][2]:
        preds, _ = _apply(rule, idxs, lam, tl, tm)
        w, c = _corpus(idxs, preds)
        if best is None or (w, c) < (best[1], best[2]):
            best = (lam, w, c)
    return best[0]


# ----------------------------------------------------------------------------
# 1. Decode validation at OP-B and score every list once with the LLM
# ----------------------------------------------------------------------------
print(f"[rev] building decoder at alpha={REV_ALPHA} ...")
rev_decoder = build_decoder(lm_weight=REV_ALPHA)
t0 = time.time()
REV_RES = []
for i in range(0, len(val_emissions), DECODE_BATCH):
    REV_RES.extend(rev_decoder(val_emissions[i:i + DECODE_BATCH],
                               val_lengths[i:i + DECODE_BATCH]))
t_decode = (time.time() - t0) / max(len(REV_RES), 1)

REV_REFS = list(val_refs)
REV_DAYS = list(val_days)
REV_HYPS = [[" ".join(h.words) if h.words else "" for h in r] if r else [""] for r in REV_RES]
KEEP = [i for i, r in enumerate(REV_REFS) if r and r.strip()]

print(f"[rev] LLM-scoring all {len(KEEP)} lists (no gate) ...")
# LLM scores are saved every 100 lists, so a restarted session resumes here.
import pickle
_LLM_CACHE = os.path.join(REV_OUT, f'rev_llm_scores_alpha{REV_ALPHA}.pkl')
REV_LLM, llm_times = [None] * len(REV_RES), []
if os.path.exists(_LLM_CACHE):
    with open(_LLM_CACHE, 'rb') as fh:
        _c = pickle.load(fh)
    if _c.get('hyps') == REV_HYPS:
        REV_LLM, llm_times = _c['llm'], _c['times']
        print(f"[rev] resumed {sum(x is not None for x in REV_LLM)} cached LLM score lists")
try:
    from tqdm.auto import tqdm as _tq
except ImportError:
    _tq = lambda x, **k: x
for n_done, i in enumerate(_tq(KEEP, desc="[rev] LLM scoring (all lists)")):
    if REV_LLM[i] is None and REV_RES[i] and any(REV_HYPS[i]):
        t1 = time.time()
        REV_LLM[i] = compute_llm_scores(REV_HYPS[i], batch_size=len(REV_HYPS[i]))
        llm_times.append(time.time() - t1)
        if n_done % 100 == 0:
            with open(_LLM_CACHE, 'wb') as fh:
                pickle.dump({'hyps': REV_HYPS, 'llm': REV_LLM, 'times': llm_times}, fh)
with open(_LLM_CACHE, 'wb') as fh:
    pickle.dump({'hyps': REV_HYPS, 'llm': REV_LLM, 'times': llm_times}, fh)
t_llm = float(np.mean(llm_times)) if llm_times else float('nan')
t_llm_med = float(np.median(llm_times)) if llm_times else float('nan')

# Sanity check against the notebook's own metric
_b, _ = _apply('baseline', KEEP, 0.0, -np.inf, np.inf)
_my_w, _my_c = _corpus(KEEP, _b)
_jw = corpus_wer([REV_REFS[i] for i in KEEP], [_b[i] if _b[i].strip() else "<empty>" for i in KEEP])
print(f"[rev] baseline WER own={_my_w*100:.3f}%  jiwer={_jw*100:.3f}%  (paper: 7.37%)")
if abs(_my_w - _jw) >= 2e-3:
    print("[rev] WARNING: own WER differs from jiwer by more than 0.2 points; check normalization")

# ----------------------------------------------------------------------------
# 2. Transductive comparison (as in the paper) and 3. session-disjoint 2-fold
# ----------------------------------------------------------------------------
TL, TM = _calibrate([REV_RES[i] for i in KEEP])
print(f"[rev] thresholds (alpha={REV_ALPHA}): tau_low={TL:.4f}  tau_m={TM:.4f}")

sessions = sorted({REV_DAYS[i] for i in KEEP})
rng_f = np.random.RandomState(REV_SEED)
perm = rng_f.permutation(sessions)
foldA = set(perm[: len(perm) // 2]); foldB = set(perm[len(perm) // 2:])
IA = [i for i in KEEP if REV_DAYS[i] in foldA]
IB = [i for i in KEEP if REV_DAYS[i] in foldB]

table, preds_store = {}, {}
for rule in RULES:
    lam = _tune(rule, KEEP, TL, TM)
    p, calls = _apply(rule, KEEP, lam, TL, TM)
    w, c = _corpus(KEEP, p)
    # 2-fold: fit thresholds and lambda on one half, apply to the other
    pcv, calls_cv = {}, 0
    for fit, ev in ((IA, IB), (IB, IA)):
        tl, tm = _calibrate([REV_RES[i] for i in fit])
        lam_cv = _tune(rule, fit, tl, tm)
        pp, cc = _apply(rule, ev, lam_cv, tl, tm)
        pcv.update(pp); calls_cv += cc
    wcv, ccv = _corpus(KEEP, pcv)
    table[rule] = dict(lam=lam, wer=w, cer=c, llm_frac=calls / len(KEEP),
                       wer_cv=wcv, cer_cv=ccv, llm_frac_cv=calls_cv / len(KEEP),
                       cost=t_llm * calls / len(KEEP))
    preds_store[rule] = (p, pcv)
    print(f"  {rule:16s} lam={lam:<5} WER={w*100:5.2f} CER={c*100:5.2f} "
          f"LLM={calls/len(KEEP)*100:5.1f}%  | 2-fold WER={wcv*100:5.2f} CER={ccv*100:5.2f}")

# ----------------------------------------------------------------------------
# 4. Significance (transductive and 2-fold), paired over the same utterances
# ----------------------------------------------------------------------------
def _per_utt(preds):
    return np.array([_counts(REV_REFS[i], preds[i]) for i in KEEP], dtype=np.float64)


def _paired(pa, pb, seed):
    A, B = _per_utt(pa), _per_utt(pb)          # columns: ew, nw, ec, nc
    d_w = A[:, 0].sum() / A[:, 1].sum() - B[:, 0].sum() / B[:, 1].sum()
    d_c = A[:, 2].sum() / A[:, 3].sum() - B[:, 2].sum() / B[:, 3].sum()
    rng = np.random.RandomState(seed)
    n = len(KEEP)
    bw, bc = np.empty(REV_N_BOOT), np.empty(REV_N_BOOT)
    for k in range(REV_N_BOOT):
        s = rng.randint(0, n, n)
        bw[k] = A[s, 0].sum() / A[s, 1].sum() - B[s, 0].sum() / B[s, 1].sum()
        bc[k] = A[s, 2].sum() / A[s, 3].sum() - B[s, 2].sum() / B[s, 3].sum()
    # session (cluster) bootstrap
    days = np.array([REV_DAYS[i] for i in KEEP])
    uniq = np.unique(days); groups = [np.where(days == u)[0] for u in uniq]
    sw = np.empty(REV_N_BOOT)
    for k in range(REV_N_BOOT):
        s = np.concatenate([groups[g] for g in rng.randint(0, len(groups), len(groups))])
        sw[k] = A[s, 0].sum() / A[s, 1].sum() - B[s, 0].sum() / B[s, 1].sum()
    p_two = lambda x, obs: float(min(1.0, 2 * min((x >= 0).mean(), (x <= 0).mean()))) if obs != 0 else 1.0
    better = int((A[:, 0] < B[:, 0]).sum()); worse = int((A[:, 0] > B[:, 0]).sum())
    from scipy.stats import binomtest
    p_sign = binomtest(better, better + worse, 0.5).pvalue if better + worse else 1.0
    # per-session improvements
    sess_better = sess_worse = 0
    for g in groups:
        da = A[g, 0].sum() / A[g, 1].sum(); db = B[g, 0].sum() / B[g, 1].sum()
        sess_better += da < db; sess_worse += da > db
    return dict(d_wer=d_w, ci_wer=(np.percentile(bw, 2.5), np.percentile(bw, 97.5)),
                p_wer=p_two(bw, d_w), d_cer=d_c,
                ci_cer=(np.percentile(bc, 2.5), np.percentile(bc, 97.5)), p_cer=p_two(bc, d_c),
                ci_wer_sess=(np.percentile(sw, 2.5), np.percentile(sw, 97.5)),
                p_wer_sess=p_two(sw, d_w), utt_better=better, utt_worse=worse,
                p_sign=float(p_sign), sess_better=int(sess_better),
                sess_worse=int(sess_worse), n_sess=len(groups))


competitors = [r for r in RULES if r not in ('baseline', 'ours')]
best_comp = min(competitors, key=lambda r: (table[r]['wer_cv'], table[r]['cer_cv']))
sig = {
    'ours_vs_base': _paired(preds_store['ours'][0], preds_store['baseline'][0], REV_SEED),
    'ours_vs_base_cv': _paired(preds_store['ours'][1], preds_store['baseline'][1], REV_SEED + 1),
    'ours_vs_best_cv': _paired(preds_store['ours'][1], preds_store[best_comp][1], REV_SEED + 2),
}
for k, v in sig.items():
    print(f"[rev] {k}: dWER={v['d_wer']*100:+.3f} CI=[{v['ci_wer'][0]*100:+.3f},"
          f"{v['ci_wer'][1]*100:+.3f}] p={v['p_wer']:.3f} | sessCI=[{v['ci_wer_sess'][0]*100:+.3f},"
          f"{v['ci_wer_sess'][1]*100:+.3f}] | dCER={v['d_cer']*100:+.3f} p={v['p_cer']:.3f} "
          f"| utt +{v['utt_better']}/-{v['utt_worse']} sign p={v['p_sign']:.3f} "
          f"| sessions +{v['sess_better']}/-{v['sess_worse']} of {v['n_sess']}")

# ----------------------------------------------------------------------------
# 6. Test submissions with thresholds frozen from validation at alpha = 5.5
# ----------------------------------------------------------------------------
if 'block_load_test_data' in globals() and 'extract_emissions' in globals():
    print("[rev] decoding test at alpha=5.5 with validation-frozen thresholds ...")
    _ts = block_load_test_data(CONFIG['data_dir'], session2idx, feat_mean, feat_std)
    _te, _tlens, _tids = extract_emissions(model, _ts, CONFIG['device'])
    _tres = []
    for i in range(0, len(_te), DECODE_BATCH):
        _tres.extend(rev_decoder(_te[i:i + DECODE_BATCH], _tlens[i:i + DECODE_BATCH]))
    lam_ours = table['ours']['lam']
    base_txt, ours_txt, n_calls = [], [], 0
    print(f"[rev] test: tau_low={TL:.4f} tau_m={TM:.4f} lambda_0={lam_ours} (all from validation)")
    for res in _tq(_tres, desc="[rev] test gated LLM fusion"):
        hyps = [" ".join(h.words) if h.words else "" for h in res] if res else [""]
        base_txt.append(hyps[0])
        if not res or not any(hyps) or lam_ours == 0 or not RULES['ours'][1](res, TL, TM):
            ours_txt.append(hyps[0]); continue
        n_calls += 1
        llm = compute_llm_scores(hyps, batch_size=len(hyps))
        ours_txt.append(hyps[RULES['ours'][0](res, llm, lam_ours, TL, TM)])
    import pandas as pd
    from glob import glob as _glob
    _id_col, _txt_col = 'id', 'text'
    for _pat in [os.path.join(os.path.dirname(CONFIG['data_dir']), '**', 'sample_submission.csv'),
                 '/kaggle/input/**/sample_submission.csv']:
        _m = _glob(_pat, recursive=True)
        if _m:
            _sdf = pd.read_csv(_m[0]); _id_col, _txt_col = _sdf.columns[0], _sdf.columns[1]
            break
    for name, txt in (('submission_rev_baseline.csv', base_txt), ('submission_rev_opb.csv', ours_txt)):
        _df = pd.DataFrame({_id_col: _tids, _txt_col: txt}).sort_values(_id_col).reset_index(drop=True)
        _df[_id_col] = range(len(_df))           # same id convention as Block 6
        _df.to_csv(os.path.join(REV_OUT, name), index=False)
    test_llm_frac = n_calls / max(len(_tres), 1)
    print(f"[rev] wrote submission_rev_baseline.csv and submission_rev_opb.csv "
          f"(LLM called on {test_llm_frac*100:.1f}% of test). Upload both as late submissions.")
else:
    test_llm_frac = float('nan')
    print("[rev] Block 6 functions not defined; skipping test submissions.")

# ----------------------------------------------------------------------------
# 7. Export: JSON + LaTeX macros consumed by the paper
# ----------------------------------------------------------------------------
def _f(x, d=2):
    return f"{x*100:.{d}f}"


names = {'baseline': 'Base', 'ungated_raw': 'UngRaw', 'ungated_z': 'UngZ',
         'one_sided': 'OneSided', 'hard_two_sided': 'HardTwo',
         'adaptive_nogate': 'AdaptNoGate', 'ours': 'Ours'}
L = ["% Auto-generated by revision_experiments.py -- do not edit by hand",
     r"\RvReadytrue"]
for rule, tag in names.items():
    t = table[rule]
    L += [rf"\renewcommand{{\Rv{tag}WER}}{{{_f(t['wer'])}}}",
          rf"\renewcommand{{\Rv{tag}CER}}{{{_f(t['cer'])}}}",
          rf"\renewcommand{{\Rv{tag}WERcv}}{{{_f(t['wer_cv'])}}}",
          rf"\renewcommand{{\Rv{tag}CERcv}}{{{_f(t['cer_cv'])}}}",
          rf"\renewcommand{{\Rv{tag}Calls}}{{{t['llm_frac']*100:.0f}}}",
          rf"\renewcommand{{\Rv{tag}Lam}}{{{t['lam']:g}}}"]
s = sig['ours_vs_base']; scv = sig['ours_vs_base_cv']; sb = sig['ours_vs_best_cv']
L += [rf"\renewcommand{{\RvDWER}}{{{s['d_wer']*100:+.2f}}}",
      rf"\renewcommand{{\RvCIlo}}{{{s['ci_wer'][0]*100:+.2f}}}",
      rf"\renewcommand{{\RvCIhi}}{{{s['ci_wer'][1]*100:+.2f}}}",
      rf"\renewcommand{{\RvPw}}{{{s['p_wer']:.3f}}}",
      rf"\renewcommand{{\RvSessCIlo}}{{{s['ci_wer_sess'][0]*100:+.2f}}}",
      rf"\renewcommand{{\RvSessCIhi}}{{{s['ci_wer_sess'][1]*100:+.2f}}}",
      rf"\renewcommand{{\RvDCER}}{{{s['d_cer']*100:+.2f}}}",
      rf"\renewcommand{{\RvCERCIlo}}{{{s['ci_cer'][0]*100:+.2f}}}",
      rf"\renewcommand{{\RvCERCIhi}}{{{s['ci_cer'][1]*100:+.2f}}}",
      rf"\renewcommand{{\RvPc}}{{{s['p_cer']:.3f}}}",
      rf"\renewcommand{{\RvUttBetter}}{{{s['utt_better']}}}",
      rf"\renewcommand{{\RvUttWorse}}{{{s['utt_worse']}}}",
      rf"\renewcommand{{\RvPsign}}{{{s['p_sign']:.3f}}}",
      rf"\renewcommand{{\RvSessBetter}}{{{s['sess_better']}}}",
      rf"\renewcommand{{\RvSessWorse}}{{{s['sess_worse']}}}",
      rf"\renewcommand{{\RvNsess}}{{{s['n_sess']}}}",
      rf"\renewcommand{{\RvDWERcv}}{{{scv['d_wer']*100:+.2f}}}",
      rf"\renewcommand{{\RvCIloCV}}{{{scv['ci_wer'][0]*100:+.2f}}}",
      rf"\renewcommand{{\RvCIhiCV}}{{{scv['ci_wer'][1]*100:+.2f}}}",
      rf"\renewcommand{{\RvBestComp}}{{{names[best_comp]}}}",
      rf"\renewcommand{{\RvDWERbest}}{{{sb['d_wer']*100:+.2f}}}",
      rf"\renewcommand{{\RvCIloBest}}{{{sb['ci_wer'][0]*100:+.2f}}}",
      rf"\renewcommand{{\RvCIhiBest}}{{{sb['ci_wer'][1]*100:+.2f}}}",
      rf"\renewcommand{{\RvTauLow}}{{{TL:.2f}}}",
      rf"\renewcommand{{\RvTauM}}{{{TM:.2f}}}",
      rf"\renewcommand{{\RvTdec}}{{{t_decode*1000:.0f}}}",
      rf"\renewcommand{{\RvTllm}}{{{t_llm:.2f}}}",
      rf"\renewcommand{{\RvTllmMed}}{{{t_llm_med:.2f}}}",
      rf"\renewcommand{{\RvTestCalls}}{{{test_llm_frac*100:.1f}}}"]
L.append(r"\RvSigtrue" if s['ci_wer'][1] < 0 else r"\RvSigfalse")
L.append(r"\RvCERSigtrue" if s['ci_cer'][1] < 0 else r"\RvCERSigfalse")
L.append(r"\RvOursBesttrue" if table['ours']['wer_cv'] <= min(table[r]['wer_cv'] for r in competitors)
         else r"\RvOursBestfalse")

# Signed numbers get a typographic minus/plus sign in the paper.
_sgn = re.compile(r'(\\renewcommand\{\\Rv\w+\})\{([+-])(\d+(?:\.\d+)?)\}')
L = [_sgn.sub(lambda m: m.group(1) + '{\\ensuremath{' + m.group(2) + '}' + m.group(3) + '}', x) for x in L]
with open(os.path.join(REV_OUT, 'revision_results.tex'), 'w') as fh:
    fh.write("\n".join(L) + "\n")
with open(os.path.join(REV_OUT, 'revision_results.json'), 'w') as fh:
    json.dump({'alpha': REV_ALPHA, 'tau_low': TL, 'tau_m': TM, 'folds': [sorted(map(int, foldA)), sorted(map(int, foldB))],
               'table': table, 'best_competitor_cv': best_comp,
               'significance': {k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()}
                                for k, v in sig.items()},
               'latency_s': {'decode_per_utt': t_decode, 'llm_pass_mean': t_llm, 'llm_pass_median': t_llm_med},
               'test_llm_fraction': test_llm_frac}, fh, indent=2, default=float)
print(f"[rev] wrote revision_results.tex and revision_results.json to {REV_OUT}")
print("\n[rev] ===== SUMMARY (validation, OP-B lists, alpha=5.5) =====")
for rule, tag in names.items():
    t = table[rule]
    print(f"  {tag:12s} lam0={t['lam']:<5g} LLM={t['llm_frac']*100:5.1f}%  WER={t['wer']*100:5.2f}  "
          f"CER={t['cer']*100:5.2f} | session-disjoint WER={t['wer_cv']*100:5.2f} CER={t['cer_cv']*100:5.2f}")
print(f"  ours vs baseline: dWER={s['d_wer']*100:+.2f} CI=[{s['ci_wer'][0]*100:+.2f},{s['ci_wer'][1]*100:+.2f}] "
      f"| dCER={s['d_cer']*100:+.2f} CI=[{s['ci_cer'][0]*100:+.2f},{s['ci_cer'][1]*100:+.2f}]")
print(f"  latency: beam {t_decode*1000:.0f} ms/utt, LLM pass {t_llm:.2f} s mean / {t_llm_med:.2f} s median")
print("\n[rev] NEXT: download revision_results.tex (put it next to the paper's .tex and recompile),")
print("      then submit submission_rev_baseline.csv and submission_rev_opb.csv as late submissions")
print("      and report their private/public scores in Section VIII-E of the paper.")
