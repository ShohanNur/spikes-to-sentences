"""Render the replication walkthrough video (1920x1080, 30 fps, captions burnt in).

Scenes are HTML pages shot with Playwright; the two notebook scenes are tall
pages scrolled frame by frame.  Every number shown comes from the executed
notebooks in the release repository or from the paper.
"""
import html
import json
import os
import re
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
FIG = os.path.join(ROOT, 'fig_src')  # pdflatex figN.tex; pdftoppm -r 400 -png -singlefile figN.pdf figN
OUT = os.path.join(ROOT, 'scenes')
FR = os.path.join(ROOT, 'frames')
W, H, FPS = 1920, 1080, 30
ENC = ['-c:v', 'libx264', '-crf', '18', '-preset', 'veryfast', '-pix_fmt', 'yuv420p', '-r', str(FPS)]
os.makedirs(OUT, exist_ok=True)
os.makedirs(FR, exist_ok=True)

CSS = """
*{box-sizing:border-box;margin:0;padding:0}
html,body{width:1920px;height:1080px;background:#f6f7f9;color:#1f2430;
  font-family:'Inter','DejaVu Sans',sans-serif;-webkit-font-smoothing:antialiased}
.slide{position:relative;width:1920px;height:1080px;padding:96px 140px 120px}
.eyebrow{font-size:26px;letter-spacing:.08em;text-transform:uppercase;color:#3b6ea5;font-weight:600}
h1{font-size:76px;line-height:1.08;font-weight:700;margin-top:18px;letter-spacing:-.01em}
h2{font-size:52px;line-height:1.15;font-weight:700;margin-top:10px;letter-spacing:-.01em}
.sub{font-size:38px;line-height:1.3;color:#4c566a;margin-top:22px;font-weight:500}
p,li{font-size:34px;line-height:1.45;color:#2e3440}
ul{margin-top:34px;padding-left:40px} li{margin:12px 0}
.foot{position:absolute;left:140px;right:140px;bottom:48px;display:flex;justify-content:space-between;
  font-size:22px;color:#7b8394;border-top:2px solid #e3e6eb;padding-top:18px}
.fig{display:block;margin:40px auto 0;max-width:1640px;background:#fff;border-radius:14px;
  padding:26px;box-shadow:0 2px 14px rgba(30,40,60,.08)}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:28px;margin-top:40px}
.card{background:#fff;border-radius:14px;padding:28px 30px;box-shadow:0 2px 14px rgba(30,40,60,.08)}
.card b{display:block;font-size:30px;margin-bottom:8px;color:#1f2430}
.card span{font-size:27px;line-height:1.4;color:#4c566a}
.num{display:inline-block;width:46px;height:46px;border-radius:23px;background:#3b6ea5;color:#fff;
  text-align:center;line-height:46px;font-weight:700;font-size:26px;margin-right:16px}
.steps li{list-style:none;margin:22px 0;font-size:33px}
code,.mono{font-family:'DejaVu Sans Mono',monospace;font-size:27px;background:#e9edf3;border-radius:6px;padding:2px 8px}
table{border-collapse:collapse;margin-top:34px;font-size:29px;background:#fff;border-radius:14px;overflow:hidden;
  box-shadow:0 2px 14px rgba(30,40,60,.08)}
th,td{padding:13px 26px;text-align:right} th:first-child,td:first-child{text-align:left}
thead th{background:#eef2f7;color:#3b4252;font-weight:600}
tbody tr:nth-child(even){background:#fafbfc}
tr.ours td{font-weight:700;background:#fdf1e6}
.pill{display:inline-block;background:#fff;border-radius:12px;padding:16px 22px;margin:10px 14px 0 0;
  font-size:27px;box-shadow:0 2px 10px rgba(30,40,60,.08)}
.pill b{color:#c1662f}
.term{background:#1f2430;color:#e5e9f0;border-radius:14px;padding:30px 36px;font-family:'DejaVu Sans Mono',monospace;
  font-size:28px;line-height:1.6;margin-top:36px}
.term .c{color:#88c0d0}
"""

FOOT = ('<div class="foot"><span>From Spikes to Sentences &middot; ICCIT 2026</span>'
        '<span>{n} / 11</span></div>')


def page(body, n):
    return (f'<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head>'
            f'<body><div class="slide">{body}{FOOT.format(n=n)}</div></body></html>')


def img(name, maxw=1640, maxh=None):
    p = os.path.join(FIG, name + '.png')
    st = f'max-width:{maxw}px;' + (f'max-height:{maxh}px;' if maxh else '')
    return f'<img class="fig" style="{st}" src="file://{p}">'


SLIDES = {}
SLIDES['s01'] = (7, page(
    '<div style="margin-top:110px"><div class="eyebrow">ICCIT 2026 &middot; Replication walkthrough</div>'
    '<h1>From Spikes to Sentences</h1>'
    '<div class="sub">A Hybrid CTC Decoder with Confidence-Gated<br>Language Model Rescoring</div>'
    '<p style="margin-top:70px;font-size:36px;font-weight:600">Anik Biswas*, Shohanur Rahman*, Mehedi Hasan Anjan</p>'
    '<p style="font-size:28px;color:#6b7385;margin-top:8px">BUET &middot; SUST &nbsp;&nbsp; * equal contribution</p></div>', 1))
SLIDES['s02'] = (9, page(
    '<div class="eyebrow">The task</div><h2>Decode attempted speech from 256 electrodes into text</h2><ul>'
    '<li>Brain-to-Text &rsquo;25 benchmark, participant T15 (ALS, four 64-channel arrays in ventral precentral gyrus)</li>'
    '<li>512 features per 20 ms bin: threshold crossings and spike-band power</li>'
    '<li>1,426 validation sentences; 1,450 test sentences scored only by the competition server</li>'
    '<li>Metric: word error rate (WER), plus character and phoneme error rates</li></ul>', 2))
SLIDES['s03'] = (10, page(
    '<div class="eyebrow">The pipeline</div><h2>Three stages, one of them optional per sentence</h2>'
    + img('fig1', 1640) +
    '<div class="cards"><div class="card"><b>1 &middot; Acoustic model</b><span>Neural features to phoneme posteriors (CTC). Greedy PER 12.12&nbsp;%.</span></div>'
    '<div class="card"><b>2 &middot; Beam search</b><span>Lexicon-constrained, 4-gram KenLM, 30-best list per sentence.</span></div>'
    '<div class="card"><b>3 &middot; Gated LLM fusion</b><span>Qwen2.5-7B (4-bit) rescores the list only when the decoder is undecided.</span></div></div>', 3))
SLIDES['s04'] = (8, page(
    '<div class="eyebrow">Stage 1</div><h2>Hybrid CNN&ndash;BiLSTM&ndash;Transformer CTC model</h2>'
    + img('fig2', 1640, 470) +
    '<p style="margin-top:34px">28.8 M parameters, 10.8 M of them in per-session adapters tied by a drift penalty. '
    'Weights of the three best checkpoints are averaged.</p>', 4))
SLIDES['s05'] = (12, page(
    '<div class="eyebrow">Stage 3 &middot; the contribution</div><h2>A two-sided gate and a weight that fades out</h2>'
    + img('fig3', 1640, 430) +
    '<div class="cards" style="grid-template-columns:1fr 1fr"><div class="card"><b>When to skip the LLM</b>'
    '<span>The list has collapsed (top-1 score below the 1st percentile) or is already decided '
    '(top-1/top-2 margin above the 75th percentile). No transcripts needed.</span></div>'
    '<div class="card"><b>How much to trust it</b><span>Inside the gate the weight decays with the margin, '
    '<i>&lambda; = &lambda;<sub>0</sub>(1 &minus; min(m/&tau;<sub>m</sub>, 1))</i>, so it reaches zero exactly at the gate.</span></div></div>', 5))
SLIDES['s06'] = (12, page(
    '<div class="eyebrow">Run it yourself</div><h2>Four steps on Kaggle</h2><ul class="steps" style="padding-left:0;margin-top:30px">'
    '<li><span class="num">1</span>Import <code>notebooks/02_revision_experiments.ipynb</code> from the repository</li>'
    '<li><span class="num">2</span>Settings: accelerator <b>GPU T4</b>, internet <b>on</b></li>'
    '<li><span class="num">3</span>Add inputs: <code>brain-to-text-25</code> (competition),<br>'
    '<span style="display:inline-block;width:62px"></span><code>shohan3125/brain-to-text-checkpoint</code>, '
    '<code>heyyousum/quality-english-dataset-for-ngram-model-v2</code>,<br>'
    '<span style="display:inline-block;width:62px"></span><code>heyyousum/custom-4-gram-wiki-news-switchboard-updated-v3</code></li>'
    '<li><span class="num">4</span><b>Run All</b>: about 2&ndash;2.5 h on a T4; no training (checkpoint is loaded)</li></ul>'
    '<p style="font-size:26px;color:#6b7385;margin-top:26px">Model files are also in the Zenodo archive if a Kaggle dataset is not visible to you.</p>', 6))
SLIDES['s09'] = (13, page(
    '<div class="eyebrow">Results</div><h2>Seven rules on identical 30-best lists (validation)</h2>'
    '<div style="display:flex;gap:46px;align-items:flex-start"><table><thead><tr><th>Rule</th><th>LLM</th><th>WER</th><th>CER</th><th>WER*</th></tr></thead><tbody>'
    '<tr><td>Beam + 4-gram only</td><td>0&nbsp;%</td><td>7.37</td><td>5.33</td><td>7.37</td></tr>'
    '<tr><td>Ungated, raw scores</td><td>100&nbsp;%</td><td>7.33</td><td>5.21</td><td>7.43</td></tr>'
    '<tr><td>Ungated, z-scores</td><td>0&nbsp;%</td><td>7.37</td><td>5.33</td><td>7.52</td></tr>'
    '<tr><td>One-sided gate</td><td>0&nbsp;%</td><td>7.37</td><td>5.33</td><td>7.47</td></tr>'
    '<tr><td>Two-sided, hard</td><td>0&nbsp;%</td><td>7.37</td><td>5.33</td><td>7.47</td></tr>'
    '<tr><td>Adaptive, no gate</td><td>69&nbsp;%</td><td>7.28</td><td>5.19</td><td>7.43</td></tr>'
    '<tr class="ours"><td>Ours</td><td>68&nbsp;%</td><td>7.26</td><td>5.18</td><td>7.39</td></tr></tbody></table>'
    '<div style="margin-top:34px;width:640px"><div class="pill">WER &minus;0.11, 95&nbsp;% CI [&minus;0.34, +0.12]: <b>a tie</b></div>'
    '<div class="pill">CER &minus;0.15, CI [&minus;0.29, &minus;0.01], <b>p = 0.041</b></div>'
    '<div class="pill">*Session-disjoint fit: no rule beats the decoder alone; ours is the best LLM rule</div>'
    '<div class="pill">Held-out test (server): <b>9.34&nbsp;%</b> private, <b>8.21&nbsp;%</b> public</div></div></div>'
    '<p style="font-size:24px;color:#6b7385;margin-top:22px">0&nbsp;% LLM means the rule&rsquo;s best weight switched the LLM off. '
    'Latency on a T4: 67 ms beam search, 1.08 s per LLM pass.</p>', 9))
SLIDES['s10'] = (9, page(
    '<div class="eyebrow">Rebuild the paper</div><h2>Numbers flow from the notebook into the PDF</h2>'
    '<div class="term"><span class="c"># Block 7 writes /kaggle/working/revision_results.tex</span><br>'
    'cd paper<br>cp ~/Downloads/revision_results.tex .<br>pdflatex main.tex &amp;&amp; pdflatex main.tex &amp;&amp; pdflatex main.tex</div>'
    '<p style="margin-top:34px">Every number in Table II and Section VIII-B is a LaTeX macro read from that file, '
    'so the paper and the code cannot drift apart.</p>', 10))
SLIDES['s11'] = (8, page(
    '<div style="margin-top:120px"><div class="eyebrow">Everything shown is public</div>'
    '<h2>Code, executed notebooks, results and paper source</h2>'
    '<ul><li>GitHub repository and Zenodo archive with a DOI: links in Section X of the paper and in the README</li>'
    '<li>The Zenodo record also holds the checkpoint, lexicon and 4-gram model</li>'
    '<li>Outputs in this video are from the executed notebooks in the repository (Kaggle, NVIDIA T4)</li></ul>'
    '<p style="margin-top:60px;font-size:40px;font-weight:600">Thank you</p></div>', 11))


# ---------------------------------------------------------------- notebooks
def nb_cells(path):
    nb = json.load(open(path))
    out = []
    for c in nb['cells']:
        if c['cell_type'] != 'code':
            continue
        src = ''.join(c['source'])
        hdr = next((l.strip('# ').strip() for l in src.splitlines() if 'BLOCK' in l), None)
        if hdr and 'REVISION EXPERIMENTS' in hdr:
            hdr = 'BLOCK 7 · REVISION EXPERIMENTS'
        text = ''
        for o in c.get('outputs', []):
            if 'text' in o:
                text += ''.join(o['text'])
        text = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', text)
        lines = [l for l in text.splitlines()
                 if l.strip() and 'Trie label' not in l and '%|' not in l and 'it/s]' not in l
                 and 's/it]' not in l and '━' not in l and 'Downloading' not in l and 'Collecting' not in l
                 and 'manylinux' not in l and 'HF Hub' not in l and '[?25h' not in l]
        out.append((hdr, lines))
    return out


def nb_page(title, cells, keep, trim):
    blocks = []
    for hdr, lines in cells:
        if not hdr or not any(k in hdr for k in keep):
            continue
        lines = trim(hdr, lines)
        body = html.escape('\n'.join(lines)) if lines else '<span style="color:#9aa1ad">(no output)</span>'
        body = re.sub(r'(\d+\.\d+%)', r'<b style="color:#c1662f">\1</b>', body)
        blocks.append(f'<div class="cell"><div class="hdr"># {html.escape(hdr)}</div><pre>{body}</pre></div>')
    css = """
    *{box-sizing:border-box;margin:0;padding:0}
    body{width:1920px;background:#eef0f3;font-family:'Inter','DejaVu Sans',sans-serif}
    .top{background:#20beff10;padding:40px 140px 26px;font-size:30px;font-weight:700;color:#1f2430}
    .top span{display:block;font-size:22px;font-weight:500;color:#5b6475;margin-top:6px}
    .wrap{padding:16px 140px 600px}
    .cell{background:#fff;border-radius:10px;margin:18px 0;box-shadow:0 1px 6px rgba(30,40,60,.08);overflow:hidden}
    .hdr{font-family:'DejaVu Sans Mono',monospace;font-size:23px;background:#f5f7fa;color:#3b6ea5;padding:14px 22px;
         border-bottom:1px solid #e3e6eb;font-weight:600}
    pre{font-family:'DejaVu Sans Mono',monospace;font-size:22px;line-height:1.5;color:#2e3440;padding:14px 22px;white-space:pre-wrap}
    """
    return (f'<!doctype html><html><head><meta charset="utf-8"><style>{css}</style></head><body>'
            f'<div class="top">{html.escape(title)}<span>Outputs as saved in the executed notebook (Kaggle, NVIDIA T4)</span></div>'
            f'<div class="wrap">{"".join(blocks)}</div></body></html>')


def trim01(hdr, lines):
    if hdr.startswith('BLOCK 5.6'):
        keep = [l for l in lines if l.startswith(('LM_WEIGHT_GRID', 'LAMBDA_GRID', 'Joint sweep', 'Best combo', '>>> Final'))]
        thr = [l for l in lines if 'cutoff' in l]
        return keep[:2] + ['... 12 n-gram weights x 11 fusion weights, gate re-calibrated at each n-gram weight ...'] + thr[-2:] + keep[2:]
    if hdr.startswith('BLOCK 6'):
        return [l for l in lines if 'Skipped' in l or 'Finished' in l or 'Test samples' in l]
    return lines[:14]


def trim02(hdr, lines):
    if hdr.startswith('BLOCK 7'):
        i = next((k for k, l in enumerate(lines) if 'SUMMARY' in l), 0)
        head = [l for l in lines[:i] if l.startswith('[rev]') and 'NEXT' not in l]
        tail = [l for l in lines[i:i + 11] if 'NEXT' not in l and 'then submit' not in l]
        return head + tail
    return lines[:12]


def shoot(pages):
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={'width': W, 'height': H}, device_scale_factor=1)
        for name, (dur, doc) in pages.items():
            f = os.path.join(FR, name + '.html')
            open(f, 'w').write(doc)
            pg.goto('file://' + f)
            pg.wait_for_timeout(300)
            pg.screenshot(path=os.path.join(FR, name + '.png'))
        for name, doc in NB_PAGES.items():
            f = os.path.join(FR, name + '.html')
            open(f, 'w').write(doc)
            pg.goto('file://' + f)
            pg.wait_for_timeout(300)
            pg.screenshot(path=os.path.join(FR, name + '.png'), full_page=True)
        b.close()


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-2000:])
        raise SystemExit('ffmpeg failed')


def still(name, dur):
    tmp, dst = os.path.join(OUT, name + '.tmp.mp4'), os.path.join(OUT, name + '.mp4')
    run(['ffmpeg', '-y', '-framerate', str(FPS), '-i', os.path.join(FR, name + '.png'),
         '-vf', f'format=yuv420p,loop=-1:1,trim=duration={dur},fade=t=in:st=0:d=0.4,fade=t=out:st={dur - 0.4}:d=0.4',
         *ENC, tmp])
    os.replace(tmp, dst)


FONT = ImageFont.truetype('/usr/share/fonts/opentype/inter/Inter-SemiBold.otf', 34) \
    if os.path.exists('/usr/share/fonts/opentype/inter/Inter-SemiBold.otf') else ImageFont.load_default()


def ease(t):
    return t * t * (3 - 2 * t)


def scroll(name, plan):
    """plan: list of (y_from, y_to, seconds, caption); y = top of the 1080 window."""
    big = Image.open(os.path.join(FR, name + '.png')).convert('RGB')
    maxy = big.height - H
    total = sum(p[2] for p in plan)
    tmp, dst = os.path.join(OUT, name + '.tmp.mp4'), os.path.join(OUT, name + '.mp4')
    ff = subprocess.Popen(['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS),
                           '-i', '-', *ENC, tmp], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    k = 0
    nfr_total = int(round(total * FPS))
    for y0, y1, sec, cap in plan:
        n = int(round(sec * FPS))
        for i in range(n):
            y = int(round(y0 + (y1 - y0) * ease(i / max(n - 1, 1))))
            y = max(0, min(maxy, y))
            fr = big.crop((0, y, W, y + H))
            d = ImageDraw.Draw(fr, 'RGBA')
            d.rectangle((0, H - 120, W, H), fill=(31, 36, 48, 235))
            d.text((140, H - 80), cap, font=FONT, fill=(236, 239, 244))
            a = min(1.0, k / (0.4 * FPS), (nfr_total - 1 - k) / (0.4 * FPS))
            if a < 1:
                fr = Image.blend(Image.new('RGB', (W, H)), fr, max(a, 0))
            ff.stdin.write(fr.tobytes())
            k += 1
    ff.stdin.close()
    if ff.wait() != 0:
        raise SystemExit('ffmpeg scroll failed')
    os.replace(tmp, dst)
    return big.height


NB_PAGES = {
    's07': nb_page('notebooks/01_full_pipeline_executed.ipynb', nb_cells(os.path.join(REPO, 'notebooks/01_full_pipeline_executed.ipynb')),
                   ['BLOCK 1 ', 'BLOCK 5.1', 'BLOCK 5.2', 'BLOCK 5.3', 'BLOCK 5.4', 'BLOCK 5.5', 'BLOCK 5.6'], trim01),
    's08': nb_page('notebooks/02_revision_experiments_executed.ipynb', nb_cells(os.path.join(REPO, 'notebooks/02_revision_experiments_executed.ipynb')),
                   ['BLOCK 7'], trim02),
}

if __name__ == '__main__':
    shoot(SLIDES)
    if len(sys.argv) > 1 and sys.argv[1] == 'shots':
        for n in NB_PAGES:
            print(n, Image.open(os.path.join(FR, n + '.png')).size)
        raise SystemExit
    for name, (dur, _) in SLIDES.items():
        still(name, dur)
    plans = json.load(open(os.path.join(ROOT, 'scroll_plan.json')))
    for name, plan in plans.items():
        scroll(name, plan)
    order = ['s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10', 's11']
    lst = os.path.join(OUT, 'list.txt')
    open(lst, 'w').write(''.join(f"file '{os.path.join(OUT, n + '.mp4')}'\n" for n in order))
    tmp = os.path.join(ROOT, 'walkthrough.tmp.mp4')
    run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', lst, '-c', 'copy', os.path.join(ROOT, 'video_noaudio.mp4')])
    run(['ffmpeg', '-y', '-i', os.path.join(ROOT, 'video_noaudio.mp4'), '-f', 'lavfi', '-i', 'anullsrc=r=48000:cl=stereo',
         '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', '-shortest', '-movflags', '+faststart', tmp])
    os.replace(tmp, os.path.join(ROOT, 'spikes_to_sentences_walkthrough.mp4'))
    print('done')
