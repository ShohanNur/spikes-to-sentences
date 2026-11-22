# Walkthrough video

`spikes_to_sentences_walkthrough.mp4` (2 min 13 s, 1080p, captions burnt in,
silent track) is ready to upload as it is. Every number on screen comes from
the paper or from the executed notebooks in `notebooks/`. It is rebuilt by
`docs/video/make_video.py` (Playwright + ffmpeg) if a slide needs changing.

Reviewer 2 asked for a video "to show the validity of their initial work". The
strongest version adds your own voice and, ideally, 30-60 s of a real screen
recording of the Kaggle notebook running. Both are optional; the timings below
let you do either without re-editing.

## Scenes and narration (read at a calm pace; each fits its slot)

| # | Time | On screen | Narration |
| --- | --- | --- | --- |
| 1 | 0:00-0:07 | Title, authors | "This is the replication walkthrough for From Spikes to Sentences, our ICCIT 2026 paper." |
| 2 | 0:07-0:16 | The task | "The task is to decode attempted speech from 256 intracortical electrodes into text, on the Brain-to-Text 2025 benchmark." |
| 3 | 0:16-0:26 | Pipeline, Fig. 1 | "A CTC model gives phoneme posteriors, a beam search with a 4-gram model gives thirty candidates, and an LLM rescores them only when the decoder is undecided." |
| 4 | 0:26-0:34 | Acoustic model, Fig. 2 | "It combines convolution, a bidirectional LSTM and a patch Transformer, with per-session adapters." |
| 5 | 0:34-0:46 | Gate, Fig. 3 | "The gate skips the language model when the list has collapsed or is already decided. Inside it, the weight fades to zero with the score margin." |
| 6 | 0:46-0:58 | Four steps on Kaggle | "To reproduce it, import notebook 02 into Kaggle, enable a T4 GPU and internet, add the four data sources, and press Run All." |
| 7 | 0:58-1:20 | Notebook 01 outputs | "These are the saved outputs of the run in the paper: 7.55 percent word error rate at the first setting, phoneme error 12.12 percent, and after the 132-point grid, 7.26 percent at the best setting." |
| 8 | 1:20-1:42 | Notebook 02, Block 7 | "Block 7 scores every list once with the language model and compares seven rules on the same lists. The word error gain is a statistical tie; the character error gain is significant. With calibration on other sessions, no rule beats the decoder alone." |
| 9 | 1:42-1:55 | Results table | "Our rule is the best of the language-model rules and calls the model on 68 percent of sentences. On the hidden test set it scores 9.34 percent private and 8.21 percent public." |
| 10 | 1:55-2:04 | Rebuild the paper | "Block 7 writes a LaTeX file; drop it next to the paper and recompile, and every number in the table updates." |
| 11 | 2:04-2:13 | Closing | "Code, notebooks, results and the model files are on GitHub and Zenodo. Thank you." |

## Adding your voice (10 minutes)

1. Record the narration with any phone or laptop app, one take per row, and
   join the takes into `voice.m4a` (or record it in one go against the video).
2. Mix it in without re-encoding the picture:
   ```bash
   ffmpeg -i spikes_to_sentences_walkthrough.mp4 -i voice.m4a \
     -map 0:v -map 1:a -c:v copy -c:a aac -shortest walkthrough_voice.mp4
   ```

## Optional: a real run on screen (strongest evidence)

Record 30-60 s with OBS Studio or the Windows Game Bar (Win+Alt+R) while the
Kaggle notebook shows: the input panel with the four data sources, the
"Model ready" and "Decoder ready" lines, and the Block 7 summary. Put it after
scene 6:

```bash
ffmpeg -i live.mp4 -vf "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,fps=30,format=yuv420p" \
  -c:v libx264 -crf 18 -preset veryfast -an live_1080.mp4
```
and concatenate (all parts must be 1920x1080, 30 fps, H.264).

## Upload

- YouTube, visibility **Public** or **Unlisted**.
- Title: *From Spikes to Sentences (ICCIT 2026): replication walkthrough*
- Description:
  > Walkthrough of the replication package for "From Spikes to Sentences: A
  > Hybrid CTC Decoder with Confidence-Gated Language Model Rescoring"
  > (A. Biswas, S. Rahman, M. H. Anjan, ICCIT 2026).
  > Code: GITHUB_URL · Archive and model files: https://doi.org/ZENODO_DOI
- Put the link in `paper/main.tex` (`\VideoLink`) and in `README.md`.
