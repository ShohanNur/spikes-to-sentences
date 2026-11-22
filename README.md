# From Spikes to Sentences

**A Hybrid CTC Decoder with Confidence-Gated Language Model Rescoring**

Anik Biswas\*, Md. Shohanur Rahman\*, Mehedi Hasan, Celia Shahnaz
*2026 29th International Conference on Computer and Information Technology (ICCIT), Cox's Bazar, Bangladesh*
\* equal contribution

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24432296.svg)](https://doi.org/10.5281/zenodo.24432296)
Video walkthrough: https://youtu.be/OtxaGQm6QDg

This repository holds the code, executed notebooks, results and paper source for
the ICCIT 2026 paper. The archived release on Zenodo
([10.5281/zenodo.24432296](https://doi.org/10.5281/zenodo.24432296)) holds the
trained checkpoint (`swa_model.pt`), the pronunciation lexicon and token list
(`lexicon.txt`, `tokens.txt`) and the 4-gram KenLM model (`custom_4gram.bin`,
19.1 GB).

## What the system does

1. A 28.8 M-parameter hybrid CNN-BiLSTM-Transformer, trained with CTC, maps
   256-electrode intracortical features (participant T15, Brain-to-Text '25) to
   phoneme posteriors. Per-session affine adapters are tied by a drift penalty.
2. A lexicon-constrained beam search with a 4-gram KenLM turns the posteriors
   into a 30-best list.
3. A 4-bit Qwen2.5-7B rescores the list only when the decoder is undecided: a
   two-sided confidence gate skips collapsed and already-resolved lists, and
   inside the gate the fusion weight decays to zero with the top-1/top-2 margin.

## Results in the paper

Validation split (1,426 sentences):

| Setting | WER (%) | CER (%) | LLM calls |
| --- | --- | --- | --- |
| Beam + 4-gram, alpha = 5.5 | 7.37 | 5.33 | 0 % |
| Gated adaptive fusion, OP-B (alpha = 5.5, lambda_0 = 1.25) | 7.26 | 5.18 | 68 % |
| Same, thresholds fitted on the other half of the sessions | 7.39 | 5.31 | not logged |

- WER change -0.11 points, 95 % paired bootstrap CI [-0.34, +0.12] (not significant).
- CER change -0.15 points, CI [-0.29, -0.01], p = 0.041 (significant when calibrated on validation).
- PER (greedy CTC) 12.12 %.
- Held-out test (competition server, thresholds frozen from validation): 9.34 % WER private, 8.21 % public.
- Latency on an NVIDIA T4: 67 ms beam search, 1.08 s per batched LLM pass (median 0.79 s).

Full comparison with six other rescoring rules: `results/revision_results.json`
and Table II of the paper.

## Repository layout

| Path | Contents |
| --- | --- |
| `notebooks/01_full_pipeline_executed.ipynb` | Training code, beam search, gated fusion at OP-A, the 132-point sweep of Fig. 4 and the test submission, **with the outputs of the run reported in the paper**. |
| `notebooks/02_revision_experiments.ipynb` | Clean notebook that reproduces Table II, the statistics, latency and test files (about 2-2.5 h on a T4). |
| `notebooks/02_revision_experiments_executed.ipynb` | The same notebook **with the outputs of the run reported in the paper**. |
| `scripts/revision_experiments.py` | Block 7 of notebook 02 as a standalone file. |
| `tests/mock_harness.py` | Runs Block 7 on synthetic N-best lists, no GPU or data needed. |
| `results/` | `revision_results.json` / `.tex` (every number in Table II and Section VIII-B), `val_wer.json` (OP-A run), `opb_nbest_llm_scores.json` (the 30-best lists and their LLM scores). |
| `paper/` | LaTeX source of the paper (`main.tex`, `IEEEtran.cls`, `revision_results.tex`). |
| `docs/` | Release steps and the video script. |

## Reproducing on Kaggle

1. Create a Kaggle notebook and import `notebooks/02_revision_experiments.ipynb`
   (File, Import Notebook).
2. Settings: Accelerator **GPU T4**, Internet **on**.
3. Add inputs:
   - competition data `brain-to-text-25`;
   - dataset `shohan3125/brain-to-text-checkpoint` (`swa_model.pt`);
   - dataset `heyyousum/quality-english-dataset-for-ngram-model-v2` (`lexicon.txt`, `tokens.txt`);
   - dataset `heyyousum/custom-4-gram-wiki-news-switchboard-updated-v3` (`custom_4gram.bin`).

   If a dataset is not visible to you, download the four files (`swa_model.pt`,
   `lexicon.txt`, `tokens.txt`, `custom_4gram.bin`) from the
   [Zenodo record](https://doi.org/10.5281/zenodo.24432296), upload them as your
   own Kaggle dataset and change the dataset names in Block 1
   (`PRETRAINED_CKPT_DATASET`) and Block 5.2.
4. Run All. The notebook loads the checkpoint (no training), reports OP-A
   (WER 7.55 %, CER 5.38 % to 5.31 %, PER 12.12 %), then Block 7 decodes at
   OP-B, scores every list once with the LLM, compares seven rules, runs the
   session-disjoint check and the bootstrap, times both stages and writes
   `submission_rev_baseline.csv` and `submission_rev_opb.csv`.
5. Outputs land in `/kaggle/working`: `revision_results.tex`,
   `revision_results.json` and the two submission files.

To retrain the acoustic model, set `SKIP_TRAINING = False` in notebook 01
(82 epochs, early stopping on validation WER, weight averaging of the three best
checkpoints). To redo the 7.6 h sweep of Fig. 4, set
`RUN_HYPERPARAM_SWEEPS = True` in notebook 01.

## Rebuilding the paper

```bash
cd paper
# put your own revision_results.tex here to rebuild with your numbers
pdflatex main.tex && pdflatex main.tex && pdflatex main.tex
```

## Checking the statistics code without a GPU

```bash
pip install numpy scipy jiwer editdistance
mkdir -p mock_out
REV_N_BOOT=400 python3 tests/mock_harness.py ./mock_out scripts/revision_experiments.py
```

The mock numbers are synthetic; the run only shows that the code executes end
to end and writes valid LaTeX.

## Citation

```bibtex
@inproceedings{biswas2026spikes,
  author    = {Biswas, Anik and Rahman, Md. Shohanur and Hasan, Mehedi and
               Shahnaz, Celia},
  title     = {From Spikes to Sentences: A Hybrid {CTC} Decoder with
               Confidence-Gated Language Model Rescoring},
  booktitle = {2026 29th International Conference on Computer and
               Information Technology (ICCIT)},
  address   = {Cox's Bazar, Bangladesh},
  year      = {2026},
  note      = {Model files and replication package:
               \url{https://doi.org/10.5281/zenodo.24432296}}
}
```

## License

Code: MIT (see `LICENSE`). The neural data belong to the Brain-to-Text '25
competition and keep its terms; the released checkpoint was trained on those
data and is shared for research use under the same terms.

## Contact

Anik Biswas, 1906125@eee.buet.ac.bd
