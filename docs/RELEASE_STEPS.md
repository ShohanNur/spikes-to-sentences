# Release and camera-ready steps

Do these in order. The DOI has to exist before the paper can cite it, so the
Zenodo record is reserved first and published last.

## 1. GitHub (code)

1. Create a public repository, e.g. `spikes-to-sentences`.
2. Replace `OWNER` in `CITATION.cff` and `.zenodo.json` with your GitHub user
   name, then push this folder:
   ```bash
   git init && git add . && git commit -m "Release for ICCIT 2026"
   git branch -M main
   git remote add origin https://github.com/ShohanNur/spikes-to-sentences.git
   git push -u origin main
   ```
   `.gitignore` keeps model files out of git (GitHub refuses files over 100 MB).

## 2. Zenodo (archive with a DOI)

1. Log in at zenodo.org (you can use your GitHub account) and click **New upload**.
2. Click **Reserve DOI** first and copy it (`10.5281/zenodo.NNNNNNN`).
3. Files to upload:
   - `spikes-to-sentences-v1.0.0.zip`: this repository (GitHub "Download ZIP").
   - `models.zip`, made from your Kaggle datasets:
     - `swa_model.pt` from `shohan3125/brain-to-text-checkpoint`
     - `lexicon.txt`, `tokens.txt` from `heyyousum/quality-english-dataset-for-ngram-model-v2`
     - `custom_4gram.bin` from `heyyousum/custom-4-gram-wiki-news-switchboard-updated-v3`

     Download them with `kaggle datasets download <name>` or from each dataset
     page. Zenodo accepts up to 50 GB per record.
   - The final paper PDF (after step 3), optional.
4. Metadata: copy the fields from `.zenodo.json`. Resource type **Software**,
   license **MIT**, related identifier = the GitHub URL ("is supplement to").
5. Publish only after step 3, so the archived PDF already carries the DOI.

Using Zenodo's automatic GitHub release would archive the source only, without
the model files, so the manual upload above is the simpler route here.

## 3. Fill the paper

In `paper/main.tex`, edit only the block marked `CAMERA-READY DETAILS`:

```latex
\newcommand{\AuthEmailB}{shohanur's e-mail}
\newcommand{\AuthEmailC}{anjan's e-mail}
\newcommand{\ZenodoLink}{\url{https://doi.org/10.5281/zenodo.NNNNNNN}}
\newcommand{\GitHubLink}{\url{https://github.com/ShohanNur/spikes-to-sentences}}
\newcommand{\VideoLink}{\url{https://youtu.be/XXXXXXXXXXX}}
```

Compile three times. Nothing red may remain in the PDF:
`pdftotext main.pdf - | grep -n "\["` should print no placeholder.
Also put the DOI badge and video link at the top of `README.md` and push.

## 4. Video

Follow `docs/VIDEO_SCRIPT.md`; upload to YouTube as **Public** or **Unlisted**
(never Private: reviewers and readers cannot open it).

## 5. IEEE PDF eXpress (required by ICCIT 2026)

1. pdf-express.org, **Conference ID 72912X**, create an account with the paper ID.
2. Upload the compiled PDF (or the LaTeX source) and wait for the check.
3. Save the approved PDF as `PID.pdf` and the success e-mail as `PID_express.pdf`.

## 6. CMT camera-ready upload (ICCIT 2026 guidelines)

Deadline on the guidelines page: 25 November 2026 (the Important Dates page
says 15 November; confirm with the organizers).

| File | What |
| --- | --- |
| `PID.pdf` | PDF eXpress-approved paper, at most 6 pages, under 5 MB, fonts embedded |
| `PID_express.pdf` | PDF eXpress success message |
| `PID_copyright.pdf` | IEEE electronic copyright form (done in CMT) |
| `PID_payment.pdf` | registration payment slip |
| `PID_ieee.pdf`, `PID_student.pdf`, `PID_online.pdf` | only if they apply |

The paper title in the PDF must match the title in CMT and on the copyright
form: *From Spikes to Sentences: A Hybrid CTC Decoder with Confidence-Gated
Language Model Rescoring*. The first-page header and the copyright line
`979-8-3195-0171-4/26/$31.00 (c)2026 IEEE` are already in `main.tex`.
Similarity must be under 30 %.

## Only if the paper goes back to double-blind reviewers

Use `main_blind.tex` and an anonymized mirror of the GitHub repository from
anonymous.4open.science instead of the Zenodo and GitHub links. Do not use the
named repository or the Zenodo record while the review is blind.
