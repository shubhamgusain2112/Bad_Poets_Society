# Bad Poets Society
## Computational Creativity Assignment 2

A poem generator: a fine-tuned Qwen3-0.6B-Base writes the lines, and a controller gives each poem a poet's line length and rhyme scheme and picks the last word of every line.

### Contributors
| Team | Student Number | GitHub |
| - | - | - |
| Rahul Ray | s4875710 | [[LINK]](https://github.com/rahulrayy) |
| Shubham Gusain | s4846761 | [[LINK]](https://github.com/shubhamgusain2112/) |

### Sources
| Use | Name | Link |
| - | - | - |
| LLM (writes the poems) | Qwen 3 Base (0.6B) | [[LINK]](https://huggingface.co/Qwen/Qwen3-0.6B-Base) |
| LLM (fluency score) | Qwen 3 Base (1.7B) | [[LINK]](https://huggingface.co/Qwen/Qwen3-1.7B-Base) |
| Pronunciation database | CMUDict | [[LINK]](https://github.com/Alexir/CMUdict) |
| Rhyme database | PeRDict Database | [[LINK]](https://github.com/scrosseye/PeRDict-database) |
| Syllable database | English phonetic and syllable count dictionary | [[LINK]](https://www.kaggle.com/datasets/schwartstack/english-phonetic-and-syllable-count-dictionary) |
| Poetry dataset | DanFosing/public-domain-poetry | [[LINK]](https://huggingface.co/datasets/DanFosing/public-domain-poetry) |
| Poetry dataset (main training data) | Gutenberg Poetry Corpus | [[LINK]](https://huggingface.co/datasets/biglam/gutenberg-poetry-corpus) |
| Authors of the Gutenberg books | Project Gutenberg catalogue | [[LINK]](https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv) |
| Word vectors (system M) | GloVe (wiki-gigaword-300) | [[LINK]](https://github.com/RaRe-Technologies/gensim-data/releases/tag/glove-wiki-gigaword-300) |

### Structure
- `scripts/` - all the code, as notebooks
- `lora_weights/` - the fine-tuned model (LoRA weights), the meaning head (`meaning_head.pt`), the line-end vocabulary (`vocab.npz`) and the training log (`train_log.txt`, has the loss and header check numbers from the report)
- `outputs/` - our results:
  - `sweep/` and `sweep_scores.csv` - the 120 poems from the evaluation and their scores
  - `lambda_curve.png` - the plot in the report
  - `final_poems.txt` - the 160 poems the presented poems were picked from, with poet, system, λ, seed and rhyme pairs above each one
  - `final/` and `final_poems.json` - the same 160 poems with a trace for every line (what Qwen wanted, what got picked and why)

### Rerunning the experiments
An Nvidia GPU is useful for running the experiments.

**1. Install**

We used Python 3.12.
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install jupyter
```
(On Linux activate with `source .venv/bin/activate`.)

**2. Get the models** (stored in `model/`, ~1.2GB and ~3.4GB)
```
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B-Base', local_dir='model/Qwen3-0.6B-Base')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-1.7B-Base', local_dir='model/Qwen3-1.7B-Base')"
```

**3. Get the data**
The data is sourced from the sources mentioned above.
```
data/cmudict/cmudict-0.7b
data/train_test/train.parquet, val.parquet, named_poets.csv
data/author_profiles.csv
data/pdp_clean.parquet
```
To build it yourself instead, download the three datasets from the sources table and CMUdict, then run `clean_pdp`, `clean_gutenberg`, `data_prep` and `author_profiles` in `scripts/`, in that order. Gutenberg updates its catalogue every day, so a few author names may come out slightly different from ours.

**4. Run**

Open the notebooks from inside `scripts/` and run all cells:
1. `evaluate.ipynb` - makes the 120 evaluation poems (B1 and M, λ 0/0.25/0.5/0.75/1, 3 rhymed poets, seeds 0-3), scores them on rhyme, fluency and copying, makes the table and plot, and then the 160 final poems. It loads `generate.ipynb` and `rhyme.ipynb` itself. Takes about an hour.
2. `inspiring_set.ipynb` - selects the real poems of the inspiring set and compares them to the generated ones (line length, rhyme scheme, copying).

Poems that already exist in `outputs/` are skipped. So to make the poems again instead of just rescoring ours, move `outputs/sweep/` and `outputs/final/` somewhere else first. The same seed should give the same poem, but a different GPU or library version can change a few words.

To make a single poem, run `generate.ipynb` and call e.g. `generate_poem("George Gordon Byron", lam=0.5, system="M")`. Any poet from `data/author_profiles.csv` will work.