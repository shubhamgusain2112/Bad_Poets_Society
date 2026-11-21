# Bad Poets Society:
## Computational Creativity Assignment 2

### Contributors
| Team | Student Number | GitHub |
| - | - | - |
| Rahul Ray | s4875710 |[[LINK]](https://github.com/rahulrayy) |
| Shubham Gusain | s4846761| [[LINK]](https://github.com/shubhamgusain2112/)

### Sources 
| Use | Name | Link |
| - | - | - |
| LLM | Qwen 3 Base (0.6B) | [[LINK]](https://huggingface.co/Qwen/Qwen3-0.6B-Base) | 
| Pronunciation Database | CMUDict | [[LINK]](https://github.com/Alexir/CMUdict) |
| Rhyme Database | PeRDict Database | [[LINK]](https://github.com/scrosseye/PeRDict-database) | 
| Syllable Database | English phonetic and syllable count dictionary | [[LINK]](https://www.kaggle.com/datasets/schwartstack/english-phonetic-and-syllable-count-dictionary) | 
| Poetry Database | DanFosing/public-domain-poetry | [[LINK]](https://huggingface.co/datasets/DanFosing/public-domain-poetry) | 

### Other stuff we use
- [Gutenberg Poetry Corpus](https://huggingface.co/datasets/biglam/gutenberg-poetry-corpus) - main training data, 3.1M lines from 1,191 books
- [Gutenberg catalogue](https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv) - to find out who wrote each of those books
- [GloVe word vectors](https://github.com/RaRe-Technologies/gensim-data/releases/tag/glove-wiki-gigaword-300) - the "meaning" space we pick the last word of each line from

### What's in the repo
**`scripts/`** - all the code, as notebooks (what each one does and the order to run them is under "How to run")
- `clean_pdp.ipynb`, `clean_gutenberg.ipynb`, `data_prep.ipynb`, `author_counts.ipynb` - data cleaning and the training data (step 4)
- `train.ipynb` - LoRA fine-tuning (step 5)
- `rhyme.ipynb`, `author_profiles.ipynb`, `inspiring_set.ipynb`, `meaning_head.ipynb`, `generate.ipynb`, `evaluate.ipynb` - the poem generator and its evaluation (step 8)

**`lora_weights/`** - our fine-tuned model (the LoRA part, goes on top of Qwen3-0.6B-Base)
- `adapter_config.json`, `adapter_model.safetensors` - the LoRA weights
- `train_log.txt` - the log of the training run on vibranium: loss every 10 steps, val loss and the header check every 250 steps
- `meaning_head.pt` - the trained meaning head from `meaning_head.ipynb` (system M needs it), so you don't have to train it yourself
- `vocab.npz` - the line-end vocabulary from `meaning_head.ipynb`: 52,544 words with their GloVe vectors and rhyme richness (the generator needs it, so you don't need to download GloVe)

**`chatbot/`** - 2 small scripts to try the untuned model (see "Chatbot scripts" at the bottom)

**`outputs/`** - everything the generator made (from `generate.ipynb` and `evaluate.ipynb`)
- `final_poems.txt` - the easiest one to read: 160 poems, 5 per showcase poet for both systems (B1 and M) at every λ (Longfellow only λ 1). Above every poem: poet, system, λ, seed and how many of its rhyme pairs actually rhyme. The poems we present are picked from here
- `final_poems.json` - the same 160 poems with their traces, in one file
- `final/` - the same 160 poems again, one file each
- `sweep/` - the 120 poems for the evaluation: B1 and M, λ 0/0.25/0.5/0.75/1, the 3 rhymed showcase poets, seeds 0-3
- `traces/` - the 20 example poems at the bottom of `generate.ipynb` (seed 0)
- `sweep_scores.csv` - one row per sweep poem: system, poet, λ, seed, rhyme pairs, how many rhymed, perplexity under Qwen3-1.7B (`ppl`) and share of lines with 5 words in a row from the training data (`copied`)
- `lambda_curve.png` - rhyme rate vs perplexity for B1 and M as λ changes, the main result plot

The poem files are named `<system>_<poet>_lam<λ>_seed<seed>.json`. Each one has the poem and its settings (poet, system, λ, scheme, seed, `stop` = why it stopped), plus a trace for every line in `lines`:
- `text`, `syllables`, `target` - the line, its syllables and how many syllables we aimed for
- `partner` - the word this line had to rhyme with (empty if it didn't have to)
- `qwen_top` - the word Qwen itself wanted to end the line with
- `meant` - the system's best word before rhyme counts (for M: the meaning head's guess)
- `chosen` - the word that got picked
- `candidates` - the top 10 words with their probability `p`, rhyme score `rhyme` and final `score`

**Not in the repo** (too big, see "How to run" for how to get or make them): `model/` (Qwen, the 1.7B judge model, training checkpoints) and `data/` (the datasets, CMUdict, GloVe and everything the notebooks make from them)

### How to run
We used Windows + Python 3.12, should work on other setups too.

**1. Install stuff**
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
That's the exact versions we used. The torch in there is the Nvidia GPU version, so no Nvidia GPU? take the torch line out of `requirements.txt` and just do `pip install torch`. To open the notebooks in the browser also do `pip install jupyter` (PyCharm / VS Code don't need it).

**2. Get the model** (~1.2GB, goes in `model/` which isn't in the repo)
```
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B-Base', local_dir='model/Qwen3-0.6B-Base')"
```

**3. Get the data**

The data isn't in the repo either. Easiest is to ask us for our `data/` folder, then you get the exact same files we used. Or download these 3 and put them here:
```
data/public-domain-poetry/poems.json
data/gutenberg-poetry-corpus/data/train-00000-of-00001-fa9fb9e1f16eed7e.parquet
data/gutenberg-catalog/pg_catalog.csv
```
- poems.json: [[LINK]](https://huggingface.co/datasets/DanFosing/public-domain-poetry/resolve/main/poems.json)
- the parquet: [[LINK]](https://huggingface.co/datasets/biglam/gutenberg-poetry-corpus/resolve/main/data/train-00000-of-00001-fa9fb9e1f16eed7e.parquet)
- pg_catalog.csv: [[LINK]](https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv)

(Gutenberg updates the catalogue every day, so if you download it yourself a few author names might come out slightly different.)

The poem-making notebooks (step 8) also need these two (GloVe only if you retrain the meaning head):
- `data/cmudict/cmudict-0.7b` [[LINK]](https://raw.githubusercontent.com/Alexir/CMUdict/HEAD/cmudict-0.7b)
- `data/glove/glove-wiki-gigaword-300.gz` (~400MB) from [[LINK]](https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-300/glove-wiki-gigaword-300.gz)

**4. Run the notebooks**

Open them from inside `scripts/` (Jupyter, PyCharm or VS Code all work) and run all cells, in this order:

1. `clean_pdp.ipynb` - cleans the poetry dataset: removes short poems, prose, translations and duplicates, fixes the OCR thing where "O" became "0", tidies author names. Makes `data/pdp_clean.parquet` (should be 34,245 poems)
2. `clean_gutenberg.ipynb` - same idea for the Gutenberg corpus, plus works out who wrote each book. Needs step 1 done first and takes a few minutes. Makes `data/gutenberg_clean.parquet` (869 books)
3. `data_prep.ipynb` - builds the training data: takes out the Gutenberg bits that are also in the poetry dataset, every poet with 500+ lines keeps their name (rest become "Unknown"), cuts everything into 40-line pieces that start with `Author: <name>`, splits 90/5/5 by poem/book and tokenises. Needs 1 and 2 done first, takes ~10 mins. Makes `data/train_test/train.parquet`, `val.parquet`, `test.parquet` (about 89k / 5.8k / 5k pieces) and `named_poets.csv` (486 poets)
4. `author_counts.ipynb` - optional, just counts how many lines each poet has. Makes `data/author_counts.csv`

**5. Train (on vibranium)**

`scripts/train.ipynb` fine-tunes Qwen with LoRA on the data from step 4. We run it on vibranium.liacs.nl (LIACS REL, 2 RTX 3090s with 24GB each, no slurm, first come first served). This assumes you're logged in there already.

1. Put everything in `/local/$USER`, not in your home dir (7.5GB quota) or `/data`. The server's default python is 3.9, which is too old, so make the venv with uv:
```
mkdir -p /local/$USER && cd /local/$USER
git clone https://github.com/shubhamgusain2112/Bad_Poets_Society.git && cd Bad_Poets_Society
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt --index-strategy unsafe-best-match
```
(the `--index-strategy` bit is needed so uv takes the CUDA torch from the pytorch index. No uv? `pip install uv` first)

2. Get the model and data: either do steps 2-4 here on the server (there's no Jupyter window, so run the notebooks with the same `python -m nbconvert --to notebook --execute --inplace ...` command as below), or copy our folders over from a laptop that has them. For that, first do `mkdir -p model data` on the server, then on the laptop:
```
scp -r -J USERNAME@ssh.liacs.nl,USERNAME@homeowl.research.liacs.nl model/Qwen3-0.6B-Base USERNAME@vibranium.liacs.nl:/local/USERNAME/Bad_Poets_Society/model/
scp -r -J USERNAME@ssh.liacs.nl,USERNAME@homeowl.research.liacs.nl data/train_test USERNAME@vibranium.liacs.nl:/local/USERNAME/Bad_Poets_Society/data/
```

3. Check which of the 2 GPUs is free with `nvidia-smi` (or `nvtop`) and pick it with `CUDA_VISIBLE_DEVICES`. Run inside tmux so it keeps going when you log out:
```
tmux new -s train
cd /local/$USER/Bad_Poets_Society && source .venv/bin/activate
export JUPYTER_PATH=$PWD/.venv/share/jupyter
```
(the `JUPYTER_PATH` line makes sure it runs on the venv's python, not some other Jupyter on the server)

First the quick test (~5 mins):
```
CUDA_VISIBLE_DEVICES=0 BATCH=4 python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 scripts/train.ipynb
```
If that finishes without errors, start the real run:
```
CUDA_VISIBLE_DEVICES=0 TEST_RUN=0 BATCH=4 python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 scripts/train.ipynb
```
`BATCH` is how many pieces go through the GPU at once. It only changes speed and memory, not the result (every step is 32 pieces either way). 4 should fit on a 3090 and 8 won't, going by what we measured on our laptop (not tested on vibranium yet). If it stops saying it needs more free GPU memory, someone else is using that GPU: pick the other one or use `BATCH=2`. Ctrl+b then d to leave tmux, `tmux attach -t train` to go back.

4. Watch it with `tail -f model/train_log.txt`. It saves a checkpoint every 250 steps in `model/lora_runs/` (keeps the last 3) and logs the val loss and a "header check" there: if the gap goes up, the model is learning the different poets' styles. You can stop (Ctrl+C) once those stop improving, it doesn't have to finish the whole pass.

5. The servers reboot on the 2nd Sunday of every month around 23:30, which kills tmux too. If that happens, or it stops for any other reason, just run the real-run command again and it carries on from the last checkpoint.

6. When it's done, the LoRA weights are in `model/lora/`. Put them on GitHub (step 6 below) and clean up `/local/$USER` so the space is free for others.

**6. Put the LoRA weights on GitHub**

`model/` isn't in the repo, so copy the 2 files that matter into `lora_weights/` (unlike `model/`, git doesn't ignore that folder) and push. On vibranium, in `/local/$USER/Bad_Poets_Society`:
```
mkdir -p lora_weights
cp model/lora/adapter_config.json model/lora/adapter_model.safetensors lora_weights/
git add lora_weights
git commit -m "Add LoRA weights"
git pull
git push
```
(if `git pull` complains about `scripts/train.ipynb`, that's just the output of your runs: `git checkout scripts/train.ipynb` and pull again)
- Stopped early? Take the 2 files from the last checkpoint instead, `model/lora_runs/checkpoint-<biggest number>/`. Only those 2, not the optimizer files, those are big.
- It's ~40MB, fine for GitHub (the limit is 100MB per file).
- GitHub won't take your password for the push. Use a personal access token (GitHub → Settings → Developer settings → Personal access tokens) or an ssh key. Or scp the `lora_weights` folder to your laptop and push from there.

**7. Use the trained model on your own machine**

`git pull` to get `lora_weights/`. You also need steps 1 and 2 (packages + base model). Then run this from the repo folder:
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("model/Qwen3-0.6B-Base")
base = AutoModelForCausalLM.from_pretrained("model/Qwen3-0.6B-Base", dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device)
model = PeftModel.from_pretrained(base, "lora_weights").eval()

prompt = "Author: Emily Dickinson\n\n"
inputs = tok(prompt, return_tensors="pt").to(device)
out = model.generate(**inputs, max_new_tokens=120, do_sample=True, temperature=0.8, top_p=0.95, pad_token_id=tok.eos_token_id)
print(tok.decode(out[0], skip_special_tokens=True))
```
Swap in any poet from `data/train_test/named_poets.csv`, or `Author: Unknown`. It stops by itself when the model ends the poem, or after `max_new_tokens`. Works without a GPU too, just slower.

**8. Make poems**

This is the actual generator. The model above just writes, these notebooks give the poem the poet's line length and rhyme scheme and pick the last word of every line. Needs steps 1-4 and `lora_weights/` (already on GitHub, the training is done). A GPU is needed from 4 on. Run from inside `scripts/`, in this order:

1. `rhyme.ipynb` - pronunciations from CMUdict (plus ~30 old words like o'er that we added by hand), syllable counts, a rhyme score and "rhyme richness" (how many words rhyme with a word). The other notebooks load this one themselves with `%run`, so you don't have to run it on its own
2. `author_profiles.ipynb` - works out every named poet's line length and rhyme scheme (AABB, ABAB, ABCB or none) from their poems, and picks the 4 showcase poets (the one with the most lines in each scheme). Makes `data/author_profiles.csv`, takes a few mins
   - Why: the generator follows the profile while it writes. It sets how long each line is, which lines have to rhyme with which, and when the poem stops. The fine-tuned model can't do this by itself: the header check in `lora_weights/train_log.txt` shows it barely learned anything from the poet's name, and on its own it rhymes ~2% of the time
   - How: line length is the middle half (25th-75th percentile) of the poet's syllables per line. For the scheme, the line ends are cut into blocks of 4 and each block gets a label, trying all 4 starting points since the data has no stanza breaks. A poet rhymes if at least 30% of their blocks do, and then their scheme is the label they use most
3. `inspiring_set.ipynb` - 8 real poems per showcase poet, picked by rule and printed in the notebook. The bottom part compares them with generated poems, that needs 6 first
   - Why: the generator doesn't use these at all, they're only there to judge the output (Ritchie's criteria). Typical: do the generated poems have the poet's line length and rhyme scheme as often as the poet's real poems do? New: do they copy lines from them (5 words in a row)?
   - How: a poem gets in if it's 12-400 lines, has the poet's line length and fits the poet's scheme in at least half its blocks (80% unrhymed lines for Longfellow), then the best 8 per poet. Picked by rule so we can't cherry-pick
   - Result so far: line length matches, the scheme only gets close at λ 0, and 0% of lines are copied
4. `meaning_head.ipynb` - the GloVe word list and the "meaning head", a small network that guesses what word Qwen means at the end of a line. ~20 mins the first time. Makes `lora_weights/vocab.npz`, `data/line_end_states.pt` (~650MB, so it's quicker the next time) and `lora_weights/meaning_head.pt`. Both files in `lora_weights/` are already on GitHub, so you only need this notebook to retrain the head (it needs GloVe, see step 3)
5. `generate.ipynb` - writes the poems. For your own poem: `generate_poem("George Gordon Byron", lam=0.5, system="M")`. Any poet from `data/author_profiles.csv` works. `system="B1"` picks the line ends from Qwen's own words instead of the meaning head's, `lam` goes from 0 (only rhyme counts) to 1 (no rhyming at all). The examples at the bottom take ~7 mins, every poem's trace goes in `outputs/traces/`
6. `evaluate.ipynb` - B1 vs M at 5 values of `lam` (120 poems, ~30 mins), scores them on rhyme, fluency and copying and makes the plot. The fluency score needs a bigger model too (~3.4GB):
```
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-1.7B-Base', local_dir='model/Qwen3-1.7B-Base')"
```
Everything ends up in `outputs/` (what each file is: see "What's in the repo" at the top). It also makes the 160 poems in `outputs/final/` we pick from (~30 mins more). If it crashes just run it again, poems that are already there get skipped.

### Which poems to use in the report
- **The poems to present:** pick them from `outputs/final_poems.txt` (160 poems, all made after the method was fixed, none of them looked at while building it). Every poem has a header with poet, system, λ, seed and how many rhyme pairs rhyme, so search for e.g. `| B1 | λ 0.0` to jump to a group. Write down how many we read to pick them (e.g. "3 out of 160"), the report has to say that
- **Where the good ones are:** λ 0 and 0.25 are the ones that actually rhyme. B1 reads better, M rhymes a bit more but has more grammar slips. λ 0.75 and 1 barely rhyme, so they're only good as a "without rhyme pressure" example. Skip poems that turned into Gutenberg footnotes or glossaries (numbers, "vol. i.", lists of names)
- **Why a word got picked:** open the same poem in `outputs/final/` (same name as in the header, e.g. `B1_George_Gordon_Byron_lam0.0_seed12.json`). The trace shows for every line what Qwen wanted, what the meaning head meant and what got picked, good for a "behind the rhyme" example
- **Numbers and the plot:** `evaluate.ipynb` (Results section) for the table, `outputs/lambda_curve.png` for the plot, `outputs/sweep_scores.csv` if you want the scores of one poem
- **Best and worst by the numbers:** sort `outputs/sweep_scores.csv` by rhyme (`rhymed / pairs`) and `ppl`. These are the sweep poems (seeds 0-3), so they're for showing what goes right and wrong, not for presenting as our best work
- **Real poems next to generated ones:** the bottom of `inspiring_set.ipynb`

### Chatbot scripts
Just for playing around with the untuned model (needs step 1 and 2).
- `python chatbot/try_qwen.py` - type something and it continues it, `\n` for a new line
- `python chatbot/chat_qwen.py` - makes the base model act like a chatbot and shows tokens/sec, `reset` clears the chat

Type `quit` to exit either one.
