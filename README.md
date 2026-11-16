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

The later steps (rhymes and the meaning head) also need these two, not needed for the notebooks below yet:
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

### Chatbot scripts
Just for playing around with the untuned model (needs step 1 and 2).
- `python chatbot/try_qwen.py` - type something and it continues it, `\n` for a new line
- `python chatbot/chat_qwen.py` - makes the base model act like a chatbot and shows tokens/sec, `reset` clears the chat

Type `quit` to exit either one.
