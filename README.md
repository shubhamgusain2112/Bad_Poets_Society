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
3. `author_counts.ipynb` - optional, just counts how many lines each poet has. Makes `data/author_counts.csv`

### Chatbot scripts
Just for playing around with the untuned model (needs step 1 and 2).
- `python chatbot/try_qwen.py` - type something and it continues it, `\n` for a new line
- `python chatbot/chat_qwen.py` - makes the base model act like a chatbot and shows tokens/sec, `reset` clears the chat

Type `quit` to exit either one.
