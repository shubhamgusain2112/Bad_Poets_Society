# Bad Poets Society:
## Computational Creativity Assignment 2

### Contributors
| Team | Student Number | GitHub |
| - | - | - |
| Rahul Ray | s4875710 |[LINK](https://github.com/rahulrayy) |
| Shubham Gusain | s4846761| [LINK](https://github.com/shubhamgusain2112/)

### Sources 
| Use | Name | Link |
| - | - | - |
| LLM | Qwen 3 Base (0.6B) | [LINK](https://huggingface.co/Qwen/Qwen3-0.6B-Base) | 
| Pronunciation Database | CMUDict | [LINK](https://github.com/Alexir/CMUdict) |
| Rhyme Database | PeRDict Database | [LINK](https://github.com/scrosseye/PeRDict-database) | 
| Syllable Database | English phonetic and syllable count dictionary | [LINK](https://www.kaggle.com/datasets/schwartstack/english-phonetic-and-syllable-count-dictionary) | 
| Poetry Database | DanFosing/public-domain-poetry | [LINK](https://huggingface.co/datasets/DanFosing/public-domain-poetry) | 

### Other downloads
Also used by the project. Like the sources above, they go in `data/`, which is not in the repo.

| Name | Why | Link |
| - | - | - |
| Gutenberg Poetry Corpus | Main training data: 3.1M lines of public-domain poetry from 1,191 books | [LINK](https://huggingface.co/datasets/biglam/gutenberg-poetry-corpus) |
| Project Gutenberg catalogue | Finds the author of each book in the corpus | [LINK](https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv) |
| GloVe word vectors (300d) | The "meaning" space the model predicts each next word in | [LINK](https://github.com/RaRe-Technologies/gensim-data/releases/tag/glove-wiki-gigaword-300) |

### Chatbot scripts
Both scripts use the untuned Qwen 3 Base model. Type `quit` to exit either one.

| File | What it does |
| - | - |
| `chatbot/try_qwen.py` | Type any text and the model continues it, e.g. the first line of a poem. Type `\n` for a line break. Being a base model, it continues text rather than answering questions. |
| `chatbot/chat_qwen.py` | Makes the base model act like a chatbot. The conversation is written as a `User:` / `Assistant:` transcript that starts with two example exchanges, and each reply stops when the model begins a new `User:` line. It remembers the conversation (`reset` clears it) and shows tokens per second after each answer. |

**Setup.** The model weights are not in the repo (`model/` is in `.gitignore`), so download them once:
```
pip install torch transformers huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B-Base', local_dir='model/Qwen3-0.6B-Base')"
```
Then run `python chatbot/try_qwen.py` or `python chatbot/chat_qwen.py`. 

### Data notebooks
The notebooks in `scripts/` prepare the training data. They read the downloads from `data/` and write their results there too (none of it is in the repo). Open them from the `scripts/` folder.

| Notebook | What it does | Output |
| - | - | - |
| `scripts/author_counts.ipynb` | Counts how many lines each poet has in the two corpora (each distinct line once), to choose the authors the model is conditioned on | `data/author_counts.csv` |
| `scripts/clean_pdp.ipynb` | Cleans public-domain-poetry: drops very short poems, prose, translations and duplicates, fixes the OCR "0" → "O" slip, tidies author names | `data/pdp_clean.parquet` (34,245 poems) |
| `scripts/clean_gutenberg.ipynb` | Cleans the Gutenberg Poetry Corpus: drops translations, prose and duplicate books and lines, fixes small text problems, and finds each book's author, checked against public-domain-poetry | `data/gutenberg_clean.parquet` (869 books) |

Run `clean_pdp.ipynb` before `clean_gutenberg.ipynb`, which uses its output.

**Setup.** `pip install pandas pyarrow jupyter`. The notebooks expect these downloads in `data/`:
- `gutenberg-poetry-corpus/data/train-00000-of-00001-fa9fb9e1f16eed7e.parquet`
- `public-domain-poetry/poems.json`
- `gutenberg-catalog/pg_catalog.csv`