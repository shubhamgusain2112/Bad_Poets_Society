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

### Scripts
Both scripts use the untuned Qwen 3 Base model. Type `quit` to exit either one.

| File | What it does |
| - | - |
| `try_qwen.py` | Type any text and the model continues it, e.g. the first line of a poem. Type `\n` for a line break. Being a base model, it continues text rather than answering questions. |
| `chat_qwen.py` | Makes the base model act like a chatbot. The conversation is written as a `User:` / `Assistant:` transcript that starts with two example exchanges, and each reply stops when the model begins a new `User:` line. It remembers the conversation (`reset` clears it) and shows tokens per second after each answer. |

**Setup.** The model weights are not in the repo (`model/` is in `.gitignore`), so download them once:
```
pip install torch transformers huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B-Base', local_dir='model/Qwen3-0.6B-Base')"
```
Then run `python try_qwen.py` or `python chat_qwen.py`. 