# Data

Nothing in this folder is tracked. The code reads these files from here (`data_dir` is `../data` in every config):

| File | What it is | How it is made |
|---|---|---|
| `corpus.json` | 20 Newsgroups, cleaned, train and test split | `experiments/data/corpus.py` |
| `lda_octis.joblib`, `octis_words.json`, `octis/` | the LDA teacher (20 topics) and its vocabulary | `python -m experiments.data.teacher --config configs/paper/teacher.json` |
| `translations.json` | 1,000 test posts (first 150 words) in English, German, French, Spanish and Chinese | `experiments/data/translations.py`, Marian `Helsinki-NLP/opus-mt-en-{de,fr,es,zh}` |
| `train_de.json`, `lda_german.joblib` | German training posts and the German LDA baseline | `experiments/baselines/lda_german.py` |

The extensions that train on all five languages also read `extensions/multilingual_decode/data/train_{fr,es,zh}.json`,
made by `extensions/multilingual_decode/translate_train.py`.

A re-translation will not match the released files byte for byte (Marian versions and GPU kernels differ), so scores
can move by a few documents. To reproduce the reported numbers exactly, use the released files.
