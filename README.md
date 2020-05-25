# tweet-sentiment-extraction

A neural network for extracting sentiments from tweet data. Uses `torch`, `transformers`, and BERT. Based on [BERT Base Uncased Using PyTorch](https://www.kaggle.com/abhishek/bert-base-uncased-using-pytorch/) and uses data from the [Tweet Sentiment Extraction](https://www.kaggle.com/c/tweet-sentiment-extraction) competition on Kaggle.

[Link to the workspace](https://web.spell.run/spellrun/workspaces/106).

To run on Spell (requires being in the `spellrun` org for the GH integration):

```bash
prodspell run \
  --machine-type V100 \
  --github-url https://github.com/ResidentMario/tweet-sentiment-extraction.git \
  --pip transformers --pip tokenizers --pip tqdm --pip kaggle \
  "chmod +x /spell/scripts/download_data.sh; /spell/scripts/download_data.sh; python /spell/models/model_1.py"
```
