# tweet-sentiment-extraction <a href="https://web.spell.ml/workspace_create?workspaceName=tweet-sentiment-extraction&githubUrl=https%3A%2F%2Fgithub.com%2Fspellml%2Ftweet-sentiment-extraction&pip=kaggle&envVars=KAGGLE_USERNAME%3DYOUR_USERNAME,KAGGLE_KEY=YOUR_KEY"><img src=https://spell.ml/badge.svg height=20px/></a>

A neural network for extracting sentiments from tweet data. Uses `torch`, `transformers`, and BERT. Based on [BERT Base Uncased Using PyTorch](https://www.kaggle.com/abhishek/bert-base-uncased-using-pytorch/) and uses data from the [Tweet Sentiment Extraction](https://www.kaggle.com/c/tweet-sentiment-extraction) competition on Kaggle.

To run code and notebooks in a Spell workspace:

```bash
spell jupyter --lab \
  --github-url https://github.com/spellml/tweet-sentiment-extraction.git \
  --env KAGGLE_USERNAME=YOUR_USERNAME \
  --env KAGGLE_KEY=YOUR_KEY \
  tweet-sentiment-extraction
```

To execute the training scripts in a Spell run:

```bash
spell run \
  --machine-type t4 \
  --github-url https://github.com/spellml/tweet-sentiment-extraction.git \
  --pip transformers --pip tokenizers --pip kaggle \
  --env KAGGLE_USERNAME=YOUR_USERNAME \
  --env KAGGLE_KEY=YOUR_KEY \
  --tensorboard-dir /spell/tensorboards/model_1 \
  "chmod +x /spell/scripts/download_data.sh /spell/scripts/upgrade_env.sh; /spell/scripts/download_data.sh; /spell/scripts/upgrade_env.sh; python /spell/models/model_1.py"
```

```bash
spell run \
  --machine-type t4 \
  --github-url https://github.com/spellml/tweet-sentiment-extraction.git \
  --pip transformers --pip tokenizers --pip kaggle \
  --env KAGGLE_USERNAME=YOUR_USERNAME \
  --env KAGGLE_KEY=YOUR_KEY \
  --tensorboard-dir /spell/tensorboards/model_2 \
  "chmod +x /spell/scripts/download_data.sh; chmod +x /spell/scripts/upgrade_env.sh; /spell/scripts/download_data.sh; /spell/scripts/upgrade_env.sh; python /spell/models/model_2.py"
```
