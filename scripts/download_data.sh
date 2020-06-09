export KAGGLE_USERNAME=YOUR_USERNAME KAGGLE_KEY=YOUR_KEY
kaggle competitions download tweet-sentiment-extraction
unzip tweet-sentiment-extraction.zip -d /mnt/tweet-sentiment-extraction/
rm tweet-sentiment-extraction.zip

kaggle datasets download abhishek/bert-base-uncased
unzip bert-base-uncased.zip -d /mnt/bert-base-uncased/
rm bert-base-uncased.zip
