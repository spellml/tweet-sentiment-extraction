export KAGGLE_USERNAME=residentmario KAGGLE_KEY=ea4163444bc3b21c1b810b218c385038
kaggle competitions download tweet-sentiment-extraction
unzip tweet-sentiment-extraction.zip -d /mnt/tweet-sentiment-extraction/
rm tweet-sentiment-extraction.zip

kaggle datasets download abhishek/bert-base-uncased
unzip bert-base-uncased.zip -d /mnt/bert-base-uncased/
rm bert-base-uncased.zip
