# Simple NLP project with docker
Root folder contains of Docker files(to run it in Docker), main.py (which is file with API) and utils

# utils folder
This folder is most essential(here is main logic). Three files is included:

1. `text_processing.py` (logic to preprocess text)
2. `traditional_ml.py` (logic to preprocess data, train model and save model, vectorizer, label_encoder into .pkl files to use it)
3. `lifecycle_manager.py` (manager to properly use model, vectorizer, label_encoder with posibility to clear cache, load them and so on)
