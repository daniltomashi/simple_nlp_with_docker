# Simple NLP project with docker
Root folder contains of Docker files(to run it in Docker), main.py (which is file with API) and utils

# utils folder
This folder is most essential(here is main logic). Three files is included:

1. `text_processing.py` (logic to preprocess text)
2. `traditional_ml.py` (logic to preprocess data, train model and save model, vectorizer, label_encoder into .pkl files to use it)
3. `lifecycle_manager.py` (manager to properly use model, vectorizer, label_encoder with posibility to clear cache, load them and so on)

# How to use it
### First step
First step is to train your own model through `traditional_ml.py` file (you should do it only once). To do it, folder `data` should exist with dataset in it(it should be one `.csv` file). Do it from root folder.
```
python3 utils/traditional_ml.py
```

### Second step
Then run in your terminal next commands (or do it through Docker interface)

```
docker build -t fastapi_app .
docker run -d -p 8000:80 fastapi_app
```

### Third step
After this documentation could be found on `http://127.0.0.1:8000/docs` (Swagger) and prediction can be maid through next curl

```
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"text": "Hello, it's my initial message"}'
```

Or with python next way

```
import requests

requests.post('http:127.0.0.1:8000/predict', json={"text": "Hello, it's my initial message"}).json()
```
