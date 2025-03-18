import joblib
from pathlib import Path
from utils.text_processing import TextProcessing


class ModelLifecycleManager:
    def __init__(self, model_path: str, vectorizer_path:str, label_encoder_path:str, cache: bool = True):
        """
        Initialize the ModelLifecycleManager.
        
        :param model_path: Path to the model file.
        :param cache: Whether to cache the loaded model.
        """
        self.model_path = Path(model_path)
        self.label_encoder_path = Path(label_encoder_path)
        self.vectorizer_path = Path(vectorizer_path)
        self.cache = cache
        self._model = None
        self._is_loaded = False
    
    def _load_model(self):
        """
        Load the model from the specified path.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

        # Simulating the loading of a model (using joblib for a pickled model here)
        self._model = joblib.load(self.model_path)
        self._is_loaded = True
        print(f"Model loaded from {self.model_path}")

    def _load_vectorizer(self):
        """
        Load the vectorizer from the specified path.
        """
        if not self.vectorizer_path.exists():
            raise FileNotFoundError(f"Vectorizer file not found at {self.vectorizer_path}")

        # Simulating the loading of a model (using joblib for a pickled model here)
        self._vectorizer = joblib.load(self.vectorizer_path)
        self._is_loaded = True
        print(f"Vectorizer loaded from {self.vectorizer_path}")

    def _load_label_encoder(self):
        """
        Load the label encoder from the specified path.
        """
        if not self.label_encoder_path.exists():
            raise FileNotFoundError(f"Label encoder file not found at {self.label_encoder_path}")

        # Simulating the loading of a model (using joblib for a pickled model here)
        self._label_encoder = joblib.load(self.label_encoder_path)
        self._is_loaded = True
        print(f"Label encoder loaded from {self.label_encoder_path}")

    def get_tools(self):
        """
        Lazy-load all tools
        
        :return: Loaded model
        """
        if not self._is_loaded:
            print("Lazy loading tools...")
            self._load_model()
            self._load_vectorizer()
            self._load_label_encoder()
        
        return self._model, self._vectorizer, self._label_encoder

    def clear_cache(self):
        """
        Clear the cached model. This will unload the model from memory.
        """
        self._model = None
        self._vectorizer = None
        self._label_encoder = None
        self._is_loaded = False
        print("Model cache cleared.")

    def reload_tools(self):
        """
        Reload the model (e.g., in case the model file has changed).
        """
        self.clear_cache()  # First, clear the old model from memory
        self._load_model()
        self._load_vectorizer()
        self._load_label_encoder()
        print("Tools reloaded.")

    def is_model_loaded(self):
        """
        Check if the model is currently loaded in memory.
        """
        return self._is_loaded
    
    def make_prediction(self, example):
        pass



# Example usage
if __name__ == "__main__":
    model_path, vectorizer_path, label_encoder_path = "tools/model.pkl", "tools/vectorizer.pkl", "tools/label_encoder.pkl"
    model_manager = ModelLifecycleManager(model_path, vectorizer_path, label_encoder_path)

    # Lazy load the model when needed
    model, vectorizer, label_encoder = model_manager.get_tools()

    # Check if the model is loaded
    if model_manager.is_model_loaded():
        print("Model is successfully loaded.")

    # Clear the cached model after usage
    model_manager.clear_cache()

    # Reload the model if necessary
    model_manager.reload_tools()