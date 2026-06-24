import joblib
from tensorflow import keras

model_name = "main"
model = keras.models.load_model(f'MLP/MODELS/{model_name}.h5')
scaler = joblib.load("MLP/MLP.scaler.gz")

