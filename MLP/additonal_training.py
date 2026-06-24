import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.client import device_lib
from math import log10
#{"R": [R], "D": [D],"H": [H], "F":[log10(f)],"RP": [RP]})
DATA = pd.read_csv("MLP/DATA/data.csv")
DATA.pop("Unnamed: 0")

ROW_NUM = len(DATA.index)
TRAINING_NUM = int(ROW_NUM / 10 * 7)
VALIDATION_NUM = int(ROW_NUM / 20 * 3 )

TRAINING_DATA = DATA.iloc[ :TRAINING_NUM,:]
VALIDATION_DATA = DATA.iloc[TRAINING_NUM:TRAINING_NUM+VALIDATION_NUM,:]
TEST_DATA = DATA.iloc[TRAINING_NUM+VALIDATION_NUM:,:]
TRAINING_VAL = pd.DataFrame(TRAINING_DATA.pop('RP'))
VALIDATION_VAL = pd.DataFrame(VALIDATION_DATA.pop('RP'))
TEST_VAL = pd.DataFrame(TEST_DATA.pop('RP'))
DATA.pop('RP')

min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
min_max_scaler.fit(DATA)
joblib.dump(min_max_scaler, "MLP/MLP.scaler.gz") 

TRAINING_DATA = min_max_scaler.transform(TRAINING_DATA)
TRAINING_DATA = pd.DataFrame(TRAINING_DATA)
VALIDATION_DATA = min_max_scaler.transform(VALIDATION_DATA)
VALIDATION_DATA = pd.DataFrame(VALIDATION_DATA)
TEST_DATA = min_max_scaler.transform(TEST_DATA)
TEST_DATA = pd.DataFrame(TEST_DATA)

epochs=500
batch=32

for i in range(1,2):

    model_name = f"candidate{i}"
    model = keras.models.load_model(f'MLP/MODELS/{model_name}.h5')
    history = model.fit(TRAINING_DATA, TRAINING_VAL, epochs=epochs, batch_size=batch, validation_data=(VALIDATION_DATA, VALIDATION_VAL))
    model.save(f"MLP/MODELS/{model_name}.h5")
    scores = model.evaluate(TEST_DATA, TEST_VAL, batch_size=20)
    plt.plot([10*log10(loss) for loss in history.history['loss'][4:]])
    plt.plot([10*log10(loss) for loss in history.history['val_loss'][4:]])
    plt.title(f'model loss {scores}')
    plt.ylabel('loss dB')
    plt.xlabel('epoch')
    plt.legend(['TRAIN', 'VALID'], loc='upper left')
    plt.savefig(f"MLP/TRAINING_PROCESS/{model_name}.png")