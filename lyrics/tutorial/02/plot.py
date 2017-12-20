# ----- MODEL ----- #
from keras.layers import Dense, Input, GlobalMaxPool1D
from keras.layers import Conv1D, MaxPool1D, Embedding
from keras.models import Model

MAX_SEQUENCE_LENGTH = 1000

def model():
    embedding_layer = Embedding(200000,100)
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPool1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPool1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(10, activation='softmax')(x)

    model = Model(sequence_input, preds)
    return model

# ----- PLOT ----- #
from keras.utils import plot_model

model = model()
plot_model(model, to_file=('model.png'))
