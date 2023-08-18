import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, add, concatenate
from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.preprocessing import text, sequence
from keras.utils import pad_sequences
from gensim.models import KeyedVectors

import warnings
warnings.filterwarnings('ignore')

EMBEDDING_FILES = [
    '../input/gensim-embeddings-dataset/crawl-300d-2M.gensim',
    '../input/gensim-embeddings-dataset/glove.840B.300d.gensim'
]
NUM_MODELS = 2
BATCH_SIZE = 512
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
EPOCHS = 4
MAX_LEN = 220
IDENTITY_COLUMNS = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
]
AUX_COLUMNS = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
TEXT_COLUMN = 'comment_text'
TARGET_COLUMN = 'target'
CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'

train_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
test_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')

x_train = train_df[TEXT_COLUMN].astype(str)
y_train = train_df[TARGET_COLUMN].values
y_aux_train = train_df[AUX_COLUMNS].values
x_test = test_df[TEXT_COLUMN].astype(str)

for column in IDENTITY_COLUMNS + [TARGET_COLUMN]:
    train_df[column] = np.where(train_df[column] >= 0.5, True, False)
    
tokenizer = text.Tokenizer(filters=CHARS_TO_REMOVE, lower=False) 
# bu kütüphane yazıları istenildiği gibi düzenlemeye yarıyor.Örneğin bütün harfleri küçült gibi.
tokenizer.fit_on_texts(list(x_train) + list(x_test))

x_train = tokenizer.texts_to_sequences(x_train) # metindeki kelimeleri sayısallaştırıyor.
x_test = tokenizer.texts_to_sequences(x_test)

x_train = pad_sequences(x_train, maxlen=MAX_LEN) 
# matris içindeki düzenlemeleri yapıyor. En uzun olana göre ayarlıyor boyutu. Diğerlerinin boş yerlerine 0 veriyor.
x_test = pad_sequences(x_test, maxlen=MAX_LEN)

sample_weights = np.ones(len(x_train), dtype=np.float32)
#print(sample_weights)
sample_weights += train_df[IDENTITY_COLUMNS].sum(axis=1)
#print(sample_weights)
sample_weights += train_df[TARGET_COLUMN] * (~train_df[IDENTITY_COLUMNS]).sum(axis=1)
#print(sample_weights)
sample_weights += (~train_df[TARGET_COLUMN]) * train_df[IDENTITY_COLUMNS].sum(axis=1) * 5
#print(sample_weights)
sample_weights /= sample_weights.mean()
#print(sample_weights)

def build_matrix(word_index, path):
    embedding_index = KeyedVectors.load(path, mmap='r')
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        for candidate in [word, word.lower()]: 
            # bir kelimenin hem normal hali hem de bütün harfleri küçültülmüş hali diğer veri setinde var mı diye kontrol ediyor. Eğer varsa değiştirme işlemi yapılıyor.
            # muhtemelen sayıya çevrilen kelimelerde swimmign gibi kelimeler var onları swim gibi veriye dönüştürdü diye tahmin ediyorum.
            if candidate in embedding_index:
                embedding_matrix[i] = embedding_index[candidate]
                break
    return embedding_matrix

# burada sıkıntılı kelimeleri düzetmek için başka bir veri tablosu kullanılarak cümlelerin fazlalığı azaltıldı.
# örneğin swiming -> swiming
embedding_matrix = np.concatenate(
    [build_matrix(tokenizer.word_index, f) for f in EMBEDDING_FILES], axis=-1)
# "tokenizer.word_index" değeri hangi kelimelere hangi sayı atandığını gösteren bir değişken döndürüyor.
checkpoint_predictions = []
weights = []

def build_model(embedding_matrix, num_aux_targets):
    words = Input(shape=(None,))
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x) # çift yönlü LSTM için. 
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

    hidden = concatenate([
        GlobalMaxPooling1D()(x),
        GlobalAveragePooling1D()(x),
    ])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    result = Dense(1, activation='sigmoid')(hidden)
    aux_result = Dense(num_aux_targets, activation='sigmoid')(hidden)
    
    model = Model(inputs=words, outputs=[result, aux_result]) # modeli burda birleştiriyor gibi. Modelin başlangıcını bitişini gösteriyor
    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model

for model_idx in range(NUM_MODELS): # bu for neden anlamadım. 2 kez eğitiyor ama bir değişiklik yapmadan 2 kez.
    model = build_model(embedding_matrix, y_aux_train.shape[-1]) # "y_aux_train" içinde hakaretin ne yönde olduğu. Örnek cisiyetçi. Bunun özelliklerin sayısı(6).
    for global_epoch in range(EPOCHS):
        model.fit(
            x_train,
            [y_train, y_aux_train],
            batch_size=BATCH_SIZE,
            epochs=1,
            verbose=2,
            sample_weight=[sample_weights.values, np.ones_like(sample_weights)]
        )
        checkpoint_predictions.append(model.predict(x_test, batch_size=2048)[0].flatten())
        weights.append(2 ** global_epoch)
        
predictions = np.average(checkpoint_predictions, weights=weights, axis=0)

submission = pd.DataFrame.from_dict({
    'id': test_df.id,
    'prediction': predictions
})
