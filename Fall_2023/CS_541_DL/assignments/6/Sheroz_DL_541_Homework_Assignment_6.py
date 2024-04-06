# =============================
# importing libraries
# =============================

import tensorflow as tf
from tensorflow.keras.layers import Input,Dense,Embedding,GRU
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
import numpy as np
import unicodedata
import re
import matplotlib.pyplot as plt
%matplotlib inline

# =============================
# problem_set_1
# =============================

def scaled_dot_product_attention(query, key, value):
    """
    Compute the scaled dot product attention.
    Arguments:
    query: tensor with shape (batch_size, input_sequence_length, query_dim)
    key: tensor with shape (batch_size, input_sequence_length, key_dim)
    value: tensor with shape (batch_size, input_sequence_length, value_dim)
    
    Returns:
    output: tensor with shape (batch_size, input_sequence_length, value_dim)
    """
    if query.ndim > 3:
        query = np.reshape(query, (query.shape[0], query.shape[1], -1))
    if key.ndim > 3:
        key = np.reshape(key, (key.shape[0], key.shape[1], -1))
    if value.ndim > 3:
        value = np.reshape(value, (value.shape[0], value.shape[1], -1))

    dot_product = np.matmul(query, key.transpose(0, 2, 1))
    scaled_dot_product = dot_product / np.sqrt(query.shape[-1])
    attention_scores = np.exp(scaled_dot_product)
    attention_weights = attention_scores / np.sum(attention_scores, axis=-1, keepdims=True)
    output = np.matmul(attention_weights, value)
    return output

def split_heads(x, num_heads):
    """
    Compute Split Heads
    Arguments:
    x: tensor of shape (batch_size, input_sequence_length, num_heads * query_dim/key_dim/value_dim)
    num_heads: integer
    
    Returns:
    output: tensor with shape (batch_size, input_sequence_length, num_heads, -1)
    """
    batch_size, input_sequence_length, concatenated_dim = x.shape
    head_dim = concatenated_dim // num_heads
    reshaped_tensor = np.reshape(x, (batch_size, input_sequence_length, num_heads, head_dim))
    return reshaped_tensor

def multi_head_scaled_attention(query, key, value, num_heads, W_q, W_k, W_v):
    """
    Compute the multi-head attention.
    Arguments:
    query: tensor with shape (batch_size, input_sequence_length, query_dim)
    key: tensor with shape (batch_size, input_sequence_length, key_dim)
    value: tensor with shape (batch_size, input_sequence_length, value_dim)
    num_heads: integer
    W_q: matrix with shape (query_dim, num_heads * query_dim)
    W_k: matrix with shape (key_dim, num_heads * key_dim)
    W_v: matrix with shape (value_dim, num_heads * value_dim) 
    
    Returns:
    output: tensor with shape (batch_size, input_sequence_length, num_heads * value_dim)
    """
    projected_query = np.matmul(query, W_q)
    projected_key = np.matmul(key, W_k)
    projected_value = np.matmul(value, W_v)
    query_heads = split_heads(projected_query, num_heads)
    key_heads = split_heads(projected_key, num_heads)
    value_heads = split_heads(projected_value, num_heads)
    attention_heads = scaled_dot_product_attention(query_heads, key_heads, value_heads)
    concatenated_attention = np.reshape(attention_heads, (query.shape[0], query.shape[1], -1))    
    return concatenated_attention

# Testing out with following input values
input_seq_len=5 # Maximum length of the input sequence
d_q=8           # Dimensionality of the linearly projected queries
d_k=8           # Dimensionality of the linearly projected keys
d_v=8           # Dimensionality of the linearly projected values
batch_size=64   # Batch size from the training process
num_heads=8     # Number of self-attention heads
query = np.random.randn(batch_size, input_seq_len, d_q) # generating input query matrix
key = np.random.randn(batch_size, input_seq_len, d_k)   # generating input key matrix
value = np.random.randn(batch_size, input_seq_len, d_v) # generating input value matrix
W_q = np.random.randn(d_q, num_heads*d_q) # for generating num head projection matrices for queries
W_k = np.random.randn(d_k, num_heads*d_k) # for generating num head projection matrices for keys
W_v = np.random.randn(d_v, num_heads*d_v) # for generating num head projection matrices for values 

# Testing code of scaled dot product attention
attention=scaled_dot_product_attention(query, key, value)
print("Scaled Dot Product Attention:", attention)
print("Scaled Dot Product Attention Shape:", attention.shape)

# Testing code of multi head scaled attention
multi_head_attention=multi_head_scaled_attention(query, key, value, num_heads, W_q, W_k, W_v)
print("Multi Head Scaled Attention", multi_head_attention)
print("Multi Head Scaled Attention Shape:", multi_head_attention.shape)

# =============================
# problem_set_2
# =============================

start_token = 'sos'
end_token = 'eos'
oov_token = 'unk'
BATCH_SIZE = 32
EPOCHS = 31
GRU_UNITS = 256

def txt_pre_processing(txt:str)->str:
  txt = txt.lower().strip()
  txt = unicodedata.normalize('NFKD',txt).encode('ascii','ignore').decode('utf-8')
  txt = re.sub(pattern=r'[^\sa-z\d\.\?\!\,]',repl='',string=str(txt))
  txt = re.sub(pattern=r'([\.\?\!\,])',repl=r' \1 ',string=str(txt))
  txt = re.sub(pattern=r'\s+',repl=r' ',string=str(txt)).strip()
  txt = start_token + ' ' + txt + ' ' + end_token
  return txt

def load_data() -> tuple:
  context : list = list()
  target : list = list()
  with open(file='./eng-fra.txt',mode='r',encoding='utf-8') as inputstream:
    for text in inputstream:
      lines = text.replace('\n','').replace('\r','').split('\t')
      eng_txt = lines[0]
      fr_txt = lines[1]
      eng_txt = txt_pre_processing(txt=eng_txt)
      fr_txt = txt_pre_processing(txt=fr_txt)
      context.append(eng_txt)
      target.append(fr_txt)
  context = np.array(context)
  target = np.array(target)
  return context,target

eng_sentences,fr_sentences = load_data()
shuffling_indices = np.arange(len(eng_sentences))
np.random.shuffle(shuffling_indices)
eng_sentences = eng_sentences[shuffling_indices]
fr_sentences = fr_sentences[shuffling_indices]

max_seq_length = max([len(x.split(' ')) for x in eng_sentences])

eng_tokenizer = Tokenizer()
eng_tokenizer.fit_on_texts(eng_sentences)
eng_vocab_words = eng_tokenizer.word_index.keys()
eng_tokenizer.word_index[oov_token] = len(eng_tokenizer.word_index) + 1
eng_vocab_size = len(eng_tokenizer.word_index) + 1

fr_tokenizer = Tokenizer()
fr_tokenizer.fit_on_texts(fr_sentences)
fr_vocab_size = len(fr_tokenizer.word_index) + 1

eng_sequences = eng_tokenizer.texts_to_sequences(eng_sentences)
fr_sequences = fr_tokenizer.texts_to_sequences(fr_sentences)

eng_sequences = pad_sequences(eng_sequences,maxlen=max_seq_length,padding='post')
fr_sequences = pad_sequences(fr_sequences,maxlen=max_seq_length,padding='post')

split_80_20: int = int(eng_sequences.shape[0]*0.8)
X_train,y_train = eng_sequences[:split_80_20,:],fr_sequences[:split_80_20]
X_test,y_test = eng_sequences[split_80_20:,:],fr_sequences[split_80_20:]
y_train = to_categorical(y_train,num_classes=fr_vocab_size)
y_test = to_categorical(y_test,num_classes=fr_vocab_size)

# =============================
# Load Glove Embedding
# =============================
glove_embeddings_mapping : dict = dict()
glove_embeddings_size = 50
with open(file='./glove.6B.50d.txt',mode='r',encoding='utf-8') as inputstream:
    for text in inputstream:
        text = text.split()
        glove_embeddings_mapping[text[0]] = np.asarray(text[1:],dtype='float32')

# =============================
# Glove Matrix
# =============================
glove_embedding_matrix = np.zeros(shape=(eng_vocab_size,glove_embeddings_size))
for txt,idx in eng_tokenizer.word_index.items():
    if txt in glove_embeddings_mapping:
        glove_embedding_matrix[idx] = glove_embeddings_mapping[txt]

# =============================
# enc + dec
# =============================
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(input_dim=eng_vocab_size,output_dim=glove_embeddings_size,weights=[glove_embedding_matrix],trainable=False)(encoder_inputs)
encoder_gru = GRU(GRU_UNITS,return_state=True)
encoder_outputs,encoder_state = encoder_gru(encoder_embedding)
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(input_dim=fr_vocab_size,output_dim=glove_embeddings_size)(decoder_inputs)
decoder_gru = GRU(GRU_UNITS,return_sequences=True,return_state=True)
decoder_outputs,_ = decoder_gru(decoder_embedding,initial_state=encoder_state)
decoder_dense = Dense(fr_vocab_size,activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
model = Model([encoder_inputs,decoder_inputs],decoder_outputs)
model.compile(optimizer=Adam(learning_rate=3e-5,epsilon=1e-07,),loss='categorical_crossentropy',metrics=['accuracy'])

# =============================
# Testing Translation
# =============================
test_sentence = X_test[-1].reshape(1,-1)
translations_tracking : dict = dict()
history_translations_tracking : dict = {
    'loss' : list(),
    'val_loss' : list(),
    'accuracy' : list(),
    'val_accuracy' : list(),
}
history_tracking : dict = {
    'loss' : list(),
    'val_loss' : list(),
    'accuracy' : list(),
    'val_accuracy' : list(),
}

for epoch in range(EPOCHS):
    history = model.fit([X_train,X_train],y_train,epochs=1,batch_size=BATCH_SIZE,validation_data=([X_test,X_test],y_test))
    history_tracking['loss'].append(history.history['loss'])
    history_tracking['val_loss'].append(history.history['val_loss'])
    history_tracking['accuracy'].append(history.history['accuracy'])
    history_tracking['val_accuracy'].append(history.history['val_accuracy'])
    if epoch == 0 or epoch % 5 == 0:
        curr_trans = model.predict([test_sentence,test_sentence],batch_size=1)
        translations_tracking[epoch] = {
                                        'correct' : eng_tokenizer.sequences_to_texts([test_sentence[0]]),
                                        'translated' : fr_tokenizer.sequences_to_texts([np.argmax(curr_trans,axis=-1)[0]]),
                                        }
        history_translations_tracking['loss'].append(history.history['loss'])
        history_translations_tracking['val_loss'].append(history.history['val_loss'])
        history_translations_tracking['accuracy'].append(history.history['accuracy'])
        history_translations_tracking['val_accuracy'].append(history.history['val_accuracy'])
    else:
        continue

# =============================
# Plotting training and the testing loss for 0th and multiple of 5 epoch.
# =============================
fig,axs = plt.subplots(2,1,figsize=(10,13))
axs[0].plot(history_translations_tracking['loss'])
axs[0].plot(history_translations_tracking['val_loss'])
axs[0].title.set_text('Enc + Dec Training Loss vs Validation Loss')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].legend(['Train','Val'])
axs[1].plot(history_translations_tracking['accuracy'])
axs[1].plot(history_translations_tracking['val_accuracy'])
axs[1].title.set_text('Enc + Dec Training Accuracy vs Validation Accuracy')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Accuracy')
axs[1].legend(['Train','Val'])

# =============================
# Plotting training and the testing loss for each epoch.
# =============================
fig,axs = plt.subplots(2,1,figsize=(10,13))
axs[0].plot(history_tracking['loss'])
axs[0].plot(history_tracking['val_loss'])
axs[0].title.set_text('Enc + Dec Training Loss vs Validation Loss')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].legend(['Train','Val'])
axs[1].plot(history_tracking['accuracy'])
axs[1].plot(history_tracking['val_accuracy'])
axs[1].title.set_text('Enc + Dec Training Accuracy vs Validation Accuracy')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Accuracy')
axs[1].legend(['Train','Val'])
