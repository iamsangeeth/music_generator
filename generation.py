from preprocess import *
from music_utils import *
from keras.models import  load_model, Model
from keras.layers import Reshape, LSTM,Input, Dense, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical

chords, abstract_grammars = get_musical_data('data/original_metheny.mid')
corpus, tones, tones_indices, indices_tones = get_corpus_data(abstract_grammars)
N_tones = len(set(corpus))
n_a = 64
x_initializer = np.zeros((1, 1, 78))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))
x_initializer = np.zeros((1, 1, 78))
a_initializer = np.zeros((1, 64))
c_initializer = np.zeros((1, 64))

def data_processing(corpus, values_indices, m = 60, Tx = 30):
    N_values = len(set(corpus))
    np.random.seed(0)
    X = np.zeros((m, Tx, N_values), dtype=np.bool)
    Y = np.zeros((m, Tx, N_values), dtype=np.bool)
    for i in range(m):
#         for t in range(1, Tx):
        random_idx = np.random.choice(len(corpus) - Tx)
        corp_data = corpus[random_idx:(random_idx + Tx)]
        for j in range(Tx):
            idx = values_indices[corp_data[j]]
            if j != 0:
                X[i, j, idx] = 1
                Y[i, j-1, idx] = 1
    
    Y = np.swapaxes(Y,0,1)
    Y = Y.tolist()
    return np.asarray(X), np.asarray(Y), N_values 

def load_music_utils():
    chords, abstract_grammars = get_musical_data('data/original_metheny.mid')
    corpus, tones, tones_indices, indices_tones = get_corpus_data(abstract_grammars)
    N_tones = len(set(corpus))
    X, Y, N_tones = data_processing(corpus, tones_indices, 60, 30)   
    return (X, Y, N_tones, indices_tones)

def djmodel(Tx, n_a, n_values):
	X = Input(shape=(Tx, n_values))
	a0 = Input(shape=(n_a,), name='a0')
	c0 = Input(shape=(n_a,), name='c0')
	a = a0
	c = c0
	outputs = []
	for t in range(Tx):
		x = Lambda(lambda x: X[:,t,:])(X)
		x = reshapor(x)
		a, _, c = LSTM_cell(x, initial_state=[a, c])
		out = densor(a)
		outputs.append(out)
	model = Model(inputs=[X, a0, c0], outputs=outputs)
	return model

def music_inference_model(LSTM_cell, densor, n_values=78, n_a=64, Ty=100):
	x0 = Input(shape=(1,n_values))
	a0 = Input(shape=(n_a,), name='a0')
	c0 = Input(shape=(n_a,), name='c0')
	a = a0
	c = c0
	x = x0
	outputs = []
	for t in range(Ty):
		a, _, c = LSTM_cell(x, initial_state=[a,c])
		out = densor(a)
		outputs.append(out)
		x = Lambda(one_hot)(out)
	inference_model = Model(inputs=[x0, a0, c0], outputs=outputs)
	return inference_model

def predict_and_sample(inference_model, x_initializer = x_initializer, a_initializer = a_initializer, c_initializer = c_initializer):
	pred = inference_model.predict([x_initializer, a_initializer, c_initializer])
	indices = np.argmax(pred, axis=-1)
	results = to_categorical(indices, num_classes=78)
	return results, indices

def generate_music(inference_model, corpus = corpus, abstract_grammars = abstract_grammars, tones = tones, tones_indices = tones_indices, indices_tones = indices_tones, T_y = 10, max_tries = 1000, diversity = 0.5):
	out_stream = stream.Stream()
	curr_offset = 0.0
	num_chords = int(len(chords) / 3)
	for i in range(1, num_chords):
		curr_chords = stream.Voice()
		for j in chords[i]:
			curr_chords.insert((j.offset % 4), j)
		_, indices = predict_and_sample(inference_model)
		indices = list(indices.squeeze())
		pred = [indices_tones[p] for p in indices]
		predicted_tones = 'C,0.25 '
		for k in range(len(pred) - 1):
			predicted_tones += pred[k] + ' ' 
	predicted_tones +=  pred[-1]
	predicted_tones = predicted_tones.replace(' A',' C').replace(' X',' C')
	predicted_tones = prune_grammar(predicted_tones)
	sounds = unparse_grammar(predicted_tones, curr_chords)
	sounds = prune_notes(sounds)
	sounds = clean_up_notes(sounds)
	for m in sounds:
		out_stream.insert(curr_offset + m.offset, m)
	for mc in curr_chords:
		out_stream.insert(curr_offset + mc.offset, mc)
	curr_offset += 4.0
	out_stream.insert(0.0, tempo.MetronomeMark(number=130))
	mf = midi.translate.streamToMidiFile(out_stream)
	mf.open("my_music.midi", 'wb')
	mf.write()
	print("Your generated music is saved in output/my_music.midi")
	mf.close()
	return out_stream

X, Y, n_values, indices_values = load_music_utils()

n_a = 64 

reshapor = Reshape((1, 78))                        
LSTM_cell = LSTM(n_a, return_state = True)         
densor = Dense(n_values, activation='softmax')     

model = djmodel(Tx = 30 , n_a = 64, n_values = 78)

opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

m = 60
a0 = np.zeros((m, n_a))
c0 = np.zeros((m, n_a))


model.fit([X, a0, c0], list(Y), epochs=100)

inference_model = music_inference_model(LSTM_cell, densor, n_values=78, n_a=64, Ty=50)

x_initializer = np.zeros((1, 1, 78))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))


results, indices = predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer)

out_stram = generate_music(inference_model)

