import sys
import os
from keras.applications import inception_v3
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Input, LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Merge, Activation, concatenate
from keras.preprocessing import image, sequence
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
import pickle
import numpy as np
import tensorflow as tf

# sys.path.append('/Users/esror/PycharmProjects/caption_app')

EMBEDDING_DIM = 128


class CaptionGenerator():

    def __init__(self, main_dir):
        self.max_cap_len = None
        self.vocab_size = None
        self.index_word = None
        self.word_index = None
        self.total_samples = None
        self.encoded_images = pickle.load( open( os.path.join(main_dir,"encoded_images.p"), "rb" ) )
        self.working_dir = main_dir
        self.variable_initializer()

    def variable_initializer(self):
        df = pd.read_csv(os.path.join(self.working_dir, 'xray_train_dataset.txt'), delimiter='\t')
        nb_samples = df.shape[0]
        iter = df.iterrows()
        caps = []
        for i in range(nb_samples):
            x = iter.__next__()
            caps.append(x[1][1])

        self.total_samples=0
        for text in caps:
            self.total_samples+=len(text.split())-1
        print ("Total samples : "+str(self.total_samples))
        
        words = [txt.split() for txt in caps]
        unique = []
        for word in words:
            unique.extend(word)

        unique = list(set(unique))
        self.vocab_size = len(unique)

        if os.path.exists(os.path.join(self.working_dir, 'word_index.p')):
            print("Load dictionaries")
            self.word_index = pickle.load(open(os.path.join(self.working_dir, 'word_index.p'), 'rb'))
            self.index_word = pickle.load(open(os.path.join(self.working_dir, 'index_word.p'), 'rb'))
        else:
            self.word_index = {}
            self.index_word = {}
            for i, word in enumerate(unique):
                self.word_index[word] = i
                self.index_word[i] = word
            print("Save dictionaries")
            pickle.dump(self.word_index, open(os.path.join(self.working_dir, 'word_index.p'), 'wb'))
            pickle.dump(self.index_word, open(os.path.join(self.working_dir, 'index_word.p'), 'wb'))

        max_len = 0
        for caption in caps:
            if(len(caption.split()) > max_len):
                max_len = len(caption.split())
        self.max_cap_len = max_len
        print ("Vocabulary size: "+str(self.vocab_size))
        print ("Maximum caption length: "+str(self.max_cap_len))
        print ("Variables initialization done!")


    def data_generator(self, batch_size = 512):
        partial_caps = []
        next_words = []
        images = []
        print ("Generating data...")
        gen_count = 0
        df = pd.read_csv(os.path.join(self.working_dir, 'xray_train_dataset.txt'), delimiter='\t')
        df = df.sample(frac=1) # shuffle
        nb_samples = df.shape[0]
        iter = df.iterrows()
        caps = []
        imgs = []
        for i in range(nb_samples):
            x = iter.__next__()
            caps.append(x[1][1])
            imgs.append(x[1][0])


        total_count = 0
        while 1:
            image_counter = -1
            for text in caps:
                image_counter+=1
                current_image = self.encoded_images[imgs[image_counter]]
                for i in range(len(text.split())-1):
                    total_count+=1
                    partial = [self.word_index[txt] for txt in text.split()[:i+1]]
                    partial_caps.append(partial)
                    next = np.zeros(self.vocab_size)
                    next[self.word_index[text.split()[i+1]]] = 1
                    next_words.append(next)
                    images.append(current_image)

                    if total_count>=batch_size:
                        next_words = np.asarray(next_words)
                        images = np.asarray(images)
                        partial_caps = sequence.pad_sequences(partial_caps, maxlen=self.max_cap_len, padding='post')
                        total_count = 0
                        gen_count+=1
                        print ("yielding count: "+str(gen_count))
                        yield [[images, partial_caps], next_words]
                        partial_caps = []
                        next_words = []
                        images = []
        
    def load_image(self, path):
        img = image.load_img(path, target_size=(224,224))
        x = image.img_to_array(img)
        return np.asarray(x)


    # def create_model__(self, ret_model = False):
    #     #base_model = VGG16(weights='imagenet', include_top=False, input_shape = (224, 224, 3))
    #     #base_model.trainable=False
    #     image_model = Sequential()
    #     #image_model.add(base_model)
    #     #image_model.add(Flatten())
    #     image_model.add(Dense(EMBEDDING_DIM, input_dim= 4096, activation='relu'))
    #
    #     image_model.add(RepeatVector(self.max_cap_len))
    #
    #     lang_model = Sequential()
    #     lang_model.add(Embedding(self.vocab_size, 10, input_length=self.max_cap_len))
    #     lang_model.add(LSTM(10,return_sequences=True))
    #     lang_model.add(TimeDistributed(Dense(EMBEDDING_DIM)))
    #
    #     model = Sequential()
    #     model.add(Merge([image_model, lang_model], mode='concat'))
    #     model.add(LSTM(50,return_sequences=False))
    #     model.add(Dense(self.vocab_size))
    #     model.add(Activation('softmax'))
    #
    #     print ("Model created!")
    #
    #     if(ret_model==True):
    #         return model
    #
    #     model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    #     plot_model(model, to_file='../Models/model.png', show_shapes=True)
    #     return model

    def create_model(self, ret_model = False):
        image_input = Input(shape=(4096,), name='image_input')
        image_model = Dense(EMBEDDING_DIM, input_dim=4096, activation='relu')(image_input)
        image_model = RepeatVector(self.max_cap_len)(image_model)

        lang_input = Input(shape=(self.max_cap_len,), name='lang_input')
        lang_model = Embedding(self.vocab_size, 10, input_length=self.max_cap_len)(lang_input)
        lang_model = LSTM(10, return_sequences=True)(lang_model)
        lang_model = TimeDistributed(Dense(EMBEDDING_DIM))(lang_model)

        concat = concatenate([image_model, lang_model])
        x = LSTM(50, return_sequences=False)(concat)
        x = Dense(self.vocab_size)(x)
        main_output = Activation(tf.nn.softmax, name='main_output')(x)
        model = Model(inputs=[image_input, lang_input], outputs=[main_output])
        print("Model created!")

        if (ret_model == True):
            return model

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        plot_model(model, to_file='../Models/model.png', show_shapes=True)
        return model

    def get_word(self,index):
        return self.index_word[index]