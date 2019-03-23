from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional
from keras.layers import *
from keras_contrib.layers import CRF


class MWEIdentifier:
    def __init__(self, language, mwe, logger):
        self.logger = logger
        self.logger.info('Initialize MWEIdentifier for %s' % language)
        self.language = language
        self.mwe = mwe

    def set_params(self, params):
        self.logger.info('Setting params...')
        self.n_units = params['n_units']
        self._dropout = params['dropout']
        self.batch_size = params['batch_size']
        self.epochs = params['epochs']

    def set_test(self):
        self.logger.info('Setting test environment...')
        self.X_tr_pos = [[self.mwe.pos2idx[w[3]] for w in s] for s in self.mwe.train_sentences]
        self.X_tr_pos = pad_sequences(maxlen=self.mwe.max_sent, sequences=self.X_tr_pos, padding="post", value=0)
        self.X_te_pos = [[self.mwe.pos2idx[w[3]] for w in s] for s in self.mwe.test_sentences]
        self.X_te_pos = pad_sequences(maxlen=self.mwe.max_sent, sequences=self.X_te_pos, padding="post", value=0)

        self.X_tr_deprel = [[self.mwe.deprel2idx[w[7]] for w in s] for s in self.mwe.train_sentences]
        self.X_tr_deprel = pad_sequences(maxlen=self.mwe.max_sent, sequences=self.X_tr_deprel, padding="post", value=0)
        self.X_te_deprel = [[self.mwe.deprel2idx[w[7]] for w in s] for s in self.mwe.test_sentences]
        self.X_te_deprel = pad_sequences(maxlen=self.mwe.max_sent, sequences=self.X_te_deprel, padding="post", value=0)

        self.X_tr_word = [[self.mwe.word2idx[w[1]] for w in s] for s in self.mwe.train_sentences]
        self.X_tr_word = pad_sequences(maxlen=self.mwe.max_sent, sequences=self.X_tr_word, padding="post", value=0)
        self.X_te_word = [[self.mwe.word2idx[w[1]] for w in s] for s in self.mwe.test_sentences]
        self.X_te_word = pad_sequences(maxlen=self.mwe.max_sent, sequences=self.X_te_word, padding="post", value=0)

        self.y = [[self.mwe.tag2idx[w[-1]] for w in s] for s in self.mwe.train_sentences]
        self.y = pad_sequences(maxlen=self.mwe.max_sent, sequences=self.y, padding="post", value=self.mwe.tag2idx["O"])
        self.y = [to_categorical(i, num_classes=self.mwe.n_tags) for i in self.y]

    def build_model(self):
        self.build_model_with_pretrained_embedding()

    def build_model_with_pretrained_embedding(self):
        self.logger.info('Building model with pretrained embedding...')
        tokens_input = Input(shape=(None,), name='words_input')
        tokens = Embedding(input_dim=self.mwe.word_embeddings.shape[0], output_dim=self.mwe.word_embeddings.shape[1],
                           weights=[self.mwe.word_embeddings],
                           trainable=False, mask_zero=True, input_length=self.mwe.max_sent, name='word_embeddings')(
            tokens_input)

        pos_embedding = np.identity(len(self.mwe.pos2idx.keys()) + 1)
        pos_input = Input(shape=(None,), name='pos_input')
        pos_tokens = Embedding(input_dim=pos_embedding.shape[0], output_dim=pos_embedding.shape[1],
                               weights=[pos_embedding],
                               trainable=False, mask_zero=True, input_length=self.mwe.max_sent,
                               name='pos_embeddings')(
            pos_input)

        deprel_embedding = np.identity(len(self.mwe.deprel2idx.keys()) + 1)
        deprel_input = Input(shape=(None,), name='deprel_input')
        deprel_tokens = Embedding(input_dim=deprel_embedding.shape[0], output_dim=deprel_embedding.shape[1],
                                  weights=[deprel_embedding],
                                  trainable=False, mask_zero=True, input_length=self.mwe.max_sent,
                                  name='deprel_embeddings')(
            deprel_input)

        inputNodes = [tokens_input, pos_input, deprel_input]
        mergeInputLayers = [tokens, pos_tokens, deprel_tokens]
        merged_input = concatenate(mergeInputLayers)

        shared_layer = merged_input
        shared_layer = Bidirectional(LSTM(self.n_units, return_sequences=True, dropout=self._dropout[0],
                                          recurrent_dropout=self._dropout[1]),
                                     name='shared_varLSTM')(shared_layer)

        output = shared_layer
        output = TimeDistributed(Dense(self.mwe.n_tags, activation=None))(output)
        crf = CRF(self.mwe.n_tags)  # CRF layer
        output = crf(output)  # output

        model = Model(inputs=inputNodes, outputs=[output])
        model.compile(optimizer="nadam", loss=crf.loss_function, metrics=[crf.accuracy])
        self.model = model

    def fit_model(self):
        self.logger.info('Fitting model...')
        self.model.fit(
            {'words_input': self.X_tr_word, 'pos_input': self.X_tr_pos, 'deprel_input': self.X_tr_deprel},
            np.array(self.y),
            batch_size=self.batch_size, epochs=self.epochs)  # , validation_split=0.2, verbose=1)

    def predict(self):
        self.logger.info('Predicting...')
        predicted_tags = []
        preds = self.model.predict([self.X_te_word, self.X_te_pos, self.X_te_deprel])
        for i in range(self.X_te_word.shape[0]):
            p = preds[i]
            p = np.argmax(p, axis=-1)
            tp = []
            for w, pred in zip(self.X_te_word[i], p):
                if w != 0:
                    tp.append(self.mwe.tags[pred])
            predicted_tags.append(tp)
        self.predicted_tags = predicted_tags

    def add_tags_to_test(self):
        self.logger.info('Tagging...')
        tags = []
        for i in range(len(self.predicted_tags)):
            for j in range(len(self.predicted_tags[i])):
                tags.append(self.predicted_tags[i][j])
            tags.append('space')
        self.mwe._test_corpus['BIO'] = tags
        self.mwe.convert_tag()
