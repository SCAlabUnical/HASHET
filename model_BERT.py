import tensorflow as tf
import constants as c
import numpy as np

SCALE_1 = 2 / 3


def cosine_distance(y_true, y_pred):
    return tf.compat.v1.losses.cosine_distance(tf.nn.l2_normalize(y_pred, 0), tf.nn.l2_normalize(y_true, 0), dim=0)


def transfer_and_fine_tune(sentences_train, sentences_test, targets_train, targets_test):
    """
        Create and train multi level perceptron with Keras API and save it.

        :param1 sentences_train: train sentences.
        :param2 sentences_test: test sentences.
        :param3 targets_train: train target embeddings.
        :param4 targets_test: test target embeddings.
    """
    from keras.callbacks import EarlyStopping
    from keras.callbacks import ModelCheckpoint
    from keras.optimizers import Adam

    print("BERT version")
    # transfer learning
    print("TRANSFER LEARNING STEP:")

    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    sequence_len = 100
    train_encodings = tokenizer(sentences_train.tolist(), truncation=True, padding='max_length',
                                max_length=sequence_len)
    test_encodings = tokenizer(sentences_test.tolist(), truncation=True, padding='max_length', max_length=sequence_len)
    import keras.layers as layers

    input1 = layers.Input(shape=(sequence_len,), dtype=tf.int32, name='input_ids')
    input2 = layers.Input(shape=(sequence_len,), dtype=tf.int32, name='attention_mask')

    input = [input1, input2]
    from transformers import TFBertModel
    encoderlayer = TFBertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    for l in encoderlayer.layers[:]:
        l.trainable = False
    encoderoutputs = encoderlayer(input)

    encoderoutput = encoderoutputs[1]
    from keras.layers.core import Dense
    n_hidden = encoderoutput.shape[1]
    h1 = Dense(int(n_hidden), activation="relu")(encoderoutput)
    h2 = Dense(int(n_hidden * SCALE_1), activation="relu")(h1)
    h3 = Dense(int(n_hidden * SCALE_1 * SCALE_1), activation="relu")(h2)

    out = Dense(len(targets_train[0]), activation='linear')(h3)
    import keras

    model = keras.Model(inputs=input, outputs=out)
    model.summary()

    # compile model
    model.compile(loss=cosine_distance, optimizer='adam', metrics=['cosine_proximity'])

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=c.LOG_LEVEL, patience=c.PATIENCE)

    best_weights_file = c.TL_MODEL_WEIGHTS_FILE_NAME
    mc = ModelCheckpoint(best_weights_file, monitor='val_loss', mode='min', verbose=c.ONE_LINE_PER_EPOCH,
                         save_best_only=True, save_weights_only=True)

    # train model testing it on each epoch
    model.fit([np.array(train_encodings["input_ids"]),
               np.array(train_encodings["attention_mask"])],
              targets_train, validation_data=(
            [np.array(test_encodings["input_ids"]),
             np.array(test_encodings["attention_mask"])],
            targets_test),
              batch_size=c.BATCH_SIZE, epochs=c.MAX_EPOCHS, verbose=c.ONE_LINE_PER_EPOCH, callbacks=[es, mc])

    # fine tuning
    print("FINE TUNING STEP:")

    # Load weights into the new model
    model.load_weights(c.TL_MODEL_WEIGHTS_FILE_NAME)

    model.trainable = True
    # compile model
    model.compile(loss=cosine_distance, optimizer=Adam(3e-5), metrics=['cosine_proximity'])

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=c.LOG_LEVEL, patience=c.PATIENCE)

    best_weights_file = c.FT_MODEL_WEIGHTS_FILE_NAME
    mc = ModelCheckpoint(best_weights_file, monitor='val_loss', mode='min', verbose=c.ONE_LINE_PER_EPOCH,
                         save_best_only=True, save_weights_only=True)
    # train model testing it on each epoch
    model.fit([np.array(train_encodings["input_ids"]),
               np.array(train_encodings["attention_mask"])],
              targets_train, validation_data=(
            [np.array(test_encodings["input_ids"]),
             np.array(test_encodings["attention_mask"])],
            targets_test),
              batch_size=c.BATCH_SIZE, epochs=c.MAX_EPOCHS, verbose=c.ONE_LINE_PER_EPOCH, callbacks=[es, mc])


def predict_top_k_hashtags(sentences, k):
    """
        Predict hashtags for input sentence embeddings (embeddings_list)

        :param1 sentences: sentences.
        :param2 k: number of hashtags to predict for each sentence.
        :returns results: list of list of (hashtag, likelihood):
    """

    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    sequence_len = 100
    sentences_encodings = tokenizer(sentences.tolist(), truncation=True, padding='max_length',
                                    max_length=sequence_len)
    import keras.layers as layers
    # Model reconstruction
    input1 = layers.Input(shape=(sequence_len,), dtype=tf.int32, name='input_ids')
    input2 = layers.Input(shape=(sequence_len,), dtype=tf.int32, name='attention_mask')

    input = [input1, input2]
    from transformers import TFBertModel
    encoderlayer = TFBertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    for l in encoderlayer.layers[:]:
        l.trainable = False
    encoderoutputs = encoderlayer(input)

    encoderoutput = encoderoutputs[1]
    from keras.layers.core import Dense
    n_hidden = encoderoutput.shape[1]
    h1 = Dense(int(n_hidden), activation="relu")(encoderoutput)
    h2 = Dense(int(n_hidden * SCALE_1), activation="relu")(h1)
    h3 = Dense(int(n_hidden * SCALE_1 * SCALE_1), activation="relu")(h2)

    out = Dense(c.LATENT_SPACE_DIM, activation='linear')(h3)
    import keras

    model = keras.Model(inputs=input, outputs=out)
    model.summary()
    from keras.optimizers import Adam

    # compile model
    model.compile(loss=cosine_distance, optimizer=Adam(3e-5),
                  metrics=['cosine_proximity'])  # Load weights into the new model
    model.load_weights(c.MODEL_WEIGHTS_FILE_NAME)

    # make probability predictions with the model
    h_list = model.predict([np.array(sentences_encodings["input_ids"]),
                            np.array(sentences_encodings["attention_mask"])])

    h_list = [np.reshape(h_vect, (len(h_vect),)) for h_vect in h_list]

    import word_embedding_model
    emb_model = word_embedding_model.load_Word2Vec_model()
    top_n_words = 1000
    result = [word_embedding_model.retain_hts(emb_model.wv.similar_by_vector(h_vect, topn=top_n_words))[:k] for h_vect
              in h_list]

    return result
