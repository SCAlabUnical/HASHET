import tensorflow as tf
import tensorflow_hub as hub
import constants as c

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

    from keras import Sequential
    from keras.callbacks import EarlyStopping
    from keras.callbacks import ModelCheckpoint
    from keras.layers.core import Dense, Activation
    from keras.models import model_from_json
    from keras.optimizers import Adam

    print("GUSE version")

    # transfer learning
    print("TRANSFER LEARNING STEP:")

    model = Sequential()
    encoderlayer = hub.KerasLayer(c.GUSE_PATH, input_shape=[], dtype=tf.string, trainable=False)
    model.add(encoderlayer)
    n_hidden = encoderlayer.output_shape[1]
    model.add(Dense(int(n_hidden)))
    model.add(Activation('relu'))
    model.add(Dense(int(n_hidden * SCALE_1)))
    model.add(Activation('relu'))
    model.add(Dense(int(n_hidden * SCALE_1 * SCALE_1)))
    model.add(Activation('relu'))
    model.add(Dense(len(targets_train[0]), activation='linear'))

    # compile model
    model.compile(loss=cosine_distance, optimizer='adam', metrics=['cosine_proximity'])

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=c.LOG_LEVEL, patience=c.PATIENCE)

    best_weights_file = c.TL_MODEL_WEIGHTS_FILE_NAME
    mc = ModelCheckpoint(best_weights_file, monitor='val_loss', mode='min', verbose=c.ONE_LINE_PER_EPOCH,
                         save_best_only=True)
    # train model testing it on each epoch
    model.fit(sentences_train, targets_train, validation_data=(sentences_test, targets_test),
              batch_size=c.BATCH_SIZE, epochs=c.MAX_EPOCHS, verbose=c.ONE_LINE_PER_EPOCH, callbacks=[es, mc])
    # serialize model to JSON
    model_json = model.to_json()
    with open(c.TL_MODEL_JSON_FILE_NAME, "w") as json_file:
        json_file.write(model_json)
    print("model saved")

    # fine tuning
    print("FINE TUNING STEP:")
    # Model reconstruction from JSON file
    with open(c.TL_MODEL_JSON_FILE_NAME, 'r') as f:
        json = f.read()
    model = model_from_json(json, custom_objects={'KerasLayer': hub.KerasLayer})

    # Load weights into the new model
    model.load_weights(c.TL_MODEL_WEIGHTS_FILE_NAME)

    model.trainable = True
    # compile model
    model.compile(loss=cosine_distance, optimizer=Adam(3e-5), metrics=['cosine_proximity'])

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=c.LOG_LEVEL, patience=c.PATIENCE)

    best_weights_file = c.FT_MODEL_WEIGHTS_FILE_NAME
    mc = ModelCheckpoint(best_weights_file, monitor='val_loss', mode='min', verbose=c.ONE_LINE_PER_EPOCH,
                         save_best_only=True)
    # train model testing it on each epoch
    model.fit(sentences_train, targets_train, validation_data=(sentences_test, targets_test),
              batch_size=c.BATCH_SIZE, epochs=c.MAX_EPOCHS, verbose=c.ONE_LINE_PER_EPOCH, callbacks=[es, mc])
    # serialize model to JSON
    model_json = model.to_json()
    with open(c.FT_MODEL_JSON_FILE_NAME, "w") as json_file:
        json_file.write(model_json)
    print("model saved")


def predict_top_k_hashtags(sentences, k):
    """
        Predict hashtags for input sentence embeddings (embeddings_list)

        :param1 sentences: sentences.
        :param2 k: number of hashtags to predict for each sentence.
        :returns results: list of list of (hashtag, likelihood):
    """
    from keras.models import model_from_json

    # Model reconstruction from JSON file
    with open(c.MODEL_JSON_FILE_NAME, 'r') as f:
        json = f.read()
    model = model_from_json(json, custom_objects={'KerasLayer': hub.KerasLayer})

    # Load weights into the new model
    model.load_weights(c.MODEL_WEIGHTS_FILE_NAME)

    # make probability predictions with the model
    h_list = model.predict(sentences)

    import numpy as np

    h_list = [np.reshape(h_vect, (len(h_vect),)) for h_vect in h_list]

    import word_embedding_model

    emb_model = word_embedding_model.load_Word2Vec_model()
    top_n_words = 1000
    result = [word_embedding_model.retain_hts(emb_model.wv.similar_by_vector(h_vect, topn=top_n_words))[:k] for h_vect
              in h_list]

    return result
