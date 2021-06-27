import getopt
import sys
import tensorflow as tf
import math
import os
import json
import pandas as pd
import file_utils
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, precision_score, f1_score, average_precision_score
from tensorflow.keras import backend as K
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

class Metrics(tf.keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []


    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
        val_targ = self.model.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(" — val_f1: % f — val_precision: % f — val_recall % f" % (_val_f1, _val_precision, _val_recall))
        return


def prepare_predictions(config, ids, predictions, labels):
    prediction_output = []

    if config["use_multiple_outputs"]:
        predictions = predictions[-1]

    true_positives = {0: 0, 1:0}
    total_expected = {0: 0, 1:0}
    total_predicted = {0: 0, 1: 0}

    binary_predictions = list()

    for i in range(0, len(labels)):
        predicted_probs = predictions[i]
        expected = labels[i]
        if(len(predicted_probs.shape)) == 1:
            predicted_class = 1 if predicted_probs[0] >= 0.5 else 0
        else:
            predicted_class = 1 if predicted_probs[1] >= 0.5 else 0

        binary_predictions.append(predicted_class)


        if expected == predicted_class:
            true_positives[expected] +=1
        total_expected[expected] +=1
        total_predicted[predicted_class] += 1

        l = {"id": str(ids[i]), "prediction": str(predicted_class), "label": str(labels[i]), "probs": predicted_probs.tolist()}
        prediction_output.append(json.dumps(l))

    recall_hate = (true_positives[1] / total_expected[1]) if total_expected[1] > 0 else 0
    recall_not_hate = (true_positives[0] / total_expected[0]) if total_expected[0] > 0 else 0

    precision_hate = (true_positives[1] / total_predicted[1]) if total_predicted[1] > 0 else 0
    precision_not_hate = (true_positives[0] / total_predicted[0]) if total_predicted[0] > 0 else 0

    f1_hate = ((2 * precision_hate * recall_hate) / (precision_hate + recall_hate)) if ( precision_hate + recall_hate) > 0 else 0
    f1_not_hate = ((2 * precision_not_hate * recall_not_hate) / (precision_not_hate + recall_not_hate)) if (precision_not_hate + recall_not_hate) > 0 else 0

    accuracy = (true_positives[0] + true_positives[1]) / (total_expected[0] + total_expected[1])

    binary_predictions = np.array(binary_predictions)
    average_precision = average_precision_score(binary_predictions, labels)
    f1 = f1_score(binary_predictions, labels, average='binary')
    macro_f1 = f1_score(binary_predictions, labels, average='macro')
    weighted_f1 = f1_score(binary_predictions, labels, average='weighted')
    recall = recall_score(binary_predictions, labels, average='binary')
    precision = precision_score(binary_predictions, labels, average='binary')


    score_output = {"accuracy": accuracy, "average_precision":average_precision,  "weighted_f1":weighted_f1, "f1":f1,  "macro_f1":macro_f1,  "recall":recall, "precision":precision, "Offensive": {"precision": precision_hate, "recall": recall_hate, "f1": f1_hate, "support": total_expected[1]},
                    "NotOffensive": {"precision": precision_not_hate, "recall": recall_not_hate, "f1": f1_not_hate, "support": total_expected[0]}}

    return prediction_output, score_output

def get_modalities_as_string(config):
    modalities = ''
    if "tweet_text" in config["modalities"]:
        modalities += ' tweet_text: clip'

    if "image_text" in config["modalities"]:
        modalities += ' image_text: clip'

    if "image" in config["modalities"]:
        modalities += ' image: clip'

    return modalities.strip()

def create_learning_rate_scheduler(config):
    max_learning_rate, end_learning_rate, total_epoch_count, warmup_epoch_count, learning_rate_decay = config["max_learning_rate"], config["end_learning_rate"], config['epochs'], config["warmup_epoch_count"], config["learning_rate_decay"]


    def old_lr_scheduler(epoch):
        if epoch < warmup_epoch_count:
            alpha = (max_learning_rate / warmup_epoch_count) * (epoch + 1)
        else:
            alpha = max_learning_rate * math.exp(
                math.log(end_learning_rate / max_learning_rate) * (epoch - warmup_epoch_count + 1) / (
                            total_epoch_count - warmup_epoch_count + 1))
        return float(alpha)

    def exponential_decay(epoch):
        if epoch < warmup_epoch_count:
            return max_learning_rate
        else:
            return float(max_learning_rate * tf.math.exp(0.1 * (10 - epoch)))

    def step_decay(epoch):
        if epoch < warmup_epoch_count:
            return max_learning_rate

        factor = 0.25
        drop_epoch = 3.0

        exp = np.floor((1 + epoch) / drop_epoch)
        alpha = max_learning_rate * (factor ** exp)

        return float(alpha)

    def polynomial_decay(epoch):
        if epoch < warmup_epoch_count:
            return max_learning_rate

        power = 1.0
        decay = (1 - (epoch / float(total_epoch_count))) ** power
        alpha = max_learning_rate * decay

        return float(alpha)

    if learning_rate_decay == "polynomial":
        learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(polynomial_decay, verbose=1)
    elif learning_rate_decay == "step":
        learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(step_decay, verbose=1)
    elif learning_rate_decay == "exponential":
        learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay, verbose=1)
    else:
        learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(old_lr_scheduler, verbose=1)

    return learning_rate_scheduler


def custom_f1(y_true, y_pred):
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = TP / (Positives + K.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = TP / (Pred_Positives + K.epsilon())
        return precision

    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)

    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def get_model(config):
    """Creates a classification model."""
    inputs = []
    outputs = []

    joined_modalities = []

    if "image_text" in config["modalities"]:
        image_text_input = tf.keras.layers.Input(shape=(512,), dtype='float32', name="image_text_clip")
        inputs.append(image_text_input)
        image_text_bn = tf.keras.layers.BatchNormalization()(image_text_input)
        joined_modalities.append(image_text_bn)

    if "image" in config["modalities"]:
        image_input = tf.keras.layers.Input(shape=(512,), dtype='float32', name="image_clip")
        inputs.append(image_input)
        image_bn = tf.keras.layers.BatchNormalization()(image_input)
        joined_modalities.append(image_bn)

    if len(config["modalities"]) > 1:
        concatenated = tf.keras.layers.concatenate(joined_modalities)
        concatenated = tf.keras.layers.BatchNormalization()(concatenated)
    else:
        concatenated = joined_modalities[0]

    out = tf.keras.layers.Dense(units=512*len(config["modalities"]), activation="relu", name='fc_4')(concatenated)
    out = tf.keras.layers.Dropout(0.5)(out)

    if len(config["modalities"]) == 3:
        out = tf.keras.layers.Dense(units=1024, activation="relu", name='fc_1')(out)
        out = tf.keras.layers.Dropout(0.5)(out)

    if len(config["modalities"]) > 1:
        out = tf.keras.layers.Dense(units=512, activation="relu", name='fc_2')(out)
        out = tf.keras.layers.Dropout(0.5)(out)

    out = tf.keras.layers.Dense(units=256, activation="relu", name='fc_3')(out)
    out = tf.keras.layers.Dropout(0.5)(out)
    last_layer_output = tf.keras.layers.Dense(units=1, activation="sigmoid", name='output_label_multimodal')(out)
    outputs.append(last_layer_output)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    if config["optimizer"] == "sgd":
        optimizer = tf.keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9)
    elif config["optimizer"] == "rmsprop":
        optimizer = tf.keras.optimizers.RMSProp()
    elif config["optimizer"] == "adagrad":
        optimizer = tf.keras.optimizers.Adagrad()
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(loss='binary_crossentropy', optimizer= "adam", metrics=['acc'])

    # model.compile(optimizer=optimizer,
    #               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               metrics=[tf.keras.metrics.PrecisionAtRecall(0.6)])

    # model.compile(optimizer=optimizer,
    #               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")])

    model.summary()

    return model

def embed_labels(data: list):
    y = list()
    label_to_index = {"not_offensive": 0, "slight": 1, "very_offensive":1, "hateful_offensive": 1}

    for label in data:
        y.append(label_to_index[label])
    return np.array(y)

def load_split(config, df, split_name):
    ids = df["image_name"].tolist()
    labels = df["offensive"].tolist()

    X, y = {}, {}

    data = np.array(np.load(config['base_dir'] + '/resources/Memotion7k/memotion_clip_features_'+split_name+'.npy'))

    if "image_text" in config["modalities"]:
        X['image_text_clip'] = data[:, 512:1024]
    if "image" in config["modalities"]:
        X['image_clip'] = data[:, 0:512]

    y['output_label_multimodal'] = embed_labels(labels)

    return X, y, ids


def load_dataset(config):
    train_df = pd.read_json(config['base_dir'] + '/resources/Memotion7k/train_over_sampled.txt', lines=True, orient='string')
    test_df = pd.read_json(config['base_dir'] + '/resources/Memotion7k/test.txt', lines=True, orient='string')
    valid_df = pd.read_json(config['base_dir'] + '/resources/Memotion7k/valid.txt', lines=True, orient='string')

    X_valid, y_valid, y_valid_ids = load_split(config, valid_df, 'valid')
    X_train, y_train, y_train_ids = load_split(config, train_df, 'train_over_sampled')
    X_test, y_test, y_test_ids = load_split(config, test_df, 'test')

    return X_train, y_train, y_train_ids, X_valid, y_valid, y_valid_ids, X_test, y_test, y_test_ids

def train(config):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    for i in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[i], True)

    # Load the train/valid/test datasets
    print("Loading the data")
    X_train, y_train, y_train_ids, X_valid, y_valid, y_valid_ids, X_test, y_test, y_test_ids = load_dataset(config)

    print("Training input file shapes")
    for k in X_train:
        print('\t'+k + " shape: " + str(X_train[k].shape))

    model = get_model(config)

    print('Training with config: '+ str(config))

    if not file_utils.path_exists(config["base_dir"]+ "/" "memotion_clip_results"):
        file_utils.create_folder(config["base_dir"]+ "/" "memotion_clip_results")

    now = datetime.now()
    model_dir_path = now.strftime("%d-%m-%Y %H:%M:%S").replace(" ", "_")

    if not file_utils.path_exists(config["base_dir"]+ "/" "memotion_clip_results/"+model_dir_path):
        file_utils.create_folder(config["base_dir"]+ "/" "memotion_clip_results/"+model_dir_path)

    model_dir_path = config["base_dir"]+ "/" "memotion_clip_results/"+model_dir_path

    output_label = "output_label_multimodal"
    train_accuracy_labels = ['acc']
    valid_accuracy_labels = ['acc']
    train_loss_labels = ['loss']
    valid_loss_labels = ['val_loss']


    model_check_point_callback = tf.keras.callbacks.ModelCheckpoint(model_dir_path + '/best_model.hdf5',
                                                                    save_best_only=True, save_weights_only = False,
                                                                    monitor=valid_accuracy_labels[-1], mode='max')

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(patience=config["epoch_patience"],
                                                               restore_best_weights=True, monitor=valid_accuracy_labels[-1])

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=model_dir_path+"/logs")

    callbacks = [early_stopping_callback,
                 tensorboard_callback,
                 model_check_point_callback]

    if config["use_lr_scheduler"]:
        learning_rate_callback = create_learning_rate_scheduler(config)
        callbacks.append(learning_rate_callback)

    weight_for_0 = (1 / 2289) * (5943 / 2.0)
    weight_for_1 = (1 / 3654) * (5943 / 2.0)

    class_weights = {0: weight_for_0, 1: weight_for_1}
    class_weights = None

    history = model.fit(X_train, y_train,
              validation_data=(X_valid, y_valid),
              batch_size=config['batch_size'],
              shuffle=True,
              epochs=config['epochs'],
              callbacks=callbacks)



    # predict on validation data
    predictions = model.predict(X_train, batch_size=config['batch_size'])
    prediction_output, score_output = prepare_predictions(config, y_train_ids, predictions, y_train[output_label])

    print('Training Accuracy: ', score_output['accuracy'])
    print('Training macro-f1: ', score_output['macro_f1'])
    print('Training f1: ', score_output['f1'])

    # save prediction score, predictions
    file_utils.save_string_to_file(json.dumps(score_output),
                                   model_dir_path + '/training_prediction_score.json')
    file_utils.save_list_to_file(prediction_output,
                                 model_dir_path + '/training_predictions.jsonl')

    # predict on validation data
    predictions = model.predict(X_valid, batch_size=config['batch_size'])

    prediction_output, score_output = prepare_predictions(config, y_valid_ids, predictions, y_valid[output_label])

    print('Validation Accuracy: ', score_output['accuracy'])
    print('Validation macro-f1: ', score_output['macro_f1'])
    print('Validation f1: ', score_output['f1'])

    # save prediction score, predictions
    file_utils.save_string_to_file(json.dumps(score_output),
                                   model_dir_path + '/validation_prediction_score.json')
    file_utils.save_list_to_file(prediction_output,
                                 model_dir_path + '/validation_predictions.jsonl')

    # predict on test data
    predictions = model.predict(X_test, batch_size=config['batch_size'])

    prediction_output, score_output = prepare_predictions(config, y_test_ids, predictions, y_test[output_label])

    print('Test accuracy: ', score_output['accuracy'])
    print('Test macro-f1: ', score_output['macro_f1'])
    print('Test f1: ', score_output['f1'])

    # save the model
    model.save(model_dir_path + "/model.h5")
    # save prediction score, predictions
    file_utils.save_string_to_file(json.dumps(score_output),
                                   model_dir_path + '/test_prediction_score.json')
    file_utils.save_list_to_file(prediction_output, model_dir_path + '/test_predictions.jsonl')

    # save model test accuracy to a file
    model_results = {}
    if file_utils.path_exists(config["base_dir"] + '/' + "memotion_clip_results/model_results.txt"):
        content = file_utils.read_file_to_set(config["base_dir"] + '/' + "memotion_clip_results/model_results.txt")
        for c in content:
            t = c.split("\t")
            model_results[t[0]] = float(t[1])

    # add the current trained model with the accuracy
    modalities = get_modalities_as_string(config)
    model_results[model_dir_path + '_' + modalities] = float(score_output['accuracy'])

    # sort and save
    sorted_model_results = sorted(model_results, reverse=True, key=model_results.get)
    model_results_output = ""
    for model_name in sorted_model_results:
        model_results_output += model_name + "\t" + str(model_results[model_name]) + "\n"

    # save the content
    file_utils.save_string_to_file(model_results_output, config["base_dir"] + "/" + "memotion_clip_results/model_results.txt")


    N = len(history.epoch)
    plt.style.use("ggplot")
    plt.figure()
    for train_loss_label in train_loss_labels:
        plt.plot(np.arange(1, N+1), history.history[train_loss_label], label=train_loss_label)
    for val_loss_label in valid_loss_labels:
        plt.plot(np.arange(1, N+1), history.history[val_loss_label], label=val_loss_label)
    for train_acc_label in train_accuracy_labels:
        plt.plot(np.arange(1, N+1), history.history[train_acc_label], label=train_acc_label)
    for val_acc_label in valid_accuracy_labels:
        plt.plot(np.arange(1, N+1), history.history[val_acc_label], label=val_acc_label)
    plt.title("Training Loss and Accuracy on MMHS Dataset, " +config['optimizer'])
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(model_dir_path + "/history.png")
    plt.close()

    if config["use_lr_scheduler"]:
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(1, N+1), history.history["lr"], label="Learning_rate")
        plt.title("Learning Rate/Epochs, Decay:" +config['learning_rate_decay'])
        plt.xlabel("Epoch #")
        plt.ylabel("Learning Rate")
        plt.legend(loc="lower left")
        plt.savefig(model_dir_path + "/learning_rate.png")
        plt.close()

        plt.style.use("ggplot")
        plt.figure()
        plt.plot(history.history["lr"], history.history["loss"], label="Loss/Learning Rate")
        plt.title("Loss/Learning Rate, Optimizer:" + config['optimizer'] + ', Decay:' + config['learning_rate_decay'])
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.legend(loc="lower left")
        plt.savefig(model_dir_path + "/loss_learning_rate.png")
        plt.close()

if __name__ == "__main__":
    argv = (sys.argv[1:])
    config_path = 'clip_memotion_training_config.json'
    try:
        opts, args = getopt.getopt(argv, "hc:o:")
    except getopt.GetoptError:
        print('train_clip_model.py -c <config_path>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('train_clip_model.py -c <config_path>')
            sys.exit()
        elif opt  == "-c":
            config_path = arg

    if config_path != '':
        with open(config_path) as json_file:
            config = json.load(json_file)
            train(config)
    else:
        print('train_clip_model.py -c <config_path>')
