import getopt
import numpy as np
from datetime import datetime
import pandas as pd
import json
import sys
import os
import tensorflow as tf
import file_utils
import operator
import matplotlib.pyplot as plt
import random
import string
from sklearn.metrics import recall_score, precision_score, f1_score, average_precision_score

def random_string_generator(len=5, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for x in range(len))

def read_file_to_set(file_path):
    content = set()
    if os.path.isfile(file_path):
        with open(file_path, "r") as file:
            for l in  file.readlines():
                content.add(l.strip())
    return content
def get_modalities(config: dict):
    output = ''
    output += "__text_models__"
    for k in config['text_models']:
        output+=k+"__+"

    output += "__external_text_models__"
    for k in config['external_text_models']:
        output += config["all_models_names"][k] + "__+"

    output += "__hatewords_models__"
    for k in config['hatewords_models']:
        output += config["all_models_names"][k] + "__+"

    output += "__image-text_relations__"
    for k in config['image_text_relations']:
        output+=k+"_"

    output += "__visual_sentiment__"
    for k in config['visual_sentiment']:
        output += k + "_"

    output += "__image_models__"
    for k in config['image_models']:
        output += config["all_models_names"][k] + "__+"
    return output
def prepare_predictions(config, ids, predictions, labels):
    prediction_output = []


    true_positives = {0: 0, 1:0}
    total_expected = {0: 0, 1:0}
    total_predicted = {0: 0, 1: 0}

    binary_predictions = list()

    for i in range(0, len(labels)):
        predicted_probs = predictions[i]
        expected = labels[i]
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
    recall = recall_score(binary_predictions, labels, average='binary')
    precision = precision_score(binary_predictions, labels, average='binary')


    score_output = {"accuracy": accuracy, "average_precision":average_precision, "f1":f1,  "macro_f1":macro_f1,  "recall":recall, "precision":precision, "Offensive": {"precision": precision_hate, "recall": recall_hate, "f1": f1_hate, "support": total_expected[1]},
                    "NotOffensive": {"precision": precision_not_hate, "recall": recall_not_hate, "f1": f1_not_hate, "support": total_expected[0]}}

    return prediction_output, score_output
def path_exists(dir_path):
    return os.path.exists(dir_path)
def create_folder(dir_path):
    os.mkdir(dir_path)
    pass
def save_list_to_file(input_list: list, file_path):
    file = open(file_path, 'w')
    file.write("\n".join(str(item) for item in input_list))
    file.close()

def save_string_to_file(text, file_path):
    file = open(file_path, 'w')
    file.write(text)
    file.close()

def prepare_feature_importance_scores(all_feature_labels:dict, importance_scores: dict):

    sorted_importance_scores = sorted(importance_scores.items(), reverse=True, key=operator.itemgetter(1))

    output = ""
    for i in sorted_importance_scores:
        (feature_index, weight) = i
        label = all_feature_labels[feature_index]
        weight = importance_scores[feature_index]
        output += label +": "+str(weight)+"\n"
    return output

def embed_text_with_hate_words(data: list, hate_words: list, is_binary= False):
    x = list()
    for text in data:
        # tokenize
        tokens = text.split(' ')
        multihot_encoding_array = np.zeros(len(hate_words), dtype=int)

        has_hateword = False
        for t in tokens:
            if t in hate_words:
                index = hate_words.index(t)
                multihot_encoding_array[index] = 1
                has_hateword = True

        if is_binary:
            label = 1 if has_hateword else 0
            x.append(label)
        else:
            x.append(multihot_encoding_array)
    return np.array(x)

def encode_labels(data: list):
    y = list()
    label_to_index = {"NotHate": 0, "Hate": 1}

    for label in data:
        y.append(label_to_index[label])
    return np.array(y)

def get_model(config, text_feature_size, image_feature_size):
    """Creates a classification model."""
    inputs = []
    outputs = []

    joined_modalities = []

    if text_feature_size > 0:
        text_input = tf.keras.layers.Input(shape=(text_feature_size,), dtype='float32', name="text_features")
        inputs.append(text_input)
        text_bn = tf.keras.layers.BatchNormalization()(text_input)
        joined_modalities.append(text_bn)

    if image_feature_size > 0:
        image_input = tf.keras.layers.Input(shape=(image_feature_size,), dtype='float32', name="image_features")
        inputs.append(image_input)
        image_bn = tf.keras.layers.BatchNormalization()(image_input)
        joined_modalities.append(image_bn)


    ## MULTIMODAL PART
    if len(joined_modalities) > 1:
        concat_embedding = tf.keras.layers.concatenate(joined_modalities)
        fusion_layer_output = concat_embedding

        fusion_layer_size = max(text_feature_size + image_feature_size, 100)
        counter = 1
        while fusion_layer_size > 256:

            fusion_layer_output = tf.keras.layers.Dense(fusion_layer_size, activation='relu',
                                                        name="fusion_layer_" + str(counter))(fusion_layer_output)
            drop_out = tf.keras.layers.Dropout(0.5)(fusion_layer_output)

            if counter == 1:
                adapted_layer_size = np.power(2, int(np.log2(fusion_layer_size)))
                if adapted_layer_size == fusion_layer_size:
                    fusion_layer_size /= 2
                else:
                    fusion_layer_size = adapted_layer_size
            else:
                # decrease by half
                fusion_layer_size /= 2

            counter += 1

            fusion_layer_output = drop_out
    else:
        fusion_layer_output = joined_modalities[0]
        layer_size = max(text_feature_size + image_feature_size, 100)
        fusion_layer_output = tf.keras.layers.Dense(layer_size, activation='relu',
                                                    name="prelast_layer")(fusion_layer_output)
        fusion_layer_output = tf.keras.layers.Dropout(0.5)(fusion_layer_output)


    last_layer_output = tf.keras.layers.Dense(units=2, activation="softmax", name='output_label')(fusion_layer_output)
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

    # model.compile(loss='binary_crossentropy', optimizer= "adam", metrics=['acc'])

    # model.compile(optimizer=optimizer,
    #               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               metrics=[tf.keras.metrics.PrecisionAtRecall(0.6)])

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")])

    model.summary()

    return model

def train(config: dict):
    # merge all features into a single train/test data

    train_df = pd.read_json(config["base_dir"] + '/' + config['train_split_path'], lines=True, orient='string')
    valid_df = pd.read_json(config["base_dir"] + '/' + config['valid_split_path'], lines=True, orient='string')
    test_df = pd.read_json(config["base_dir"] + '/' + config['test_split_path'], lines=True, orient='string')

    if len(config["image_models"]) > 0:
        print('Loading image data')

        valid_image_data = np.load(config['base_dir'] + "/resources/image_predictions/valid_image_predictions_v3.npy")
        test_image_data = np.load(config['base_dir'] + "/resources/image_predictions/test_image_predictions_v3.npy")
        train_image_data = np.load(config['base_dir'] + "/resources/image_predictions/train_image_predictions_v3.npy")

    image_feature_size = 0
    for i in config['image_models']:
        if config["image_feature_type"] == "prediction":
            start_index = config['prediction_image_models'][i][0]
            end_index = config['prediction_image_models'][i][1]
        else:
            start_index = config['feature_image_models'][i][0]
            end_index = config['feature_image_models'][i][1]
        image_feature_size += end_index - start_index


    text_feature_size = len(config["text_models"]) * 1

    for m in config["external_text_models"]:
        feature_labels = config['prediction_text_models'][m]['labels']
        text_feature_size += len(feature_labels)

    for m in config["hatewords_models"]:
        text_feature_size += config["hatewords_models"][m]

    train_ids = train_df["id"].tolist()
    test_ids = test_df["id"].tolist()
    valid_ids = valid_df["id"].tolist()

    train_text = train_df["text"].tolist()
    test_text = test_df["text"].tolist()
    valid_text = valid_df["text"].tolist()

    train_labels = np.array(encode_labels(train_df["label"].tolist()))
    test_labels = np.array(encode_labels(test_df["label"].tolist()))
    valid_labels = np.array(encode_labels(valid_df["label"].tolist()))

    X_train, y_train, X_valid, y_valid, X_test, y_test = {}, {}, {}, {}, {}, {}

    if text_feature_size > 0:
        X_train['text_features'] = np.zeros((len(train_labels), text_feature_size))
        X_valid['text_features'] = np.zeros((len(valid_labels), text_feature_size))
        X_test['text_features'] = np.zeros((len(test_labels), text_feature_size))
    if image_feature_size > 0:
        X_train['image_features'] = np.zeros((len(train_labels), image_feature_size))
        X_valid['image_features'] = np.zeros((len(valid_labels), image_feature_size))
        X_test['image_features'] = np.zeros((len(test_labels), image_feature_size))

    y_train['output_label'] = train_labels
    y_valid['output_label'] = valid_labels
    y_test['output_label'] = test_labels

    all_feature_labels = {}
    last_feature_size = 0
    added_text_features = 1
    for m in config["text_models"]:
        print(config["base_dir"] + '/' + config["text_models"][m] + "/test_predictions.jsonl")
        test_probs_df = pd.read_json(config["base_dir"] + '/' + config["text_models"][m] + "/test_predictions.jsonl",
                                     lines=True, orient='string')
        valid_probs_df = pd.read_json(config["base_dir"] + '/' + config["text_models"][m] + "/valid_predictions.jsonl",
                                      lines=True, orient='string')
        train_probs_df = pd.read_json(config["base_dir"] + '/' + config["text_models"][m] + "/train_predictions.jsonl",
                                      lines=True, orient='string')
        print('Merging features from text model: ', m)

        test_probs = test_probs_df["probs"].tolist()
        valid_probs = valid_probs_df["probs"].tolist()
        train_probs = train_probs_df["probs"].tolist()

        test_probs = [n[1] for n in test_probs]
        valid_probs = [n[1] for n in valid_probs]
        train_probs = [n[1] for n in train_probs]

        # test_probs = [1 if n[1] >= 0.5 else 0 for n in test_probs]
        # valid_probs = [1 if n[1] >= 0.5 else 0 for n in valid_probs]
        # train_probs = [1 if n[1] >= 0.5 else 0 for n in train_probs]

        t1 = np.array(test_probs).reshape((len(test_labels), 1))
        t2 = np.array(train_probs).reshape((len(train_labels), 1))
        v1 = np.array(valid_probs).reshape((len(valid_labels), 1))

        X_test['text_features'][:, last_feature_size:last_feature_size + 1] = t1
        X_valid['text_features'][:, last_feature_size:last_feature_size + 1] = v1
        X_train['text_features'][:, last_feature_size:last_feature_size + 1] = t2

        last_feature_size += 1

    for m in config["external_text_models"]:
        print('Merging features from text model: ', m)

        test_probs_df = pd.read_json(config["base_dir"] + '/' + config["external_text_models"][m] + "/test_predictions.jsonl",
                                     lines=True, orient='string')
        valid_probs_df = pd.read_json(config["base_dir"] + '/' + config["external_text_models"][m] + "/valid_predictions.jsonl",
                                      lines=True, orient='string')
        train_probs_df = pd.read_json(config["base_dir"] + '/' + config["external_text_models"][m] + "/train_predictions.jsonl",
                                      lines=True, orient='string')

        feature_labels = config['prediction_text_models'][m]['labels']
        num_features = len(feature_labels)

        if 'hatebert' in m:
            test_probs = test_probs_df["prediction"].tolist()
            valid_probs = valid_probs_df["prediction"].tolist()
            train_probs = train_probs_df["prediction"].tolist()

            num_features = 1
        else:
            test_probs = test_probs_df["probs"].tolist()
            valid_probs = valid_probs_df["probs"].tolist()
            train_probs = train_probs_df["probs"].tolist()

            if m == "founta_dataset_model":
                test_probs = [n[1] for n in test_probs]
                valid_probs = [n[1] for n in valid_probs]
                train_probs = [n[1] for n in train_probs]

                num_features = 1

        t1 = np.array(test_probs).reshape((len(test_labels), num_features))
        t2 = np.array(train_probs).reshape((len(train_labels), num_features))
        v1 = np.array(valid_probs).reshape((len(valid_labels), num_features))

        X_test['text_features'][:, last_feature_size:last_feature_size + num_features] = t1
        X_valid['text_features'][:, last_feature_size:last_feature_size + num_features] = v1
        X_train['text_features'][:, last_feature_size:last_feature_size + num_features] = t2

        last_feature_size += num_features

    hate_words_list = file_utils.read_file_to_list(config["base_dir"] + '/resources/hate_words.txt')
    for m in config["hatewords_models"]:
        num_features = config["hatewords_models"][m]

        if m == "binary_hateword":
            train_multiclass_hateword_encoding = embed_text_with_hate_words(train_text, hate_words_list, True)
            test_multiclass_hateword_encoding = embed_text_with_hate_words(test_text, hate_words_list, True)
            valid_multiclass_hateword_encoding = embed_text_with_hate_words(valid_text, hate_words_list, True)
        else:
            train_multiclass_hateword_encoding = embed_text_with_hate_words(train_text, hate_words_list)
            test_multiclass_hateword_encoding = embed_text_with_hate_words(test_text, hate_words_list)
            valid_multiclass_hateword_encoding = embed_text_with_hate_words(valid_text, hate_words_list)

        t1 = np.array(test_multiclass_hateword_encoding).reshape((len(test_multiclass_hateword_encoding), num_features))
        t2 = np.array(train_multiclass_hateword_encoding).reshape((len(train_multiclass_hateword_encoding), num_features))
        v1 = np.array(valid_multiclass_hateword_encoding).reshape((len(valid_multiclass_hateword_encoding), num_features))

        X_test['text_features'][:, last_feature_size:last_feature_size + num_features] = t1
        X_valid['text_features'][:, last_feature_size:last_feature_size + num_features] = v1
        X_train['text_features'][:, last_feature_size:last_feature_size + num_features] = t2

        last_feature_size += num_features


    image_pred_feature_labels = file_utils.read_file_to_list('resources/image_predictions/feature_labels.txt')
    # take features from image predictions
    last_feature_size = 0
    for i in config['image_models']:
        if config["image_feature_type"] == "prediction":
            start_index = config['prediction_image_models'][i][0]
            end_index = config['prediction_image_models'][i][1]
        else:
            start_index = config['feature_image_models'][i][0]
            end_index = config['feature_image_models'][i][1]

        num_features = end_index-start_index
        print('Merging', config["image_feature_type"], 'data from image model: ', i, '#feat.', num_features)

        test_probs = test_image_data[:, start_index:end_index]
        valid_probs = valid_image_data[:, start_index:end_index]
        train_probs = train_image_data[:, start_index:end_index]
        if i == "fine-tuned-inception-v3":
            test_probs = [1 if n[0] >= 0.5 else 0 for n in test_probs]
            valid_probs = [1 if n[0] >= 0.5 else 0 for n in valid_probs]
            train_probs = [1 if n[0] >= 0.5 else 0 for n in train_probs]

            test_probs = np.array(test_probs).reshape((len(test_labels), 1))
            valid_probs = np.array(valid_probs).reshape((len(valid_labels), 1))
            train_probs = np.array(train_probs).reshape((len(train_labels), 1))


        X_test['image_features'][:, last_feature_size:last_feature_size + num_features] = np.array(test_probs)
        X_valid['image_features'][:, last_feature_size:last_feature_size + num_features] = np.array(valid_probs)
        X_train['image_features'][:, last_feature_size:last_feature_size + num_features] = np.array(train_probs)

        last_feature_size += num_features

        f_labels = image_pred_feature_labels[start_index:end_index]

    if config["sample_size"] is not None:
        for k in X_train:
            X_train[k] = X_train[k][0:config["sample_size"], :]
            X_valid[k] = X_valid[k][0:config["sample_size"], :]
            X_test[k] = X_test[k][0:config["sample_size"], :]
        for k in y_train:
            y_train[k] = y_train[k][0:config["sample_size"]]
            y_valid[k] = y_valid[k][0:config["sample_size"]]
            y_test[k] = y_test[k][0:config["sample_size"]]

    print("Training input file shapes")
    for k in X_train:
        print('\t' + k + " shape: " + str(X_train[k].shape))
    print("Training output file shapes")
    for k in y_train:
        print('\t' + k + " shape: " + str(y_train[k].shape))
    print('Training with config: ' + str(config))


    results_dir_path = "mmhs_dnn_results"
    if not file_utils.path_exists(config["base_dir"] + "/" + results_dir_path):
        file_utils.create_folder(config["base_dir"] + "/" + results_dir_path)

    now = datetime.now()
    model_dir_path = now.strftime("%d-%m-%Y %H:%M:%S").replace(" ", "_")

    if not file_utils.path_exists(config["base_dir"] + "/" + results_dir_path + "/" + model_dir_path):
        file_utils.create_folder(config["base_dir"] + "/" + results_dir_path + "/" + model_dir_path)

    model_dir_path = config["base_dir"] + "/" + results_dir_path + "/" + model_dir_path

    output_label = "output_label"
    train_accuracy_labels = ['acc']
    valid_accuracy_labels = ['val_acc']
    train_loss_labels = ['loss']
    valid_loss_labels = ['val_loss']

    model_check_point_callback = tf.keras.callbacks.ModelCheckpoint(
        model_dir_path + '/best_model-epoch-{epoch:03d}-acc-{acc:03f}-val_acc-{val_acc:03f}.h5',
        save_best_only=True,
        monitor=valid_accuracy_labels[-1], mode='max')

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(patience=config["epoch_patience"],
                                                               restore_best_weights=True, monitor='val_acc')

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=model_dir_path + "/logs")

    callbacks = [early_stopping_callback,
                 tensorboard_callback,
                 model_check_point_callback]

    print('Using GPUs: ' + str(tf.test.is_gpu_available()))

    model = get_model(config, text_feature_size, image_feature_size)
    history = model.fit(X_train, y_train,
                        validation_data=(X_valid, y_valid),
                        batch_size=config['batch_size'],
                        shuffle=True,
                        epochs=config['epochs'],
                        callbacks=callbacks)

    # predict on validation data
    predictions = model.predict(X_train, batch_size=config['batch_size'])
    prediction_output, score_output = prepare_predictions(config, train_ids, predictions, y_train[output_label])

    print('Model dir:' + model_dir_path)
    print('Training Accuracy: ', score_output['accuracy'])

    # save prediction score, predictions
    file_utils.save_string_to_file(json.dumps(score_output),
                                   model_dir_path + '/training_prediction_score.json')
    file_utils.save_list_to_file(prediction_output,
                                 model_dir_path + '/training_predictions.jsonl')

    # predict on validation data
    predictions = model.predict(X_valid, batch_size=config['batch_size'])

    prediction_output, score_output = prepare_predictions(config, valid_ids, predictions, y_valid[output_label])

    print('Validation Accuracy: ', score_output['accuracy'])

    # save prediction score, predictions
    file_utils.save_string_to_file(json.dumps(score_output),
                                   model_dir_path + '/validation_prediction_score.json')
    file_utils.save_list_to_file(prediction_output,
                                 model_dir_path + '/validation_predictions.jsonl')

    # predict on test data
    predictions = model.predict(X_test, batch_size=config['batch_size'])

    prediction_output, score_output = prepare_predictions(config, test_ids, predictions, y_test[output_label])

    print('Test accuracy: ', score_output['accuracy'])

    # save the model
    model.save(model_dir_path + "/model.h5")
    # save prediction score, predictions
    file_utils.save_string_to_file(json.dumps(score_output),
                                   model_dir_path + '/test_prediction_score.json')
    file_utils.save_list_to_file(prediction_output, model_dir_path + '/test_predictions.jsonl')

    ## save config
    file_utils.save_string_to_file(json.dumps(config),
                                   model_dir_path + '/training_config.json')

    # save model test accuracy to a file
    model_results = {}
    if file_utils.path_exists(config["base_dir"] + '/' + results_dir_path + "/" + config['result_file_name']):
        content = file_utils.read_file_to_set(config["base_dir"] + '/' + results_dir_path + "/"+ config['result_file_name'])
        for c in content:
            t = c.split("\t")
            model_results[t[0]] = float(t[1])

    # add the current trained model with the accuracy
    modalities = get_modalities(config)
    model_results[model_dir_path + '_' + modalities] = float(score_output['accuracy'])

    # sort and save
    sorted_model_results = sorted(model_results, reverse=True, key=model_results.get)
    model_results_output = ""
    for model_name in sorted_model_results:
        model_results_output += model_name + "\t" + str(model_results[model_name]) + "\n"

    # save the content
    file_utils.save_string_to_file(model_results_output,
                                   config["base_dir"] + "/" + results_dir_path + "/"+ config['result_file_name'])

    N = len(history.epoch)
    plt.style.use("ggplot")
    plt.figure()
    for train_loss_label in train_loss_labels:
        plt.plot(np.arange(1, N + 1), history.history[train_loss_label], label=train_loss_label)
    for val_loss_label in valid_loss_labels:
        plt.plot(np.arange(1, N + 1), history.history[val_loss_label], label=val_loss_label)
    for train_acc_label in train_accuracy_labels:
        plt.plot(np.arange(1, N + 1), history.history[train_acc_label], label=train_acc_label)
    for val_acc_label in valid_accuracy_labels:
        plt.plot(np.arange(1, N + 1), history.history[val_acc_label], label=val_acc_label)
    plt.title("Training Loss and Accuracy on MMHS Dataset, " + config['optimizer'])
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(model_dir_path + "/history.png")
    plt.close()




if __name__ == "__main__":
    argv = (sys.argv[1:])
    config_path = 'mmhs_dnn_config.json'
    try:
        opts, args = getopt.getopt(argv, "hc:o:")
    except getopt.GetoptError:
        print('train_dl_boost.py -c <config_path>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('train_dl_boost.py -c <config_path>')
            sys.exit()
        elif opt  == "-c":
            config_path = arg

    if config_path != '':
        with open(config_path) as json_file:
            config = json.load(json_file)
            train(config)
    else:
        print('train_dl_boost.py -c <config_path>')