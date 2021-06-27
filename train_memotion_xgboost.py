import getopt
import numpy as np
from datetime import datetime
import pandas as pd
import json
import sys
import os
import tensorflow as tf
import xgboost as xgb
import file_utils
import operator
import matplotlib.pyplot as plt
import random
import string
from sklearn.metrics import auc, roc_auc_score, roc_curve, recall_score, precision_score, f1_score, average_precision_score
from sklearn.metrics import precision_recall_curve

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
        predicted_class = 1 if predicted_probs > 0.5 else 0

        binary_predictions.append(predicted_class)

        if expected == predicted_class:
            true_positives[expected] +=1
        total_expected[expected] +=1
        total_predicted[predicted_class] += 1

        l = {"id": str(ids[i]), "prediction": str(predicted_class), "label": str(labels[i]), "probs": str(predicted_probs)}
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

    fpr, tpr, thresholds = roc_curve(labels, binary_predictions, pos_label=1)
    macro_roc_auc_score= roc_auc_score(labels, binary_predictions)
    auc_score = auc(fpr, tpr)

    micro_precision, micro_recall, _ = precision_recall_curve(labels,binary_predictions)

    score_output = {"accuracy": accuracy, "average_precision":average_precision, "f1":f1,  "macro_f1":macro_f1,  "recall":recall, "precision":precision, "Offensive": {"precision": precision_hate, "recall": recall_hate, "f1": f1_hate, "support": total_expected[1]},
                    "NotOffensive": {"precision": precision_not_hate, "recall": recall_not_hate, "f1": f1_not_hate, "support": total_expected[0]}, "macro_roc_auc":macro_roc_auc_score, "auc":auc_score
                    }

    return prediction_output, score_output, micro_precision, micro_recall
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
    label_to_index = {"not_offensive": 0, "slight": 1, "very_offensive": 1, "hateful_offensive": 1}

    for label in data:
        y.append(label_to_index[label])
    return np.array(y)

def train(config: dict):
    # merge all features into a single train/test data

    train_df = pd.read_json(config["base_dir"] + '/' + config['train_split_path'], lines=True, orient='string')
    valid_df = pd.read_json(config["base_dir"] + '/' + config['valid_split_path'], lines=True, orient='string')
    test_df = pd.read_json(config["base_dir"] + '/' + config['test_split_path'], lines=True, orient='string')

    if len(config["image_models"]) > 0:
        print('Loading image data')
        if config["image_feature_type"] == "prediction":
            valid_data = np.load(config['base_dir'] + "/resources/Memotion7k/image_predictions/valid_image_predictions.npy")
            test_data = np.load(config['base_dir'] + "/resources/Memotion7k/image_predictions/test_image_predictions.npy")
            train_data = np.load(config['base_dir'] + "/resources/Memotion7k/image_predictions/train_over_sampled_image_predictions.npy")
        else:
            valid_data = np.load(config['base_dir'] + "/resources/image_features/valid_image_features.npy")
            test_data = np.load(config['base_dir'] + "/resources/image_features/test_image_features.npy")
            train_data = np.load(config['base_dir'] + "/resources/image_features/train_image_features.npy")


    image_model_feature_size = 0
    for i in config['image_models']:
        if config["image_feature_type"] == "prediction":
            start_index = config['prediction_image_models'][i][0]
            end_index = config['prediction_image_models'][i][1]
        else:
            start_index = config['feature_image_models'][i][0]
            end_index = config['feature_image_models'][i][1]
        image_model_feature_size += end_index - start_index

    num_features = 1
    feature_size = len(config["text_models"]) * num_features + image_model_feature_size

    for m in config["external_text_models"]:
        feature_labels = config['prediction_text_models'][m]['labels']
        feature_size += len(feature_labels)

    for m in config["hatewords_models"]:
        feature_size += config["hatewords_models"][m]

    for metric in config["image_text_relations"]:
        feature_size += config["image_text_relations"][metric]

    for sentiment_type in config['visual_sentiment']:
        feature_size +=1

    train_ids = train_df["image_name"].tolist()
    test_ids = test_df["image_name"].tolist()
    valid_ids = valid_df["image_name"].tolist()

    train_text = train_df["text_corrected"].tolist()
    test_text = test_df["text_corrected"].tolist()
    valid_text = valid_df["text_corrected"].tolist()

    train_labels = np.array(encode_labels(train_df["offensive"].tolist()))
    test_labels = np.array(encode_labels(test_df["offensive"].tolist()))
    valid_labels = np.array(encode_labels(valid_df["offensive"].tolist()))



    test_features = np.zeros((len(test_labels), feature_size))
    valid_features = np.zeros((len(valid_labels), feature_size))
    train_features = np.zeros((len(train_labels), feature_size))

    all_feature_labels = {}
    last_feature_size = 0
    for m in config["text_models"]:
        print(config["base_dir"] + '/' + config["text_models"][m] + "/test_predictions.jsonl")
        test_probs_df = pd.read_json(config["base_dir"] + '/' + config["text_models"][m] + "/test_predictions.jsonl",
                                     lines=True, orient='string')
        valid_probs_df = pd.read_json(config["base_dir"] + '/' + config["text_models"][m] + "/validation_predictions.jsonl",
                                      lines=True, orient='string')
        train_probs_df = pd.read_json(config["base_dir"] + '/' + config["text_models"][m] + "/training_predictions.jsonl",
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

        t1 = np.array(test_probs).reshape((len(test_labels), num_features))
        t2 = np.array(train_probs).reshape((len(train_labels), num_features))
        v1 = np.array(valid_probs).reshape((len(valid_labels), num_features))


        test_features[:, last_feature_size:last_feature_size + num_features] = t1
        valid_features[:, last_feature_size:last_feature_size + num_features] = v1
        train_features[:, last_feature_size:last_feature_size + num_features] = t2

        last_feature_size += num_features

        for f_index in range(0, num_features):
            all_feature_labels['f'+str(len(all_feature_labels))] = (m+"_p"+str(f_index))

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

        test_features[:, last_feature_size:last_feature_size + num_features] = t1
        valid_features[:, last_feature_size:last_feature_size + num_features] = v1
        train_features[:, last_feature_size:last_feature_size + num_features] = t2

        last_feature_size += num_features

        for f_index in range(0, num_features):
            all_feature_labels['f'+str(len(all_feature_labels))] = (m+"_"+feature_labels[f_index])

    hate_words_list = file_utils.read_file_to_list(config["base_dir"] + '/resources/hate_words.txt')

    for m in config["hatewords_models"]:
        num_features = config["hatewords_models"][m]

        feature_size += num_features

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

        test_features[:, last_feature_size:last_feature_size + num_features] = t1
        valid_features[:, last_feature_size:last_feature_size + num_features] = v1
        train_features[:, last_feature_size:last_feature_size + num_features] = t2

        last_feature_size += num_features

        for f_index in range(0, num_features):
            if m == "binary_hateword":
                all_feature_labels['f' + str(len(all_feature_labels))] = ("Binary_hate_" + str(f_index))
            else:
                all_feature_labels['f' + str(len(all_feature_labels))] = ("HateWord_" + hate_words_list[f_index])


    image_pred_feature_labels = file_utils.read_file_to_list('resources/image_predictions/feature_labels.txt')
    # take features from image predictions
    for i in config['image_models']:
        if config["image_feature_type"] == "prediction":
            start_index = config['prediction_image_models'][i][0]
            end_index = config['prediction_image_models'][i][1]
        else:
            start_index = config['feature_image_models'][i][0]
            end_index = config['feature_image_models'][i][1]

        num_features = end_index-start_index
        print('Merging', config["image_feature_type"], 'data from image model: ', i, '#feat.', num_features)

        test_probs = test_data[:, start_index:end_index]
        valid_probs = valid_data[:, start_index:end_index]
        train_probs = train_data[:, start_index:end_index]
        if i == "fine-tuned-inception-v3":
            test_probs = [1 if n[0] >= 0.5 else 0 for n in test_probs]
            valid_probs = [1 if n[0] >= 0.5 else 0 for n in valid_probs]
            train_probs = [1 if n[0] >= 0.5 else 0 for n in train_probs]

            test_probs = np.array(test_probs).reshape((len(test_labels), 1))
            valid_probs = np.array(valid_probs).reshape((len(valid_labels), 1))
            train_probs = np.array(train_probs).reshape((len(train_labels), 1))

        test_features[:, last_feature_size:last_feature_size + num_features] = np.array(test_probs)
        valid_features[:, last_feature_size:last_feature_size + num_features] = np.array(valid_probs)
        train_features[:, last_feature_size:last_feature_size + num_features] = np.array(train_probs)

        last_feature_size += num_features

        f_labels = image_pred_feature_labels[start_index:end_index]

        if config["image_feature_type"] == "feature":
            for f_index in range(0, num_features):
                all_feature_labels['f' + str(len(all_feature_labels))] = i+"_"+str(f_index)
        else:
            for f_label in f_labels:
                all_feature_labels['f' + str(len(all_feature_labels))] = f_label

    if config["sample_size"] is not None:
        train_features = train_features[0:config["sample_size"], :]
        train_labels = train_labels[0:config["sample_size"]]

    print('Training data:', train_features.shape)
    print('Test data:', test_features.shape)
    print('Validation data:', valid_features.shape)
    print('Training labels:', train_labels.shape)

    if not path_exists(config["base_dir"] + '/' +"memotion_xgboost_results"):
        create_folder(config["base_dir"] + '/' +"memotion_xgboost_results")

    now = datetime.now()
    model_dir_path = now.strftime("%d-%m-%Y %H:%M:%S").replace(" ", "_") + "_"+random_string_generator()

    print('Model path:', model_dir_path)

    if not path_exists(config["base_dir"] + '/' +"memotion_xgboost_results/"+model_dir_path):
        create_folder(config["base_dir"] + '/' +"memotion_xgboost_results/"+model_dir_path)

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    print('Config:\n',config)

    d_train = xgb.DMatrix(train_features, label=train_labels)
    d_val = xgb.DMatrix(valid_features, label=valid_labels)
    d_test = xgb.DMatrix(test_features, label=test_labels)

    param = {'max_depth': config['max_depth'], 'learning_rate': config['learning_rate'], 'objective': 'binary:logistic'}
    # param['nthread'] = 4
    param['eval_metric'] = 'error'
    if config['use_gpu']:
        param['gpu_id'] = 0
        param['tree_method'] = 'gpu_hist'

    param['booster'] = config['booster_algorithm']
    param['num_parallel_tree'] = config['num_parallel_tree']
    if config['booster_algorithm'] == 'dart':
        param['rate_drop']= config['dart_params']['rate_drop']
        param['skip_drop'] = config['dart_params']['skip_drop']
        param['sample_type'] = config['dart_params']['sample_type']
        param['normalize_type'] = config['dart_params']['normalize_type']

    elif config['booster_algorithm'] == 'randomforest':
        param['booster'] = config['randomforest_params']['booster_algorithm']
        param['colsample_bynode'] = config['randomforest_params']['colsample_bynode']
        param['learning_rate'] = config['randomforest_params']['learning_rate']
        param['num_parallel_tree'] = config['randomforest_params']['num_parallel_tree']
        param['subsample'] = config['randomforest_params']['subsample']
        config['epochs'] = config['randomforest_params']['epochs']

    eval_list = [(d_train, 'train'), (d_val, 'val')]

    evals_result = {}

    # xb = xgb.XGBClassifier()
    # xb.fit(feat_train, np.argmax(y_train, axis=1))

    bst = xgb.train(param, d_train, config['epochs'], eval_list, early_stopping_rounds=config['epoch_patience'], evals_result = evals_result)

    #* 'weight': the number of times a feature is used to split the data across all trees.
    #* 'gain': the average gain across all splits the feature is used in.
    #* 'cover': the average coverage across all splits the feature is used in.
    #* 'total_gain': the total gain across all splits the feature is used in.
    # * 'total_cover': the total coverage across all splits the feature is used in.

    importance_list = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']

    for imp_label in importance_list:

        feature_importance_weights = bst.get_score(importance_type=imp_label)

        output_feature_weights = prepare_feature_importance_scores(all_feature_labels, feature_importance_weights)

        file_utils.save_string_to_file(output_feature_weights, config["base_dir"] + '/' + 'memotion_xgboost_results/' + model_dir_path + '/features_'+imp_label+'.txt')

    ## save the save the test features
    np.save(config["base_dir"] + '/' + 'memotion_xgboost_results/' + model_dir_path + '/test_features.npy', test_features)

    ## validation
    y_pred = bst.predict(d_val, ntree_limit=bst.best_ntree_limit, training=False)

    prediction_output, score_output,_, _ = prepare_predictions(config, valid_ids, y_pred, valid_labels)

    # save prediction score, predictions
    file_utils.save_string_to_file(json.dumps(score_output), config[
        "base_dir"] + '/' + 'memotion_xgboost_results/' + model_dir_path + '/valid_prediction_score.json')
    file_utils.save_list_to_file(prediction_output, config[
        "base_dir"] + '/' + 'memotion_xgboost_results/' + model_dir_path + '/valid_predictions.jsonl')
    file_utils.save_string_to_file(json.dumps(all_feature_labels), config[
        "base_dir"] + '/' + 'memotion_xgboost_results/' + model_dir_path + '/all_feature_labels.txt')

    print('Valid accuracy:', score_output['accuracy'])

    ## test evaluation
    y_pred = bst.predict(d_test, ntree_limit=bst.best_ntree_limit, training=False)
    prediction_output, score_output, test_micro_precision, test_micro_recall = prepare_predictions(config, test_ids, y_pred, test_labels)
    file_utils.save_string_to_file(json.dumps(score_output), config["base_dir"] + '/' + 'memotion_xgboost_results/' + model_dir_path + '/test_prediction_score.json')
    file_utils.save_list_to_file(prediction_output, config["base_dir"] + '/' + 'memotion_xgboost_results/' + model_dir_path + '/test_predictions.jsonl')

    print('Test accuracy:', score_output['accuracy'])
    print('Test macro-f1:', score_output['macro_f1'])
    # save prediction score, predictions

    bst.save_model(config["base_dir"] + "/" + "memotion_xgboost_results/" + model_dir_path + "/model.bin")

    # save training config
    save_string_to_file(json.dumps(config), config["base_dir"] + '/' + "memotion_xgboost_results/" + model_dir_path + "/training_config.json")

    if "result_file_name" in config:
        result_file_name = config["result_file_name"]
    else:
        result_file_name = "model_results.txt"


    # save model test accuracy to a file
    model_results = {}
    if path_exists(config["base_dir"] + '/' +"memotion_xgboost_results/"+result_file_name):
        content = read_file_to_set(config["base_dir"] + '/' +"memotion_xgboost_results/"+result_file_name)
        for c in content:
            t = c.split("\t")
            model_results[t[0]] = float(t[1])

    # add the current trained model with the accuracy
    model_results[model_dir_path + '_XB_' + get_modalities(config)+"image_"+config["image_feature_type"]] = float(score_output['accuracy'])

    # sort and save
    sorted_model_results = sorted(model_results, reverse=True, key=model_results.get)
    model_results_output = ""
    for model_name in sorted_model_results:
        model_results_output += model_name + "\t" + str(model_results[model_name]) + "\n"

    # save the content
    save_string_to_file(model_results_output, config["base_dir"] + '/' +"memotion_xgboost_results/"+result_file_name)

    epochs = len(evals_result['val']['error'])
    # plot classification error
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(1, epochs + 1), evals_result['train']['error'], label="train_error")
    plt.plot(np.arange(1, epochs + 1), evals_result['val']['error'], label="val_error")
    plt.xlabel("Epoch #")
    plt.ylabel("Error rate")
    plt.legend(loc="lower left")
    plt.savefig(config["base_dir"] + "/" + "memotion_xgboost_results/" + model_dir_path + "/history.png")
    plt.close()

    plt.style.use("ggplot")
    plt.figure()
    plt.step(test_micro_recall, test_micro_precision, where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.savefig(config["base_dir"] + "/" + "memotion_xgboost_results/" + model_dir_path + "/precision_recall_curve.png")
    plt.close()


if __name__ == "__main__":
    argv = (sys.argv[1:])
    config_path = 'memotion_xgboost_config.json'
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