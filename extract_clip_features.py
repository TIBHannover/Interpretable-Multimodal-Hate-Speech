import torch
import clip
from PIL import Image
import pandas as pd
import numpy as np
import os

def check_number_of_tokens(text: str):

    start = 0
    end = int(len(text) / 2)
    output_text = text[start:end]

    return output_text.strip()

def extract_features(df, base_dir, model, preprocess):

    ids = df["id"].tolist()
    tweet_docs = df["text"].tolist()
    image_text_docs = df["image_text"].tolist()

    features = []

    for id in ids:
        image_path = base_dir + 'resources/MMHS_dataset/img_resized/' + str(id) + '.jpg'
        raw_tweet_text = tweet_docs[ids.index(id)]
        raw_image_text = image_text_docs[ids.index(id)]

        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

        while True:
            try:
                tweet_text = clip.tokenize(raw_tweet_text).to(device)
                break
            except RuntimeError:
                raw_tweet_text = check_number_of_tokens(raw_tweet_text)

        while True:
            try:
                image_text = clip.tokenize(raw_image_text).to(device)
                break
            except RuntimeError:
                raw_image_text = check_number_of_tokens(raw_image_text)

        with torch.no_grad():
            image_features = model.encode_image(image)
            tweet_text_features = model.encode_text(tweet_text)
            image_text_features = model.encode_text(image_text)

            image_features = image_features.data.cpu().numpy()[0]
            tweet_text_features = tweet_text_features.data.cpu().numpy()[0]
            image_text_features = image_text_features.data.cpu().numpy()[0]

            feature = np.concatenate((image_features, tweet_text_features, image_text_features))
            features.append(feature)

    return features



device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# base_dir = "/nfs/home/hakimovs/hate-speech-detection/"

base_dir = "/home/hakimovs/NFS/hate-speech-detection/"

dataset_path = base_dir+ "resources/MMHS_dataset"

train_df = pd.read_json(dataset_path + '/train_over_sampled.txt', lines=True, orient='string')
test_df = pd.read_json(dataset_path + '/test.txt', lines=True, orient='string')
valid_df = pd.read_json(dataset_path + '/valid.txt', lines=True, orient='string')


# train_df, test_df, valid_df = train_df.head(10), test_df.head(10), valid_df.head(10)

print('Loaded the splits')

features_path = dataset_path + '/clip_features/'
if not os.path.exists(features_path):
    os.mkdir(features_path)

print('Processing training data')
train_features = extract_features(train_df, base_dir, model, preprocess)

print('Saving training data')
np.save(features_path+'clip_train_features.npy', train_features)

print('Processing validation data')
valid_features = extract_features(valid_df, base_dir, model, preprocess)

print('Saving validation data')
np.save(features_path+'clip_valid_features.npy', valid_features)

print('Processing test data')
test_features = extract_features(test_df, base_dir, model, preprocess)

print('Saving test data')
np.save(features_path+'clip_test_features.npy', test_features)