import h5py
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
import file_utils
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing import image
import cv2
import math
from PIL import Image, ImageStat

def load_labels(file_path, prefix = None):
    labels = list()
    with open(file_path, "r") as file:
        for l in file.readlines():
            label = l.replace("\n", "").strip()
            if prefix is not None:
                label = prefix + "_"+label
            labels.append(label)
    return labels


def is_gray_scale(img_path):
    im = Image.open(img_path).convert("RGB")
    stat = ImageStat.Stat(im)

    if sum(stat.sum) / 3 == stat.sum[0]:
        return True
    else:
        return False

def load_image_pixels_cv2(filename, shape):
    input_w, input_h = shape
    image = Image.open(filename)
    image = np.array(image, dtype=np.uint8)

    if len(image.shape) != 3:
        if is_gray_scale(filename):
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    image = cv2.resize(image, (input_w, input_h))
    image = np.expand_dims(image, 0)
    return image

def load_image_pixels(filename, shape):
    # load the image to get its shape
    image = load_img(filename)
    width, height = image.size
    # load the image with the required size
    image = load_img(filename, target_size=shape)
    # convert to numpy array
    image = img_to_array(image)
    # scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0
    # add a dimension so that we have one sample
    image = np.expand_dims(image, 0)
    return image


def distance(a, b):
    x1 = a[0];
    y1 = a[1]
    x2 = b[0];
    y2 = b[1]

    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

def detect_face(img, face_detector, eye_detector, target_size=(224, 224), grayscale=False):
    # -----------------------

    exact_image = False
    if type(img).__module__ == np.__name__:
        exact_image = True

    # -----------------------



    if exact_image != True:  # image path passed as input
        img = cv2.imread(img)

    img_raw = img.copy()

    # --------------------------------

    faces = face_detector.detectMultiScale(img, 1.3, 5)

    # print("found faces in ",image_path," is ",len(faces))

    extracted_faces = []

    if len(faces) > 0:
        for x, y, w, h in faces:

            try:
                detected_face = img[int(y):int(y + h), int(x):int(x + w)]
                detected_face_gray = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)

                # ---------------------------
                # face alignment

                eyes = eye_detector.detectMultiScale(detected_face_gray)

                if len(eyes) >= 2:
                    # find the largest 2 eye
                    base_eyes = eyes[:, 2]

                    items = []
                    for i in range(0, len(base_eyes)):
                        item = (base_eyes[i], i)
                        items.append(item)

                    df = pd.DataFrame(items, columns=["length", "idx"]).sort_values(by=['length'], ascending=False)

                    eyes = eyes[df.idx.values[0:2]]

                    # -----------------------
                    # decide left and right eye

                    eye_1 = eyes[0];
                    eye_2 = eyes[1]

                    if eye_1[0] < eye_2[0]:
                        left_eye = eye_1
                        right_eye = eye_2
                    else:
                        left_eye = eye_2
                        right_eye = eye_1

                    # -----------------------
                    # find center of eyes

                    left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
                    left_eye_x = left_eye_center[0];
                    left_eye_y = left_eye_center[1]

                    right_eye_center = (int(right_eye[0] + (right_eye[2] / 2)), int(right_eye[1] + (right_eye[3] / 2)))
                    right_eye_x = right_eye_center[0];
                    right_eye_y = right_eye_center[1]

                    # -----------------------
                    # find rotation direction

                    if left_eye_y > right_eye_y:
                        point_3rd = (right_eye_x, left_eye_y)
                        direction = -1  # rotate same direction to clock
                    else:
                        point_3rd = (left_eye_x, right_eye_y)
                        direction = 1  # rotate inverse direction of clock

                    # -----------------------
                    # find length of triangle edges

                    a = distance(left_eye_center, point_3rd)
                    b = distance(right_eye_center, point_3rd)
                    c = distance(right_eye_center, left_eye_center)

                    # -----------------------
                    # apply cosine rule

                    cos_a = (b * b + c * c - a * a) / (2 * b * c)
                    angle = np.arccos(cos_a)  # angle in radian
                    angle = (angle * 180) / math.pi  # radian to degree

                    # -----------------------
                    # rotate base image

                    if direction == -1:
                        angle = 90 - angle

                    img = Image.fromarray(img_raw)
                    img = np.array(img.rotate(direction * angle))

                    # you recover the base image and face detection disappeared. apply again.
                    faces = face_detector.detectMultiScale(img, 1.3, 5)
                    if len(faces) > 0:
                        x, y, w, h = faces[0]
                        detected_face = img[int(y):int(y + h), int(x):int(x + w)]

                # -----------------------

                # face alignment block end
                # ---------------------------

                # face alignment block needs colorful images. that's why, converting to gray scale logic moved to here.
                if grayscale == True:
                    detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)

                detected_face = cv2.resize(detected_face, target_size)

                img_pixels = image.img_to_array(detected_face)
                img_pixels = np.expand_dims(img_pixels, axis=0)

                # normalize input in [0, 1]
                img_pixels /= 255
                extracted_faces.append(img_pixels)
            except:
                pass
    else:

        if exact_image == True:

            if grayscale == True:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            img = cv2.resize(img, target_size)
            img_pixels = image.img_to_array(img)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255
            extracted_faces.append(img_pixels)
    return extracted_faces

def extract_predictions(image_path, shape, model, labels: list, second_image_loader = False, model_name=''):
    probabilities = None

    input_w, input_h = shape
    # load and prepare image

    image = None
    if second_image_loader == True:
        try:
            image = load_image_pixels_cv2(image_path, (input_w, input_h))
        except:
            pass

    if image is None or second_image_loader == False:
        try:
            image = load_image_pixels(image_path, (input_w, input_h))
        except:
            pass

    try:
        # do prediction
        prediction = model.predict(image)

        max_predicted_index = np.argmax(prediction[0])
        probability = prediction[0][max_predicted_index]
        max_label = labels[max_predicted_index]

        probabilities = np.zeros(len(labels))

        for l in labels:
            index = labels.index(l)
            prob = prediction[0][index]

            probabilities[labels.index(l)] = prob
    except Exception as e:
        pass

    return probabilities

def extract_face_emotion_predictions(image_path, face_detector, eye_detector, face_emotion_model, labels):


    emotion_labels = ['face_emotion_angry', 'face_emotion_afraid',  'face_emotion_happy', 'face_emotion_sad', 'face_emotion_surprised', 'face_emotion_neutral']

    probabilities = np.zeros(len(labels), dtype=float)

    try:
        faces = detect_face(image_path, face_detector, eye_detector, (48, 48), True)
    except:
        faces = list()

    if len(faces) == 0:
        return probabilities

    max_prob = 0
    max_predictions = None
    max_label = ''
    for face in faces:

        # run prediction on image
        predictions = face_emotion_model.predict(face)[0]
        max_predicted_index = np.argmax(predictions)
        probability = predictions[max_predicted_index]

        if max_prob < probability:
            max_prob = probability
            max_predictions = predictions
            max_label = emotion_labels[max_predicted_index]



    if max_prob >=0.7:
        probabilities = max_predictions
    return probabilities

def extract_all_predictions(ids, base_dir):
    ## Face emotion detector
    all_labels = load_labels(base_dir + 'resources/image_predictions/feature_labels.txt')
    face_emotion_labels = all_labels[4:10]
    imagenet_labels = all_labels[375:1375]
    nudity_labels = all_labels[0:2]
    places365_labels = all_labels[10:375]
    binary_hateword_labels = all_labels[1377:1379]
    binary_finetuned_inception_labels = all_labels[1375:1377]
    multiclass_hateword_labels = all_labels[1379:1805]

    feature_size = 1805

    all_features = np.zeros((len(ids), feature_size))

    face_detector = cv2.CascadeClassifier(base_dir + 'resources/model_weights/haarcascade_frontalface_default.xml')
    eye_detector = cv2.CascadeClassifier(base_dir + 'resources/model_weights/haarcascade_eye.xml')
    face_emotion_model = tf.keras.models.load_model(base_dir + 'resources/model_weights/face_emotions.h5')


    ## ImageNet object detector
    imagenet_model = InceptionV3(include_top=True, weights='imagenet', pooling='avg',
                                 input_tensor=tf.keras.layers.Input(shape=(299, 299, 3)))
    imagenet_shape = 299, 299

    ## Nudity Detector
    nudity_full_model = tf.keras.models.load_model(base_dir + 'resources/model_weights/nudenet.h5')
    nudity_shape = 256, 256

    ## Places365 detector
    places365_full_model = tf.keras.models.load_model(base_dir + 'resources/model_weights/places365.h5')
    places365_shape = 224, 224

    finetuned_inception_model = tf.keras.models.load_model(base_dir + 'finetune_results/11-05-2020_11:44:07/model.h5')
    binary_hateword_model = tf.keras.models.load_model(
        base_dir + 'finetune_hate_image_results/11-05-2020_22:04:21/model.h5')
    multiclass_hateword_model = tf.keras.models.load_model(
        base_dir + 'finetune_hate_image_results/11-05-2020_22:06:21/model.h5')




    c = 0

    for id in ids:
        image_path = base_dir+'resources/Memotion7k/images/' + str(id)

        features = np.zeros(feature_size, dtype=float)

        nudity_predictions = extract_predictions(image_path, nudity_shape, nudity_full_model, nudity_labels, 'nudity')
        face_emotion_predictions = extract_face_emotion_predictions(image_path, face_detector, eye_detector, face_emotion_model, face_emotion_labels)
        places365_predictions = extract_predictions(image_path, places365_shape, places365_full_model, places365_labels, second_image_loader=True, model_name='places')
        imagenet_predictions = extract_predictions(image_path, imagenet_shape, imagenet_model, imagenet_labels, 'imagenet')
        hateword_predictions = extract_predictions(image_path, (299, 299), binary_hateword_model,
                                                   binary_hateword_labels, 'binary hate')
        multiclass_hateword_predictions = extract_predictions(image_path, (299, 299), multiclass_hateword_model,
                                                   multiclass_hateword_labels, 'multi-hate')
        finetuned_inception_predictions = extract_predictions(image_path, (299, 299), finetuned_inception_model,
                                                              binary_finetuned_inception_labels, 'finetuned-inception')

        start_index = 0
        end_index = len(nudity_labels)
        features[start_index:end_index] = nudity_predictions

        start_index = end_index
        end_index = end_index + len(binary_hateword_labels)
        features[start_index:end_index] = hateword_predictions

        start_index = end_index
        end_index = end_index + len(face_emotion_labels)
        features[start_index:end_index] = face_emotion_predictions

        start_index = end_index
        end_index = end_index + len(places365_labels)
        features[start_index:end_index] = places365_predictions

        start_index = end_index
        end_index = end_index + len(imagenet_labels)
        features[start_index:end_index] = imagenet_predictions

        start_index = end_index
        end_index = end_index + len(binary_finetuned_inception_labels)
        features[start_index:end_index] = finetuned_inception_predictions

        start_index = end_index
        end_index = end_index + len(binary_hateword_labels)
        features[start_index:end_index] = hateword_predictions

        start_index = end_index
        end_index = end_index + len(multiclass_hateword_labels)
        features[start_index:end_index] = multiclass_hateword_predictions

        all_features[c, :] = features
        c += 1
        # if c % 1 == 0:
        print(str(c) + '/'+str(len(ids)))
    return all_features

print('Loading pre-trained model weights')

base_dir = "/home/hakimovs/PycharmProjects/hate-speech-detection/"

print("Loading Memotion7k dataset")

splits = ['train']

for split in splits:
    df = pd.read_json(base_dir + 'resources/Memotion7k/'+split+'.txt', lines=True, orient='string')

    ids = df["image_name"].tolist()
    print('Extracting '+split+' predictions')
    valid_features = extract_all_predictions(ids, base_dir)
    np.save(base_dir + 'resources/Memotion7k/image_predictions/'+split+'_image_predictions.npy', valid_features)

print("finished!")