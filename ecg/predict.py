from __future__ import print_function
from __future__ import division

import argparse
import numpy as np
import keras
import os
import heapq

import load
import util


def evaluate(probs, length_predicts, length_frames):
    frame_class = np.argmax(probs, axis=2)
    length_classes = probs.shape[2]
    classes_count = [([0] * length_classes) for i in range(length_predicts)]

    for num_predict in range(length_predicts):

        for num_class in range(length_classes):
            classes_count[num_predict][num_class] = \
                np.sum(frame_class[num_predict][0:length_frames[num_predict]] == num_class)

        predict_class = classes_count[num_predict].index(max(classes_count[num_predict]))
        if predict_class == 1 & (classes_count[num_predict][1] == length_frames[num_predict]):
            pass
        elif predict_class == 1:
            count_large = heapq.nlargest(2, classes_count[num_predict])
            predict_class = classes_count[num_predict].index(count_large[1])

        if predict_class == 0:
            print("Patient {} : Atrial fibrillation - {:.2%}"
                  .format((num_predict + 1), (classes_count[num_predict][0] / length_frames[num_predict])))
        elif predict_class == 1:
            print("Patient {} : Normal rhythm - {:.2%}"
                  .format((num_predict + 1), (classes_count[num_predict][1] / length_frames[num_predict])))
        elif predict_class == 2:
            print("Patient {} : Other rhythm - {:.2%}"
                  .format((num_predict + 1), (classes_count[num_predict][2] / length_frames[num_predict])))
        else:
            print("Patient {} : Noisy recording (poor signal quality) - {:.2%}"
                  .format((num_predict + 1), (classes_count[num_predict][3] / length_frames[num_predict])))

    return predict_class


def predict(data_json, model_path):
    preproc = util.load(os.path.dirname(model_path))
    dataset = load.load_dataset(data_json)
    x, y = preproc.process(*dataset)

    model = keras.models.load_model(model_path)
    probs = model.predict(x, verbose=1)

    # evaluate the model
    score = model.evaluate(x, y)

    length_frames = []
    length_predicts = len(dataset[1])

    for length_predict in range(length_predicts):
        length_frames.append(len(dataset[1][length_predict]))

    predict_class = evaluate(probs, length_predicts, length_frames)

    print("The model {} is : {:.2%}".format(model.metrics_names[1], score[1]))

    return probs, predict_class


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_json", help="path to data json")
    parser.add_argument("model_path", help="path to model")
    args = parser.parse_args()
    probs, score = predict(args.data_json, args.model_path)
