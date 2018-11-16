from random import shuffle
import sys

from keras_audio.library.cifar10 import Cifar10AudioClassifier

def predict(audio_path, model_dir_path):
    classifier = Cifar10AudioClassifier()
    classifier.load_model(model_dir_path)

    predicted_label_id = classifier.predict_class(audio_path)

    print(predicted_label_id)

if __name__ == "__main__":
    predict(sys.argv[1], sys.argv[2])