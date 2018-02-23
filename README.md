# keras-audio

keras project for audio deep learning

# Features

### Audio Classification

* The classifier [ResNet50AudioClassifier](keras_audio/library/resnet.py) converts audio into mel-spectrogram and uses the resnet-50
DCnn architecture to classifier audios based on its associated labels. 
* The classifier [Cifar10AudioClassifier](keras_audio/library/cifar10.py) converts audio into mel-spectrogram and uses the cifar-10
DCnn architecture to classifier audios based on its associated labels. 

# Usage

### Audio Classification

The audio classification uses [Gtzan](http://opihi.cs.uvic.ca/sound/genres.tar.gz) data set to train the
music classifier to recognize the genre of songs. 

The classification works by converting audio or song file into a mel-spectrogram which can be thought of
a 3-dimension matrix in a similar manner to an image 

To train on the Gtzan data set, run the following command:

```bash
cd demo
python resnet_train.py
```

The [sample codes](demo/resnet_train.py) below show how to train ResNet50Classifier to classify songs
based on its genre labels:

```python
from keras_audio.library.resnet import ResNet50AudioClassifier
from keras_audio.library.utility.gtzan_loader import download_gtzan_genres_if_not_found


def load_audio_path_label_pairs(max_allowed_pairs=None):
    download_gtzan_genres_if_not_found('./very_large_data/gtzan')
    audio_paths = []
    with open('./data/lists/test_songs_gtzan_list.txt', 'rt') as file:
        for line in file:
            audio_path = './very_large_data' + line.strip()
            audio_paths.append(audio_path)
    pairs = []
    with open('./data/lists/test_gt_gtzan_list.txt', 'rt') as file:
        for line in file:
            label = int(line)
            if max_allowed_pairs is None or len(pairs) < max_allowed_pairs:
                pairs.append((audio_paths[len(pairs)], label))
            else:
                break
    return pairs


def main():
    audio_path_label_pairs = load_audio_path_label_pairs()
    print('loaded: ', len(audio_path_label_pairs))

    classifier = ResNet50AudioClassifier()
    batch_size = 64
    epochs = 10
    history = classifier.fit(audio_path_label_pairs, model_dir_path='./models', batch_size=batch_size, epochs=epochs)


if __name__ == '__main__':
    main()

```

After training, the trained models are saved to [demo/models](demo/models). To test the trained model, run
the following command:

```bash
cd demo
python resnet_predict.py
```

The [sample codes](demo/resnet_predict.py) shows how to test the trained ResNet50AudioClassifier model:

```python
from random import shuffle

from keras_audio.library.resnet import ResNet50AudioClassifier
from keras_audio.library.utility.gtzan_loader import download_gtzan_genres_if_not_found


def load_audio_path_label_pairs(max_allowed_pairs=None):
    download_gtzan_genres_if_not_found('./very_large_data/gtzan')
    audio_paths = []
    with open('./data/lists/test_songs_gtzan_list.txt', 'rt') as file:
        for line in file:
            audio_path = './very_large_data/' + line.strip()
            audio_paths.append(audio_path)
    pairs = []
    with open('./data/lists/test_gt_gtzan_list.txt', 'rt') as file:
        for line in file:
            label = int(line)
            if max_allowed_pairs is None or len(pairs) < max_allowed_pairs:
                pairs.append((audio_paths[len(pairs)], label))
            else:
                break
    return pairs


def main():
    audio_path_label_pairs = load_audio_path_label_pairs()
    shuffle(audio_path_label_pairs)
    print('loaded: ', len(audio_path_label_pairs))

    classifier = ResNet50AudioClassifier()
    classifier.load_model(model_dir_path='./models')

    for i in range(0, 20):
        audio_path, actual_label = audio_path_label_pairs[i]
        predicted_label = classifier.predict_class(audio_path)
        print(audio_path)
        print('predicted: ', predicted_label, 'actual: ', actual_label)


if __name__ == '__main__':
    main()

```
