# gan-image-captioning

## Data set preparation
_Remark : this procedure is the one we used to create our dataset in order to use the default configuration file (`config.json`). You can adapt it for your own one._
1. Create a folder `data`.
2. Create four folders inside `data`: `captions`, `features`, `images` and `links`.
3. Inside `data/captions`, download the files `train.lc.norm.tok.en` and `val.lc.norm.tok.en` from the [multi30k dataset](https://github.com/multi30k/dataset/tree/master/data/task1/tok).
4. Download the visual features from the multi30k dataset Google Drive ([link](https://github.com/multi30k/dataset#visual-features)) and download inside `data/features` folder `train-resnet50-avgpool.npy` and `val-resnet50-avgpool.npy`.
5. The `data/images` folder contains the raw images, which can be requested [here](https://github.com/multi30k/dataset#visual-features). This step is not required if you don't want to visualise caption-image pairs.
6. Download inside `data/links` the files `train.txt` and `val.txt` from the [multi30k dataset](https://github.com/multi30k/dataset/tree/master/data/task1/image_splits). This step is not required if you don't want to visualise caption-image pairs.
7. Execute the following commands to normalize the captions :
```sh
$ python normalize_captions.py data/captions/train.lc.norm.tok.en
$ python normalize_captions.py data/captions/val.lc.norm.tok.en
```
`norm_train.lc.norm.tok.en` and `norm_val.lc.norm.tok.en` were created. Move them into `data/captions`.
8. Execute the following command to build the vocabulary corpus :
```sh
$ python vocab.py data/captions/norm_train.lc.norm.tok.en flickr_vocab.en
```
Move the `flickr_vocab.en` into `data` folder.

## Model training
To train the model, execute the following command:
```sh
$ python main.py config.json
```