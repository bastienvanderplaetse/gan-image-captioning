
# gan-image-captioning
Project realized during my master thesis in Computer Science at [UMONS (University of Mons, Belgium)](https://web.umons.ac.be/en/).
Director : Jean-Benoit DELBROUCK
Co-director : Dr Stéphane DUPONT
Reporters : Dr Hadrien MELOT & Adrien COPPENS

Our model is available [here](http://www.mediafire.com/file/jlnktm8yr5y15qm/output_epoch29_bleu0.07394183608962669/file).

## Environment preparation
Our environment uses Python 3.6.8 with Anaconda. All the libraries and requirements can be installed by running :
```sh
$ pip install -r reqs
```

## Data set preparation
We use [MS COCO](http://cocodataset.org) as data set. We propose two methods to format it for training our models :
1. Via script : prepare the data set by running our formatting script.
2. Via downloading : download the ready-to-use version of the data set.
These procedures allow you to obtain the same data set that the one we used. You can adapt it for your own needs.

_Method 1 - Via script_
1. Create a folder `cocodataset/images`.
2. Download images from [2014 release](http://cocodataset.org/#download) into `cocodataset/images` (download both [2014 Train images](http://images.cocodataset.org/zips/train2014.zip) and [2014 Val images](http://images.cocodataset.org/zips/val2014.zip)).
3. (For Windows users) Download the [MS COCO annotations zip file](msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip) into the root of the projet.
4. Run the following commands :
```sh
python prepare_coco.py
python vocab.py cocodataset/captions/train.en coco_vocab.en 4000
```
5. Move the generated `coco_vocab.en` file into the `cocodataset` folder.
6. Create a folder `cocodataset/embeddings`.
7. Download the [GloVe embeddings 6B.300d](http://nlp.stanford.edu/data/glove.6B.zip) into the `cocodataset/embeddings`.

_Method 2 - Via downloading_
1. Download the data set [here](http://www.mediafire.com/file/33o8ylurdpgiwqb/ganimagecapcocodataset.rar/file).
2. Extract the content of the zip file into the root of the project.

## Model configuration
All the configuration parameters of this project are in the `config.json` file. You can create your own copy of the configuration or modify this one.
_Note : the default parameters are the ones that gave us the best results._
The `load_dict` parameter is used to test a specific model. The default one is the one correspind to our best results. This model is available [here](http://www.mediafire.com/file/jlnktm8yr5y15qm/output_epoch29_bleu0.07394183608962669/file).

The model can be selected by uncommented the correct line in `main.py` :
* WGANBase is the basic WGAN
* WGANGP is the WGAN with gradient penalty
* WGANLip is the WGAN with alternative gradient penalty
* WGAN is the WGAN with gradient penalty and clipping
* RelativisticGAN is the RGAN

## Model training
To train the model, execute the following command:
```sh
$ python main.py config.json
```

## Model test
To test the model, execute the following command:
```sh
$ python test.py config.json
```
This script will create two output files :
* `output_argmax` : contains the  generated captions obtained by selecting arg max words.
* `output_beam` : contains the  generated captions obtained by using beam search.

To get Top 5 and Flop 5 for one of these files, execute the following command :
```sh
$ python evaluator.py cocodataset/captions/beam.en output_argmax cocodataset/links/beam.txt cocodatset/images
```
That will generate 10 PNG files containing these results.

## Notes
* As explained [here](https://pytorch.org/docs/stable/notes/randomness.html), reproducibility is not guaranteed across PyTorch releases, individual commits or different platforms. Furthermore, results need not be reproducible between CPU and GPU executions, even when using identical seeds.
* Training data set is limited to 50.000 instances. Removing this limit is possible by modifying :
	* the `datasets/caption.py` :  
	```Python
	if mode == 'train':
	    captions = captions[:50000]
	```
	* the `datasets/numpy.py` :
	```Python
	if mode == 'train':
	    self.data = self.data[:50000]
	```
	* the `datasets/text.py` :
	```Python
	if mode == 'train':
	    self.data = self.data[:50000]
	```
