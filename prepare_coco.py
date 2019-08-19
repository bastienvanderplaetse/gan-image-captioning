import json
import numpy as np
import os
import string

from ImageFactory import ImageFactory
from PIL import Image as PIL_Image
from pycocotools.coco import COCO
from utils import explorer_helper as exh

try:
    from urllib.request import urlretrieve, urlopen
except ImportError:
    from urllib import urlretrieve
    from urllib2 import urlopen
import urllib

from socket import error as SocketError
import errno

MAX_SIZE = 0

MAIN_FOLDER = "cocodataset"
IMAGE_FOLDER = "{}/images".format(MAIN_FOLDER)
CAPTIONS_FOLDER = "{}/captions".format(MAIN_FOLDER)
LINKS_FOLDER = "{}/links".format(MAIN_FOLDER)
FEATURES_FOLDER = "{}/features".format(MAIN_FOLDER)

IMAGE_FILE = IMAGE_FOLDER + "/COCO_{}2014_{}.jpg"
ID_STR_IMAGE = "COCO_{}2014_{}.jpg"
CAPTIONS_FILE = CAPTIONS_FOLDER + "/{}"
LINKS_FILE = LINKS_FOLDER + "/{}"
FEATURES_FILE = FEATURES_FOLDER + "/{}"

COCO_LINK = "http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip"
ZIP_NAME = "captions_train-val2014.zip"
TRAIN_FILE = "annotations/captions_train2014.json"
VAL_FILE = "annotations/captions_val2014.json"

FEATS = dict()

def prepare_directories():
    exh.create_directory(MAIN_FOLDER)
    exh.create_directory(IMAGE_FOLDER)
    exh.create_directory(CAPTIONS_FOLDER)
    exh.create_directory(LINKS_FOLDER)
    exh.create_directory(FEATURES_FOLDER)

def download_file():
    if not exh.file_exists(ZIP_NAME):
      os.system("wget http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip")

    if not exh.file_exists(TRAIN_FILE) or not exh.file_exists(VAL_FILE):
        os.system('unzip captions_train-val2014.zip')

def format_caption(caption):
    norm_caption = "".join([w for w in caption if w not in string.punctuation])
    norm_caption = norm_caption.replace("  ", " ").lower()
    if norm_caption[-1] == " ":
        norm_caption = norm_caption[:-1]

    return norm_caption.split('\n')[0]

def download_image(image_id, set_name, coco):
    image_filename = IMAGE_FILE.format(set_name, '0'*(12-len(str(image_id)))+str(image_id))
    if not exh.file_exists(image_filename):
        try:
            img_url = coco.loadImgs(image_id)[0]['flickr_url']
            f = open(IMAGE_FILE.format(image_id), 'wb')
            f.write(urllib.request.urlopen(img_url).read())
            f.close()
            return True
        except urllib.error.HTTPError as e:
            if e.code >= 400 and e.code < 500:
                return False
            raise
    return True

def extract_features(image_id, set_name, image_factory):
    if not image_id in FEATS:
        img = PIL_Image.open(IMAGE_FILE.format(set_name, '0'*(12-len(str(image_id)))+str(image_id)))
        if img.mode != 'RGB':
            img = img.convert(mode='RGB')
        feats = np.array(image_factory.get_features(img)).squeeze()
        FEATS[image_id] = feats
    return FEATS[image_id]

def format_set(set_name, filename, image_factory):
    set_file = exh.load_json(filename)
    captions = []
    links = []
    features = []
    beam = dict()

    coco = COCO(filename)

    tot = len(set_file['annotations'])
    el = 1

    for row in set_file['annotations']:
        print("{}/{}".format(el, tot))
        el += 1
        image_id = row['image_id']
        str_id = ID_STR_IMAGE.format(set_name, '0' * (12-len(str(image_id))) + str(image_id))
        is_ok = download_image(image_id, set_name, coco)

        if is_ok:
            caption = format_caption(row['caption'])
            captions.append(caption)
            links.append(str_id)

            feats = extract_features(image_id, set_name, image_factory)
            features.append(feats)

            if set_name == "val":
                if str_id in beam:
                    beam[str_id]["captions"].append(caption)
                else:
                    beam[str_id] = {
                        "captions": [caption],
                        "feats": feats
                    }

    if MAX_SIZE > 0:
            captions = captions[:MAX_SIZE]
    captions = '\n'.join(captions)
    exh.write_text(captions, CAPTIONS_FILE.format("{}.en".format(set_name)))
    if MAX_SIZE > 0:
            links = links[:MAX_SIZE]
    links = '\n'.join(links)
    exh.write_text(links, LINKS_FILE.format("{}.txt".format(set_name)))
    if MAX_SIZE > 0:
            features = features[:MAX_SIZE]
    features = np.array(features)
    np.save(FEATURES_FILE.format(set_name), features)

    if set_name == "val":
        captions = []
        links = []
        features = []

        for k, v in beam.items():
            links.append(str(k))
            captions.append(v['captions'])
            features.append(v['feats'])

        captions = ["###".join(sentences) for sentences in captions]
        captions = '\n'.join(captions)
        if MAX_SIZE > 0:
            captions = captions[:MAX_SIZE]
        exh.write_text(captions, CAPTIONS_FILE.format("beam.en"))
        if MAX_SIZE > 0:
            links = links[:MAX_SIZE]
        links = '\n'.join(links)
        exh.write_text(links, LINKS_FILE.format("beam.txt"))
        if MAX_SIZE > 0:
            features = features[:MAX_SIZE]
        features = np.array(features)
        np.save(FEATURES_FILE.format("beam"), features)

def run():
    prepare_directories()

    download_file()

    image_factory = ImageFactory(resize=256,crop=224)

    print("Formatting train set")
    format_set("train", TRAIN_FILE, image_factory)
    print("Formatting val set")
    format_set("val", VAL_FILE, image_factory)    

if __name__ == "__main__":
    run()
