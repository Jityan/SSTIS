from PIL import Image
import PIL
import os
import os.path
import pickle
import random
import numpy as np
import pandas as pd

import torch
import torchvision.transforms as transforms



class CUBTextDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir='./data/cub', split='train', embedding_type='cnn-rnn', imsize=64, transform=None, target_transform=None):

        self.imsize = imsize
        if transform is None:
            if split == 'train':
                self.transform = transforms.Compose([
                    transforms.RandomCrop(self.imsize),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((self.imsize, self.imsize)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.data_dir = data_dir
        if data_dir.find('cub') != -1:
            self.bbox = self.load_bbox()
            print("Bounding box loaded...")
        else:
            self.bbox = None
        self.bbox = self.load_bbox()
        split_dir = os.path.join(data_dir, split)

        self.filenames = self.load_filenames(split_dir)
        self.embeddings = self.load_embedding(split_dir, embedding_type)
        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        self.captions = self.load_all_captions()

    def get_img(self, img_path, bbox):
        img = Image.open(img_path).convert('RGB')
        width, height = img.size
        if bbox is not None:
            #print("Crop...")
            R = int(np.maximum(bbox[2], bbox[3]) * 0.75)
            center_x = int((2 * bbox[0] + bbox[2]) / 2)
            center_y = int((2 * bbox[1] + bbox[3]) / 2)
            y1 = np.maximum(0, center_y - R)
            y2 = np.minimum(height, center_y + R)
            x1 = np.maximum(0, center_x - R)
            x2 = np.minimum(width, center_x + R)
            img = img.crop([x1, y1, x2, y2])
        load_size = int(self.imsize * 76 / 64)
        img = img.resize((load_size, load_size), PIL.Image.BILINEAR)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        #print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()
            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox

    def load_all_captions(self):
        caption_dict = {}
        for key in self.filenames:
            caption_name = '%s/text_c10/%s.txt' % (self.data_dir, key)
            captions = self.load_captions(caption_name)
            caption_dict[key] = captions
        return caption_dict

    def load_captions(self, caption_name):
        cap_path = caption_name
        with open(cap_path, "r") as f:
            captions = f.read().split('\n')
        captions = [cap.replace("\ufffd\ufffd", " ")
                    for cap in captions if len(cap) > 0]
        return captions

    def load_embedding(self, data_dir, embedding_type):
        if embedding_type == 'cnn-rnn':
            embedding_filename = '/char-CNN-RNN-embeddings.pickle'
        elif embedding_type == 'cnn-gru':
            embedding_filename = '/char-CNN-GRU-embeddings.pickle'
        elif embedding_type == 'skip-thought':
            embedding_filename = '/skip-thought-embeddings.pickle'

        with open(data_dir + embedding_filename, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            embeddings = u.load()
            #embeddings = pickle.load(f)
            embeddings = np.array(embeddings)
            # embedding_shape = [embeddings.shape[-1]]
            #print('embeddings: ', embeddings.shape)
        return embeddings

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                u = pickle._Unpickler(f)
                u.encoding = 'latin1'
                class_id = u.load()
                #class_id = pickle.load(f)
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir):
        filepath = os.path.join(data_dir, 'filenames.pickle')
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)
        #print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        return filenames
    
    def load_wrong_images(self, cls_id):
        temp_id = random.randint(0, len(self.filenames)-1)
        w_id = self.class_id[temp_id]
        if cls_id != w_id:
            return self.filenames[temp_id]
        return self.load_wrong_images(cls_id)

    def __getitem__(self, index):
        key = self.filenames[index]
        cls_id = self.class_id[index]
        #print(cls_id)
        wkey = self.load_wrong_images(cls_id)
        #
        if self.bbox is not None:
            bbox = self.bbox[key]
            wbbox = self.bbox[wkey]
            data_dir = '%s/CUB_200_2011' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir

        captions = self.captions[key]
        embeddings = self.embeddings[index, :, :]
        # if is CUB, else remove CUB_200_2011/
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        wimg_name = '%s/images/%s.jpg' % (data_dir, wkey)
        img = self.get_img(img_name, bbox)
        wimg = self.get_img(wimg_name, wbbox)

        embedding_ix = random.randint(0, embeddings.shape[0]-1)
        embedding = embeddings[embedding_ix, :]
        caption = captions[embedding_ix]
        if self.target_transform is not None:
            embedding = self.target_transform(embedding)
        idata = {
            'right_images': img,
            'wrong_images': wimg,
            'right_embed': embedding,
            'txt': str(caption),
            'cid': cls_id,
        }
        return idata

    def __len__(self):
        return len(self.filenames)

if __name__ == "__main__":
    image_transform = transforms.Compose([
        transforms.RandomCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = CUBTextDataset('./data/oxford', 'train', imsize=64, transform=image_transform)
    assert dataset
    print(len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, drop_last=True, shuffle=True, num_workers=0)

    for i, sample in enumerate(dataloader, 1):
        rimg = sample['right_images']
        wimg = sample['wrong_images']
        emb = sample['right_embed']
        txt = sample['txt']
        cid = sample['cid']
        print(i, ":", rimg.shape, wimg.shape, emb.shape, len(txt))
        break
        if i > 10:
            break
    print("Complete...")