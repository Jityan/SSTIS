from PIL import Image
import PIL
import os
import os.path
import pickle
import random
import numpy as np

import torch
import torchvision.transforms as transforms


def processLabel(labels):
    newlabels = []
    temp = {}
    count = 0
    for _, label in enumerate(labels):
        if label not in temp:
            temp[label] = count
            count += 1
        newlabels.append(temp[label])
    return newlabels, count


class OxfordTextDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir='./data/oxford', split='train', embedding_type='cnn-rnn', imsize=64, transform=None, target_transform=None):

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
        split_dir = os.path.join(data_dir, split)

        self.filenames = self.load_filenames(split_dir)
        self.embeddings = self.load_embedding(split_dir, embedding_type)
        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        # preprocess class label into 0 - max-1
        self.new_class_id, self.num_classes = processLabel(self.class_id)
        self.num_classes += 1

    def get_img(self, img_path):
        img = Image.open(img_path).convert('RGB')
        load_size = int(self.imsize * 76 / 64)
        img = img.resize((load_size, load_size), PIL.Image.BILINEAR)
        if self.transform is not None:
            img = self.transform(img)
        return img

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
        w_id = self.new_class_id[temp_id]
        if cls_id != w_id:
            return self.filenames[temp_id], w_id
        return self.load_wrong_images(cls_id)
    
    def readCaptions(self, filenames, class_id):
        name = filenames
        class_name = 'class_{0:05d}/'.format(class_id)
        name = name.replace('jpg/', class_name)
        cap_path = '{}/text_c10/{}.txt'.format(self.data_dir, name)
        
        with open(cap_path, "r") as f:
            captions = f.read().split('\n')
        captions = [cap for cap in captions if len(cap) > 0]
        return captions

    def __getitem__(self, index):
        key = self.filenames[index]
        cls_id = self.new_class_id[index]
        wkey, wcls_id = self.load_wrong_images(cls_id)
        #
        data_dir = self.data_dir

        #captions = self.captions[key]
        captions = self.readCaptions(key, self.class_id[index])
        embeddings = self.embeddings[index, :, :]
        img_name = '%s/%s.jpg' % (data_dir, key)
        wimg_name = '%s/%s.jpg' % (data_dir, wkey)
        img = self.get_img(img_name)
        wimg = self.get_img(wimg_name)

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
            'wcid': wcls_id,
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
    dataset = OxfordTextDataset('./data/oxford', 'train', imsize=64, transform=image_transform)
    assert dataset
    print(len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, drop_last=True, shuffle=True, num_workers=0)

    for i, sample in enumerate(dataloader, 1):
        print(i, ":", sample['right_images'].shape, sample['wrong_images'].shape, sample['right_embed'].shape, sample['txt'])
        break
    print("Complete...")