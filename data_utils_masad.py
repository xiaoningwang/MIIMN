import os
import pickle
import random
import numpy as np
import sys
import json
import copy
import torch
import torchvision
# from data_processing.clean import clean_str, process_text
from clean import clean_str, process_text
from PIL import Image
from PIL import ImageFile
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import alexnet, resnet18, resnet50, inception_v3


base_path = sys.path[0] + "/data/"
sentiment_map = {
    'positive': 2,
    'neutral': 1,
    'negative': 0
}


def load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', errors='ignore')
    word_vec = {}
    for line in fin.readlines():
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            word_vec[tokens[0]] = np.asarray(tokens[-300:], dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, type):
    embedding_matrix_file_name = './{0}_{1}_embedding_matrix.dat'.format(str(embed_dim), type)
    if os.path.exists(embedding_matrix_file_name):
        print('loading embedding_matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('loading word vectors...')
        # idx 0 and len(word2idx)+1 are all-zeros
        # idx 0 refers to [PAD] token
        # idx len(wird2udx) refers to [UNKNOWN] token
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))
        fname = base_path + 'store/glove.twitter.27B.' + str(embed_dim) + 'd.txt' \
            if embed_dim != 300 else './datasets/EnglishWordVectors/glove.840B.300d.txt'
        word_vec = load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', embedding_matrix_file_name)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
            else:
                embedding_matrix[i] = np.random.uniform(low=-0.01, high=0.01, size=embed_dim)
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix


def build_aspect_embedding_matrix(word2idx, embed_dim, type):
    aspect_embedding_matrix_file_name = './{0}_{1}_aspect_embedding_matrix.dat'.format(str(embed_dim), type)
    if os.path.exists(aspect_embedding_matrix_file_name):
        print('loading embedding_matrix:', aspect_embedding_matrix_file_name)
        aspect_embedding_matrix = pickle.load(open(aspect_embedding_matrix_file_name, 'rb'))
    else:
        print('loading word vectors...')
        aspect_embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        fname = base_path + 'store/glove.twitter.27B.' + str(embed_dim) + 'd.txt' \
            if embed_dim != 300 else './datasets/EnglishWordVectors/glove.840B.300d.txt'
        word_vec = load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', aspect_embedding_matrix_file_name)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                aspect_embedding_matrix[i] = vec
            else:
                aspect_embedding_matrix[i] = np.random.uniform(low=-0.01, high=0.01, size=embed_dim)
        pickle.dump(aspect_embedding_matrix, open(aspect_embedding_matrix_file_name, 'wb'))
    return aspect_embedding_matrix


class Tokenizer(object):
    '''
    功能：输入cleaned text string，输出token id vector。
    
    示例：(设max_seq_len = 6)
        input: 'I have an apple'
        output: [2, 4, 5, 1, 0, 0]  
    '''
    def __init__(self, lower=False, max_seq_len=None, max_aspect_len=None):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.max_aspect_len = max_aspect_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    @staticmethod
    def pad_sequence(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0.):
        x = (np.ones(maxlen) * value).astype(dtype)
        if truncating == 'pre':
            trunc = sequence[-maxlen:]
        else:
            trunc = sequence[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == 'post':
            x[:len(trunc)] = trunc
        else:
            x[-len(trunc):] = trunc
        return x

    def text_to_sequence(self, text, reverse=False, max_seq_len=-1):
        '''
        input: 'I have an apple'
        output: [2, 4, 5, 1, 0, 0]
        '''
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx) + 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        pad_and_trunc = 'post'  # use post padding together with torch.nn.utils.rnn.pack_padded_sequence
        if reverse:
            sequence = sequence[::-1]
        if max_seq_len == -1:
            max_seq_len = self.max_seq_len
        return Tokenizer.pad_sequence(sequence, max_seq_len, dtype='int64', padding=pad_and_trunc,
                                      truncating=pad_and_trunc)


class ABSADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class ABSADatesetReader:
    @staticmethod
    def __read_text__(fname, dataset):
        '''
        input: 
            fname：数据集路径
            dataset：数据集名称
            
        ouput: 
            text：对原始文本分词后拼接构造的token string，包含文本中的所有token
            aspect_text：对原始aspect word分词后拼接构造的token string，包含aspect words中的所有token
            max_sentence_len；单个样本的最大序列长度
            max_term_len：单个aspect word的最大序列长度
        '''
        with open(fname, 'r') as f:
            data = json.load(f)
        
        '''
        data的数据格式如下：
        
        [
        {"text": "But the staff was so horrible to us.", 
        "id": "3121", 
        "opinions": {
                "aspect_category": [{"category": "service", "polarity": "negative"}], 
                "aspect_term": [{"polarity": "negative", "to": "13", "term": "staff", "from": "8"}]}
        }, 
        {"text": "To be completely fair, the only redeeming factor was the food, which was above average, but couldn't make up for all the other deficiencies of Teodora.", 
        "id": "2777", 
        "opinions": {
                "aspect_category": [{"category": "food", "polarity": "positive"}, {"category": "anecdotes/miscellaneous", "polarity": "negative"}], 
                "aspect_term": [{"polarity": "positive", "to": "61", "term": "food", "from": "57"}]}
        }, 
        {"text": "The food is uniformly exceptional, with a very capable kitchen which will proudly whip up whatever you feel like eating, whether it's on the menu or not.", 
        "id": "1634", 
        "opinions": {
                "aspect_category": [{"category": "food", "polarity": "positive"}], 
                "aspect_term": [{"polarity": "positive", "to": "8", "term": "food", "from": "4"}, {"polarity": "positive", "to": "62", "term": "kitchen", "from": "55"}, {"polarity": "neutral", "to": "145", "term": "menu", "from": "141"}]}
        }
        ]
        '''
        
        text = ''
        aspect_text = ''
        max_sentence_len = 0.0
        max_term_len = 0.0
        
        for instance in data:
            text_instance = instance['text']  # 原始text
            if dataset == "twitter":
                text_instance = text_instance.encode("utf-8")
            # print(text_instance)
            opinion = instance['opinions']
            aspect_terms = opinion['aspect_term']
            for a in aspect_terms:
                aspect = a['term']
                polarity = a['polarity']
                if polarity == "conflict":
                    continue
                from_index = int(a['from'])
                to_index = int(a['to'])
                aspect_clean = " ".join(process_text(aspect))
                if aspect == "null":
                    from_index = 0
                    to_index = 0
                left = text_instance[:from_index]  # 当前aspect word左侧的文本
                right = text_instance[to_index:]  # 当前aspect word右侧的文本
                aspect_tmp = text_instance[from_index: to_index]  # 当前aspect word(还未进行处理)
                if dataset == "twitter":
                    left = left.decode("utf-8")
                    right = right.decode("utf-8")
                    aspect_tmp = aspect_tmp.decode("utf-8")
                if aspect != aspect_tmp and aspect != 'NULL':
                    print(aspect, text_instance[from_index: to_index])
                left_clean = " ".join(process_text(left))
                right_clean = " ".join(process_text(right))
                text_raw = left_clean + " " + aspect_clean + " " + right_clean
                if len(text_raw.split(" ")) > max_sentence_len:
                    max_sentence_len = len(text_raw.split(" "))
                # print(aspect_clean)
                if len(aspect_clean.split(" ")) > max_term_len:
                    max_term_len = len(aspect_clean.split(" "))
                text += text_raw + " "
                aspect_text += aspect_clean + " "
        return text.strip(), aspect_text.strip(), max_sentence_len, max_term_len

    @staticmethod
    def __read_data__(fname, tokenizer, dataset):
        with open(fname, 'r') as f:
            data = json.load(f)

        all_data = []
        for instance in data:
            text_instance = instance['text']
            if dataset == "twitter":
                text_instance = text_instance.encode("utf-8")
            opinion = instance['opinions']
            aspect_terms = opinion['aspect_term']
            for a in aspect_terms:
                aspect = a['term']
                polarity = a['polarity']
                if polarity == "conflict":
                    continue
                from_index = int(a['from'])
                to_index = int(a['to'])
                aspect = " ".join(process_text(aspect))
                if aspect == "null":
                    from_index = 0
                    to_index = 0

                left = text_instance[:from_index]
                right = text_instance[to_index:]
                if dataset == "twitter":
                    left = left.decode("utf-8")
                    right = right.decode("utf-8")
                text_left = " ".join(process_text(left))
                text_right = " ".join(process_text(right))
                text_raw_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
                text_raw_without_aspect_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
                text_left_indices = tokenizer.text_to_sequence(text_left)
                text_left_with_aspect_indices = tokenizer.text_to_sequence(text_left + " " + aspect)
                text_right_indices = tokenizer.text_to_sequence(text_right, reverse=True)
                text_right_with_aspect_indices = tokenizer.text_to_sequence(" " + aspect + " " + text_right,
                                                                            reverse=True)
                aspect_indices = tokenizer.text_to_sequence(aspect, max_seq_len=tokenizer.max_aspect_len)
                polarity = sentiment_map[polarity]
                data = {
                    'text_raw_indices': text_raw_indices,
                    'text_raw_without_aspect_indices': text_raw_without_aspect_indices,
                    'text_left_indices': text_left_indices,
                    'text_left_with_aspect_indices': text_left_with_aspect_indices,
                    'text_right_indices': text_right_indices,
                    'text_right_with_aspect_indices': text_right_with_aspect_indices,
                    'aspect_indices': aspect_indices,
                    'polarity': polarity,
                }

                all_data.append(data)
        return all_data

    def __init__(self, dataset='twitter', embed_dim=300, max_seq_len=-1):
        print("preparing {0} dataset...".format(dataset))
        fname = {
            'twitter': {
                'train': base_path + 'data_processed/Twitter/twitter-train.json',
                'test': base_path + 'data_processed/Twitter/twitter-test.json'
            },
            'restaurants14': {
                'train': base_path + 'data_processed/SemEval2014/restaurants-train.json',
                'test': base_path + 'data_processed/SemEval2014/restaurants-test.json'
            },
            'laptop14': {
                'train': base_path + 'data_processed/SemEval2014/laptop-train.json',
                'test': base_path + 'data_processed/SemEval2014/laptop-test.json'
            },
            'restaurants15': {
                'train': base_path + 'data_processed/SemEval2015/restaurants-train.json',
                'test': base_path + 'data_processed/SemEval2015/restaurants-test.json'
            },
            'restaurants16': {
                'train': base_path + 'data_processed/SemEval2016/restaurants-train.json',
                'test': base_path + 'data_processed/SemEval2016/restaurants-test.json'
            }
        }
        ''' 读取原始数据集，获取token全集，最大序列长度等信息 '''
        text_train, aspect_text_train, max_seq_len_train, max_term_len_train = ABSADatesetReader.__read_text__(
            fname[dataset]['train'], dataset=dataset)
        text_test, aspect_text_test, max_seq_len_test, max_term_len_test = ABSADatesetReader.__read_text__(
            fname[dataset]['test'], dataset=dataset)
        text = text_train + " " + text_test
        # aspect_text = aspect_text_train + " " + aspect_text_test
        if max_seq_len < 0:
            max_seq_len = max_seq_len_train
        
        ''' 根据token全集与最大序列长度等信息，初始化tokenizer '''
        tokenizer_text = Tokenizer(max_seq_len=max_seq_len, max_aspect_len=max_term_len_train)
        tokenizer_text.fit_on_text(text.lower())
        # tokenizer_aspect = Tokenizer(max_seq_len=max_seq_len, max_aspect_len=max_term_len_train)
        # tokenizer_aspect.fit_on_text(aspect_text.lower())
        # print tokenizer_aspect.word2idx
        
        ''' 基于tokenizer构造Word Embedding矩阵与Aspect Embedding矩阵 '''
        self.embedding_matrix = build_embedding_matrix(tokenizer_text.word2idx, embed_dim, dataset)
        self.aspect_embedding_matrix = copy.deepcopy(self.embedding_matrix)
        # #build_aspect_embedding_matrix(tokenizer_text.word2idx, embed_dim, dataset)
        
        ''' 样本处理 '''
        self.train_data = ABSADataset(
            ABSADatesetReader.__read_data__(fname[dataset]['train'], tokenizer_text, dataset=dataset))
        self.test_data = ABSADataset(
            ABSADatesetReader.__read_data__(fname[dataset]['test'], tokenizer_text, dataset=dataset))
        self.dev_data = ABSADataset([])


class MasadDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class MasadDatasetReader:
    @staticmethod
    def __data_Counter__(fnames):
        '''
        fnames = [train_fname, dev_fname, test_fname]
        '''
        min_seq_len = 99999
        max_seq_len = 0
        avg_seq_len = 0
        
        lens = []
        count = 0
        
        for fname in fnames:
            with open(fname, 'r') as f:
                data = json.load(f)
            for sample in data:
                text_raw = sample['text']
                text_clean = " ".join(process_text(text_raw))
                curr_len = len(text_clean.split(" "))
                min_seq_len = min(min_seq_len, curr_len)
                max_seq_len = max(max_seq_len, curr_len)
                avg_seq_len += curr_len
                
                lens.append(curr_len)
                count += 1
        
        print(min_seq_len, avg_seq_len/count, max_seq_len)  # 1 24.365872053872053 3873
        
        lens.sort()
        print(len(lens))  # 37125 samples
        print(lens[len(lens)//4],lens[len(lens)//2],lens[(len(lens)*3)//4])  # 3 8 22
        
        # data num : 37125
        # min_seq_len = 1
        # 25%percentile = 3
        # 50%percentile = 8
        # 75%percentile = 22
        # avg_seq_len = 24.36
        # max_seq_len = 3873
        
        
    @staticmethod
    def __read_text__(fname, dataset='MASAD'):
        '''
        input: 
            fname：数据集路径
            dataset：数据集名称
            
        ouput: 
            text：对原始文本分词清理后拼接构造的token string，包含文本中的所有token
            aspect_text：对原始aspect word分词后拼接构造的token string，包含aspect words中的所有token
            max_sentence_len；单个样本的最大序列长度
            max_term_len：单个aspect word的最大序列长度
        '''
        with open(fname, 'r') as f:
            data = json.load(f)
        
        '''
        data的数据格式如下：
        
        [
        {'aspect': 'autumn',
        'id': '6267952523',
        'polarity': 'positive',
        'text': "nature 's way of painting a href http www . flickr . com photos 29065913 n04 6267952523 sizes l in photostream view large a"
        }
        ]
        '''
        
        text = ''
        aspect_text = ''
        max_sentence_len = 0.0
        max_aspect_len = 0.0
        
        for instance in data:
            text_raw = instance['text']
            aspect_word = instance['aspect']
            text_clean = " ".join(process_text(text_raw))
            aspect_clean = " ".join(process_text(aspect_word))
            
            if len(text_clean.split(" ")) > max_sentence_len:
                max_sentence_len = len(text_clean.split(" "))
            if len(aspect_clean.split(" ")) > max_aspect_len:
                max_aspect_len = len(aspect_clean.split(" "))
                
            text += text_clean + " "
            aspect_text += aspect_clean + " "
            
        return text.strip(), aspect_text.strip(), max_sentence_len, max_aspect_len
    
    def read_img(self, img_path, img_id):  # for MASAD dataset, one sample has one image.
        base_img_path = '/home/ossdata/zhoutao/masad_img/'
        # base_img_path = './dataset_processed/img/'

        try:
            with torch.no_grad():
                # 评论图片路径
                # img = Image.open('/home/xunan/code/pytorch/ZOLspider/multidata_zol/img/' + img_path).convert('RGB')
                img = Image.open(base_img_path + img_path + img_id + '.jpg').convert('RGB')
                input = self.transform_img(img).unsqueeze(0)
                img_feature = self.cnn_extractor(input).squeeze()
                img.close()
        except:
            error = 1
            img_feature = None
            print('image processing error!!!')
        
        return img_feature
    
    # @staticmethod
    def prepare_fine_grain_img(self, fname, dataset, img_path=None):
        '''
        prepare fine-grained img features for MASAD sample, store the feature as pickle file.
        '''
        with open(fname, 'r') as f:
            data = json.load(f)
        
        data_path = fname.split('.json')[0] + '/'  # './datasets/masadDataset/masadTrainData/'
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        data_path += self.cnn_model_name + '_fine_grain' + '/'  # './datasets/masadDataset/masadTrainData/resnet50_fine_grain/'
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        
        base_img_path = '/home/ossdata/zhoutao/masad_img/'
        adaptive_avg_pool = nn.AdaptiveAvgPool2d((3, 3))  # for fine-grained img features processing
        
        total_num = len(data)
        count = 0
        for instance in data:
            count += 1
            img_id = idx = instance['id']  # 样本id，用于获取图片
            sample_name = data_path + idx + '.pkl'
            print(sample_name + ' ; ' + '%d/%d' % (count, total_num))
            
            if not(os.path.exists(sample_name)):
                # prepare fine-grained img features
                try:
                    with torch.no_grad():
                        img = Image.open(base_img_path + img_path + img_id + '.jpg').convert('RGB')
                        input = self.transform_img(img).unsqueeze(0).to(self.device)
                        img_features = self.cnn_extractor(input)  # (1, 2048, M, N)
                        img_features = adaptive_avg_pool(img_features)  # (1, 2048, 3, 3)
                        if img_features.shape != (1, 2048, 3, 3):
                            print('Invalid Img Shape!!!')
                            continue
                        img_features = img_features.squeeze().view(2048, 3*3).permute(1, 0)  # (9, 2048)
                        img.close()
                except:
                    img_features = None
                    print('image processing error!!!')
                    continue
                
                sample = {
                    'imgs_fine_grain': img_features,  # tensor with shape = (9 ,2048)
                    'id': int(idx)
                }
                
                with open(sample_name, 'wb') as sample_pkl:
                    pickle.dump(sample, sample_pkl)

        return
    
    # @staticmethod
    def prepare_scene_level_img(self, fname, dataset, img_path=None):
        '''
        prepare scene-level img features for MASAD sample, store the feature as pickle file.
        '''
        with open(fname, 'r') as f:
            data = json.load(f)
        
        data_path = fname.split('.json')[0] + '/'  # './datasets/masadDataset/masadTrainData/'
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        data_path += self.cnn_model_name + '_scene_level' + '/'  # './datasets/masadDataset/masadTrainData/resnet50_scene_level/'
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        
        base_img_path = '/home/ossdata/zhoutao/masad_img/'
        
        total_num = len(data)
        count = 0
        for instance in data:
            count += 1
            img_id = idx = instance['id']  # 样本id，用于获取图片
            sample_name = data_path + idx + '.pkl'
            print(sample_name + ' ; ' + '%d/%d' % (count, total_num))
            
            if not(os.path.exists(sample_name)):
                # prepare scene-level img features
                try:
                    with torch.no_grad():
                        img = Image.open(base_img_path + img_path + img_id + '.jpg').convert('RGB')
                        input = self.transform_img(img).unsqueeze(0).to(self.device)
                        img_features = self.cnn_extractor(input).squeeze()  # (2048, )
                        img.close()
                except:
                    img_features = None
                    print('image processing error!!!')
                    continue
                
                sample = {
                    'imgs_scene_level': img_features,  # tensor with shape = (2048, )
                    'id': int(idx)
                }
                
                with open(sample_name, 'wb') as sample_pkl:
                    pickle.dump(sample, sample_pkl)

        return
    
    # @staticmethod
    def read_data(self, fname, tokenizer_text, tokenizer_aspect, dataset, img_path=None):
        with open(fname, 'r') as f:
            data = json.load(f)
        
        polarity_dic = {'positive':1, 'negative':0}
        all_data = []
        
        data_path = fname.split('.json')[0] + '/'  # './datasets/masadDataset/masadTrainData/'
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        data_path += self.cnn_model_name + '/'  # './datasets/masadDataset/masadTrainData/resnet50/'
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        
        total_num = len(data)
        count = 0
        for instance in data:
            count += 1

            text_raw = instance['text']  # 原始文本
            aspect = instance['aspect']  # 原始aspect
            polarity = instance['polarity']  # positive/negative
            idx = instance['id']  # 样本id，用于获取图片
            
            sample_name = data_path + idx + '.pkl'
            #print(sample_name + ' ; ' + '%d/%d' % (count, total_num))

            if os.path.exists(sample_name):
                with open(sample_name, 'rb') as sample_pkl:
                    sample = pickle.load(sample_pkl)
                    if 'aspect_id' not in sample:  # check aspect_id
                        aspect = " ".join(process_text(aspect))
                        aspect_id = tokenizer_aspect.word2idx[aspect]
                        sample['aspect_id'] = int(aspect_id)-1  # 0,1,...,56
            else:
                text_raw = " ".join(process_text(text_raw))
                aspect = " ".join(process_text(aspect))
                if len(text_raw) == 0 or len(aspect) == 0:
                    continue
                
                text_raw_indices = tokenizer_text.text_to_sequence(text_raw, max_seq_len=tokenizer_text.max_seq_len)
                aspect_indices_shared = tokenizer_text.text_to_sequence(aspect, max_seq_len=tokenizer_text.max_aspect_len)
                aspect_indices = tokenizer_aspect.text_to_sequence(aspect, max_seq_len=tokenizer_aspect.max_aspect_len)
                polarity = polarity_dic[polarity]
                img_feature = self.read_img(img_path, idx)
                if img_feature is None:
                    continue
                
                aspect_id = tokenizer_aspect.word2idx[aspect]  # aspect_id = 1, 2, ..., 57
                
                sample = {
                    'text_raw_indices': text_raw_indices,
                    'aspect_indices': aspect_indices, # word emb, aspect emb, independent
                    'aspect_indices_shared': aspect_indices_shared, # word emb, aspect emb, shared
                    'aspect_id': int(aspect_id)-1,  # aspect_id = 0, 1, ..., 56
                    'polarity': int(polarity),
                    'imgs': img_feature,
                    'num_imgs': 1,
                    'id': int(idx)
                }
                
                with open(sample_name, 'wb') as sample_pkl:
                    pickle.dump(sample, sample_pkl)

            all_data.append(sample)
        return all_data
    
    # @staticmethod
    def read_data_expanded(self, fname, tokenizer_text, tokenizer_aspect, dataset, img_path=None, signal=None):
        '''
        This function is designed to generate expand-format MASAD samples.
        
        Specifically, we do negative sampling for each Aspect-Real sample to reduce the number of negative samples. 
        
        As a result, each Aspect-Real sample (as positive) would get 3-to-4 Aspect-Null samples (as negative).
        '''
        with open(fname, 'r') as f:
            data = json.load(f)
        
        polarity_dic = {'positive':1, 'negative':0}
        ACSA_label_dic = {'irrelevant':2, 'positive':1, 'negative':0}
        ACD_label_dic = {'relevant':1, 'irrelevant':0}
        ASC_label_dic = {'positive':1, 'negative':0}
        all_data = []
        
        data_path = fname.split('.json')[0] + '/'  # './datasets/masadDataset/masadTrainData/'
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        data_path += self.cnn_model_name + '/'  # './datasets/masadDataset/masadTrainData/resnet50/'
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        
        # signal = 'train', 'dev', 'test'
        expanded_sample_dict_path = './expanded_masad_' + signal + '_sample_dict.npy'
        if os.path.exists(expanded_sample_dict_path):
            expanded_sample_dict = np.load(expanded_sample_dict_path, allow_pickle=True)
            expanded_sample_dict = expanded_sample_dict.item()  # from ndarray to dicionary
        else:
            expanded_sample_dict = None
            temp_sample_dict = {}  # {sample_id : [fist_null_aspect_id, second_null_aspect_id, ...]}
        
        total_num = len(data)
        count = 0
        
        for instance in data:
            count += 1

            text_raw = instance['text']  # 原始文本
            aspect = instance['aspect']  # 原始aspect
            polarity = instance['polarity']  # positive/negative
            idx = instance['id']  # 样本id，用于获取图片, type is str
            
            sample_name = data_path + idx + '.pkl'
            #print(sample_name + ' ; ' + '%d/%d' % (count, total_num))

            if os.path.exists(sample_name):
                with open(sample_name, 'rb') as sample_pkl:
                    sample = pickle.load(sample_pkl)
                    if 'aspect_id' not in sample:  # check aspect_id
                        aspect = " ".join(process_text(aspect))
                        aspect_id = tokenizer_aspect.word2idx[aspect]  # aspect_id = 1, 2, ..., 57
                        sample['aspect_id'] = int(aspect_id)-1  # 0,1,...,56
            else:
                text_raw = " ".join(process_text(text_raw))
                aspect = " ".join(process_text(aspect))
                if len(text_raw) == 0 or len(aspect) == 0:
                    continue
                
                text_raw_indices = tokenizer_text.text_to_sequence(text_raw, max_seq_len=tokenizer_text.max_seq_len)
                aspect_indices_shared = tokenizer_text.text_to_sequence(aspect, max_seq_len=tokenizer_text.max_aspect_len)
                aspect_indices = tokenizer_aspect.text_to_sequence(aspect, max_seq_len=tokenizer_aspect.max_aspect_len)
                polarity = polarity_dic[polarity]
                img_feature = self.read_img(img_path, idx)
                if img_feature is None:
                    continue
                
                aspect_id = tokenizer_aspect.word2idx[aspect]  # aspect_id = 1, 2, ..., 57
                
                sample = {
                    'text_raw_indices': text_raw_indices,
                    'aspect_indices': aspect_indices, # word emb, aspect emb, independent
                    'aspect_indices_shared': aspect_indices_shared, # word emb, aspect emb, shared
                    'aspect_id': int(aspect_id)-1,  # aspect_id = 0, 1, ..., 56
                    'polarity': int(polarity),
                    'imgs': img_feature,
                    'num_imgs': 1,
                    'id': int(idx)
                }
                
                with open(sample_name, 'wb') as sample_pkl:
                    pickle.dump(sample, sample_pkl)
            
            # STEP1 : add ACSA/ACD/ASC_label to current Aspect-Real sample
            sample['acsa_label'] = sample['polarity'] 
            sample['acd_label'] = int(ACD_label_dic['relevant'])
            sample['asc_label'] = sample['polarity']
            all_data.append(sample)
            
            # STEP2 : generate Aspect-Null samples refer to current Aspect-Real sample
            if expanded_sample_dict is not None:
                null_aspect_ids = expanded_sample_dict[sample['id']]  # null_aspect_id = 1,2, ..., 57
            else:
                null_aspect_ids = random.sample(range(1, 58), 4)  # generate 4 Aspect-Null samples
                temp_sample_dict[sample['id']] = null_aspect_ids
            for null_aspect_id in null_aspect_ids:
                if null_aspect_id == sample['aspect_id']+1:  # 避免Aspect-Null与Aspect-Real发生冲突
                    continue
                null_aspect = tokenizer_aspect.idx2word[null_aspect_id]
                tmp_sample = {}
                tmp_sample['text_raw_indices'] = sample['text_raw_indices']
                if 'text_raw_indices_trc' in sample:  # for feat_filter_model
                    tmp_sample['text_raw_indices_trc'] = sample['text_raw_indices_trc']
                tmp_sample['aspect_indices'] = tokenizer_aspect.text_to_sequence(null_aspect, max_seq_len=tokenizer_aspect.max_aspect_len)
                tmp_sample['aspect_indices_shared'] = tokenizer_text.text_to_sequence(null_aspect, max_seq_len=tokenizer_text.max_aspect_len)
                tmp_sample['aspect_id'] = int(null_aspect_id)-1
                tmp_sample['polarity'] = sample['polarity']  # not important cuz this is an Aspect-Null sample
                tmp_sample['imgs'] = sample['imgs']
                tmp_sample['num_imgs'] = sample['num_imgs']
                tmp_sample['id'] = sample['id']
                
                tmp_sample['acsa_label'] = int(ACSA_label_dic['irrelevant'])
                tmp_sample['acd_label'] = int(ACD_label_dic['irrelevant'])
                tmp_sample['asc_label'] = sample['polarity']  # not important
                
                all_data.append(tmp_sample)
            
        if expanded_sample_dict is None:  # save information for Aspect-Null samples
            np.save(expanded_sample_dict_path, temp_sample_dict)
            
        random.shuffle(all_data)
        return all_data
    
    # @staticmethod
    def read_data_expanded_with_FineGrainImg(self, fname, tokenizer_text, tokenizer_aspect, dataset, img_path=None, signal=None):
        '''
        This function is designed to generate expand-format MASAD samples.
        
        Specifically, we do negative sampling for each Aspect-Real sample to reduce the number of negative samples. 
        
        As a result, each Aspect-Real sample (as positive) would get 3-4 Aspect-Null samples (as negative).
        '''
        with open(fname, 'r') as f:
            data = json.load(f)
        
        polarity_dic = {'positive':1, 'negative':0}
        ACSA_label_dic = {'irrelevant':2, 'positive':1, 'negative':0}
        ACD_label_dic = {'relevant':1, 'irrelevant':0}
        ASC_label_dic = {'positive':1, 'negative':0}
        all_data = []
        
        data_path = fname.split('.json')[0] + '/'  # './datasets/masadDataset/masadTrainData/'
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        data_path += self.cnn_model_name + '/'  # './datasets/masadDataset/masadTrainData/resnet50/'
        if not os.path.exists(data_path):
            os.mkdir(data_path)
            
        fine_grain_img_path = data_path[:-1] + '_fine_grain' + '/'  # './datasets/masadDataset/masadTrainData/resnet50_fine_grain/'
        if not os.path.exists(fine_grain_img_path):
            print('Fine-grained img features not prepared !!')
            return
        
        # signal = 'train', 'dev', 'test'
        expanded_sample_dict_path = './expanded_masad_' + signal + '_sample_dict.npy'
        if os.path.exists(expanded_sample_dict_path):
            expanded_sample_dict = np.load(expanded_sample_dict_path, allow_pickle=True)
            expanded_sample_dict = expanded_sample_dict.item()  # from ndarray to dicionary
        else:
            expanded_sample_dict = None
            temp_sample_dict = {}  # {sample_id : [fist_null_aspect_id, second_null_aspect_id, ...]}
        
        total_num = len(data)
        count = 0
        
        for instance in data:
            count += 1

            text_raw = instance['text']  # 原始文本
            aspect = instance['aspect']  # 原始aspect
            polarity = instance['polarity']  # positive/negative
            idx = instance['id']  # 样本id，用于获取图片, type is str
            
            sample_name = data_path + idx + '.pkl'  # base sample
            fine_grain_img_file = fine_grain_img_path + idx + '.pkl'  # extra fine-grained img features
            #print(sample_name + ' ; ' + '%d/%d' % (count, total_num))
            
            if os.path.exists(fine_grain_img_file):
                with open(fine_grain_img_file, 'rb') as img_pkl:
                    img_file = pickle.load(img_pkl)
                    fine_grain_img_feat = img_file['imgs_fine_grain'].to(self.device)
                    del img_file
            else:
                continue

            if os.path.exists(sample_name):
                with open(sample_name, 'rb') as sample_pkl:
                    sample = pickle.load(sample_pkl)
                    if 'aspect_id' not in sample:  # check aspect_id
                        aspect = " ".join(process_text(aspect))
                        aspect_id = tokenizer_aspect.word2idx[aspect]  # aspect_id = 1, 2, ..., 57
                        sample['aspect_id'] = int(aspect_id)-1  # 0,1,...,56
                    if 'imgs_fine_grain' not in sample:  # check fine-grained img features
                        sample['imgs_fine_grain'] = fine_grain_img_feat
            else:
                text_raw = " ".join(process_text(text_raw))
                aspect = " ".join(process_text(aspect))
                if len(text_raw) == 0 or len(aspect) == 0:
                    continue
                
                text_raw_indices = tokenizer_text.text_to_sequence(text_raw, max_seq_len=tokenizer_text.max_seq_len)
                aspect_indices_shared = tokenizer_text.text_to_sequence(aspect, max_seq_len=tokenizer_text.max_aspect_len)
                aspect_indices = tokenizer_aspect.text_to_sequence(aspect, max_seq_len=tokenizer_aspect.max_aspect_len)
                polarity = polarity_dic[polarity]
                img_feature = self.read_img(img_path, idx)
                if img_feature is None:
                    continue
                
                aspect_id = tokenizer_aspect.word2idx[aspect]  # aspect_id = 1, 2, ..., 57
                
                sample = {
                    'text_raw_indices': text_raw_indices,
                    'aspect_indices': aspect_indices, # word emb, aspect emb, independent
                    'aspect_indices_shared': aspect_indices_shared, # word emb, aspect emb, shared
                    'aspect_id': int(aspect_id)-1,  # aspect_id = 0, 1, ..., 56
                    'polarity': int(polarity),
                    'imgs': img_feature,
                    'imgs_fine_grain': fine_grain_img_feat,
                    'num_imgs': 1,
                    'id': int(idx)
                }
                
                with open(sample_name, 'wb') as sample_pkl:
                    pickle.dump(sample, sample_pkl)
            
            # STEP1 : add ACSA/ACD/ASC_label to current Aspect-Real sample
            sample['acsa_label'] = sample['polarity'] 
            sample['acd_label'] = int(ACD_label_dic['relevant'])
            sample['asc_label'] = sample['polarity']
            all_data.append(sample)
            
            # STEP2 : generate Aspect-Null samples refer to current Aspect-Real sample
            if expanded_sample_dict is not None:
                null_aspect_ids = expanded_sample_dict[sample['id']]  # null_aspect_id = 1,2, ..., 57
            else:
                null_aspect_ids = random.sample(range(1, 58), 4)  # generate 4 Aspect-Null samples
                temp_sample_dict[sample['id']] = null_aspect_ids
            for null_aspect_id in null_aspect_ids:
                if null_aspect_id == sample['aspect_id']+1:  # 避免Aspect-Null与Aspect-Real发生冲突
                    continue
                null_aspect = tokenizer_aspect.idx2word[null_aspect_id]
                tmp_sample = {}
                tmp_sample['text_raw_indices'] = sample['text_raw_indices']
                if 'text_raw_indices_trc' in sample:  # for feat_filter_model
                    tmp_sample['text_raw_indices_trc'] = sample['text_raw_indices_trc']
                tmp_sample['aspect_indices'] = tokenizer_aspect.text_to_sequence(null_aspect, max_seq_len=tokenizer_aspect.max_aspect_len)
                tmp_sample['aspect_indices_shared'] = tokenizer_text.text_to_sequence(null_aspect, max_seq_len=tokenizer_text.max_aspect_len)
                tmp_sample['aspect_id'] = int(null_aspect_id)-1
                tmp_sample['polarity'] = sample['polarity']  # not important cuz this is an Aspect-Null sample
                tmp_sample['imgs'] = sample['imgs']
                tmp_sample['imgs_fine_grain'] = sample['imgs_fine_grain']
                tmp_sample['num_imgs'] = sample['num_imgs']
                tmp_sample['id'] = sample['id']
                
                tmp_sample['acsa_label'] = int(ACSA_label_dic['irrelevant'])
                tmp_sample['acd_label'] = int(ACD_label_dic['irrelevant'])
                tmp_sample['asc_label'] = sample['polarity']  # not important
                
                all_data.append(tmp_sample)
            
        if expanded_sample_dict is None:  # save information for Aspect-Null samples
            np.save(expanded_sample_dict_path, temp_sample_dict)
            
        random.shuffle(all_data)
        return all_data
    
    # @staticmethod
    def read_data_expanded_with_FineGrainImg_SceneLevelImg(self, fname, tokenizer_text, tokenizer_aspect, dataset, img_path=None, signal=None):
        '''
        This function is designed to generate expand-format MASAD samples.
        
        Specifically, we do negative sampling for each Aspect-Real sample to reduce the number of negative samples. 
        
        As a result, each Aspect-Real sample (as positive) would get 3-to-4 Aspect-Null samples (as negative).
        '''
        with open(fname, 'r') as f:
            data = json.load(f)
        
        polarity_dic = {'positive':1, 'negative':0}
        ACSA_label_dic = {'irrelevant':2, 'positive':1, 'negative':0}
        ACD_label_dic = {'relevant':1, 'irrelevant':0}
        ASC_label_dic = {'positive':1, 'negative':0}
        all_data = []
        
        data_path = fname.split('.json')[0] + '/'  # './datasets/masadDataset/masadTrainData/'
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        data_path += self.cnn_model_name + '/'  # './datasets/masadDataset/masadTrainData/resnet50/'
        if not os.path.exists(data_path):
            os.mkdir(data_path)
            
        fine_grain_img_path = data_path[:-1] + '_fine_grain' + '/'  # './datasets/masadDataset/masadTrainData/resnet50_fine_grain/'
        if not os.path.exists(fine_grain_img_path):
            print('Fine-grained img features not prepared !!')
            return
        
        scene_level_img_path = data_path[:-1] + '_scene_level' + '/'  # './datasets/masadDataset/masadTrainData/resnet50_scene_level/'
        if not os.path.exists(scene_level_img_path):
            print('Scene-level img features not prepared !!')
            return
        
        # signal = 'train', 'dev', 'test'
        expanded_sample_dict_path = './expanded_masad_' + signal + '_sample_dict.npy'
        if os.path.exists(expanded_sample_dict_path):
            expanded_sample_dict = np.load(expanded_sample_dict_path, allow_pickle=True)
            expanded_sample_dict = expanded_sample_dict.item()  # from ndarray to dicionary
        else:
            expanded_sample_dict = None
            temp_sample_dict = {}  # {sample_id : [fist_null_aspect_id, second_null_aspect_id, ...]}
        
        total_num = len(data)
        count = 0
        
        for instance in data:
            count += 1

            text_raw = instance['text']  # 原始文本
            aspect = instance['aspect']  # 原始aspect
            polarity = instance['polarity']  # positive/negative
            idx = instance['id']  # 样本id，用于获取图片, type is str
            
            sample_name = data_path + idx + '.pkl'  # base sample
            fine_grain_img_file = fine_grain_img_path + idx + '.pkl'  # extra fine-grained img features
            scene_level_img_file = scene_level_img_path + idx + '.pkl'  # extra scene-level img features
            #print(sample_name + ' ; ' + '%d/%d' % (count, total_num))
            
            if os.path.exists(fine_grain_img_file):
                with open(fine_grain_img_file, 'rb') as img_pkl:
                    img_file = pickle.load(img_pkl)
                    fine_grain_img_feat = img_file['imgs_fine_grain'].to(self.device)
                    del img_file
            else:
                continue
            
            if os.path.exists(scene_level_img_file):
                with open(scene_level_img_file, 'rb') as img_pkl:
                    img_file = pickle.load(img_pkl)
                    scene_level_img_feat = img_file['imgs_scene_level'].to(self.device)
                    del img_file
            else:
                continue

            if os.path.exists(sample_name):
                with open(sample_name, 'rb') as sample_pkl:
                    sample = pickle.load(sample_pkl)
                    if 'aspect_id' not in sample:  # check aspect_id
                        aspect = " ".join(process_text(aspect))
                        aspect_id = tokenizer_aspect.word2idx[aspect]  # aspect_id = 1, 2, ..., 57
                        sample['aspect_id'] = int(aspect_id)-1  # 0,1,...,56
                    if 'imgs_fine_grain' not in sample:  # check fine-grained img features
                        sample['imgs_fine_grain'] = fine_grain_img_feat
                    if 'imgs_scene_level' not in sample:  # check scene-level img features
                        sample['imgs_scene_level'] = scene_level_img_feat
            else:
                text_raw = " ".join(process_text(text_raw))
                aspect = " ".join(process_text(aspect))
                if len(text_raw) == 0 or len(aspect) == 0:
                    continue
                
                text_raw_indices = tokenizer_text.text_to_sequence(text_raw, max_seq_len=tokenizer_text.max_seq_len)
                aspect_indices_shared = tokenizer_text.text_to_sequence(aspect, max_seq_len=tokenizer_text.max_aspect_len)
                aspect_indices = tokenizer_aspect.text_to_sequence(aspect, max_seq_len=tokenizer_aspect.max_aspect_len)
                polarity = polarity_dic[polarity]
                img_feature = self.read_img(img_path, idx)
                if img_feature is None:
                    continue
                
                aspect_id = tokenizer_aspect.word2idx[aspect]  # aspect_id = 1, 2, ..., 57
                
                sample = {
                    'text_raw_indices': text_raw_indices,
                    'aspect_indices': aspect_indices, # word emb, aspect emb, independent
                    'aspect_indices_shared': aspect_indices_shared, # word emb, aspect emb, shared
                    'aspect_id': int(aspect_id)-1,  # aspect_id = 0, 1, ..., 56
                    'polarity': int(polarity),
                    'imgs': img_feature,
                    'imgs_fine_grain': fine_grain_img_feat,
                    'imgs_scene_level': scene_level_img_feat,
                    'num_imgs': 1,
                    'id': int(idx)
                }
                
                with open(sample_name, 'wb') as sample_pkl:
                    pickle.dump(sample, sample_pkl)
            
            # STEP1 : add ACSA/ACD/ASC_label to current Aspect-Real sample
            sample['acsa_label'] = sample['polarity'] 
            sample['acd_label'] = int(ACD_label_dic['relevant'])
            sample['asc_label'] = sample['polarity']
            all_data.append(sample)
            
            # STEP2 : generate Aspect-Null samples refer to current Aspect-Real sample
            if expanded_sample_dict is not None:
                null_aspect_ids = expanded_sample_dict[sample['id']]  # null_aspect_id = 1,2, ..., 57
            else:
                null_aspect_ids = random.sample(range(1, 58), 4)  # generate 4 Aspect-Null samples
                temp_sample_dict[sample['id']] = null_aspect_ids
            for null_aspect_id in null_aspect_ids:
                if null_aspect_id == sample['aspect_id']+1:  # 避免Aspect-Null与Aspect-Real发生冲突
                    continue
                null_aspect = tokenizer_aspect.idx2word[null_aspect_id]
                tmp_sample = {}
                tmp_sample['text_raw_indices'] = sample['text_raw_indices']
                if 'text_raw_indices_trc' in sample:  # for feat_filter_model
                    tmp_sample['text_raw_indices_trc'] = sample['text_raw_indices_trc']
                tmp_sample['aspect_indices'] = tokenizer_aspect.text_to_sequence(null_aspect, max_seq_len=tokenizer_aspect.max_aspect_len)
                tmp_sample['aspect_indices_shared'] = tokenizer_text.text_to_sequence(null_aspect, max_seq_len=tokenizer_text.max_aspect_len)
                tmp_sample['aspect_id'] = int(null_aspect_id)-1
                tmp_sample['polarity'] = sample['polarity']  # not important cuz this is an Aspect-Null sample
                tmp_sample['imgs'] = sample['imgs']
                tmp_sample['imgs_fine_grain'] = sample['imgs_fine_grain']
                tmp_sample['imgs_scene_level'] = sample['imgs_scene_level']
                tmp_sample['num_imgs'] = sample['num_imgs']
                tmp_sample['id'] = sample['id']
                
                tmp_sample['acsa_label'] = int(ACSA_label_dic['irrelevant'])
                tmp_sample['acd_label'] = int(ACD_label_dic['irrelevant'])
                tmp_sample['asc_label'] = sample['polarity']  # not important
                
                all_data.append(tmp_sample)
            
        if expanded_sample_dict is None:  # save information for Aspect-Null samples
            np.save(expanded_sample_dict_path, temp_sample_dict)
            
        random.shuffle(all_data)
        return all_data

    def __init__(self, dataset='MASAD', embed_dim=300, max_seq_len=25, max_aspect_len=1, max_img_len=1, cnn_model_name='resnet50', expanded=0, fine_grain_img=0, prepare_fine_grain_img=0, scene_level_img=0, prepare_scene_level_img=0):
        '''
        Params Explanation:
            
        expanded = 0 ---> regular MASAD sample preparing
        expanded = 1 ---> expanded MASAD sample preparing
        
        fine_grain_img = 1 --> involve fine-grained img features when preparing expanded MASAD sample
        prepare_fine_grain_img = 1 --> before training, call the <prepare_fine_grain_img> function to generate fine-grained img features
        
        scene_level_img = 1 --> involve scene-level img features when preparing expanded MASAD sample
        prepare_scene_level_img = 1 --> before training, call the <prepare_scene_level_img> function to generate scene-level img features
        '''
        print("preparing {0} dataset...".format(dataset))
        fname = {
            'MASAD': {
                #'train': './dataset_processed/masadTrainData.json',
                #'dev' : './dataset_processed/masadDevData.json',
                #'test': './dataset_processed/masadTestData.json',
                'train': './datasets/masadDataset/masadTrainData.json',
                'dev' : './datasets/masadDataset/masadDevData.json',
                'test': './datasets/masadDataset/masadTestData.json',
                'train_img': 'train_dev_img/',
                'dev_img': 'train_dev_img/',
                'test_img': 'test_img/'
            }
        }
        
        '''
        cnn_classes = {
            'resnet18': resnet18(pretrained=True),
            'resnet50': resnet50(pretrained=True),
            'alexnet': alexnet(pretrained=True)
        }
        '''
        cnn_classes = {
            'resnet50': resnet50(pretrained=True)
        }
       
        #self.device = torch.device('cpu')
        self.device = torch.device('cuda:0')
        
        if prepare_fine_grain_img == 1:
            print('preparing fine-grained img features...')
            self.cnn_model_name = 'resnet50'
            self.cnn_extractor = nn.Sequential(*list(cnn_classes[self.cnn_model_name].children())[:-2]).to(self.device)  # 去除最后两层预测层，获取细粒度图像特征
            self.transform_img = transforms.Compose([
                transforms.ToTensor(),
                ])
            self.prepare_fine_grain_img(fname[dataset]['train'], dataset=dataset, img_path=fname[dataset]['train_img'])
            self.prepare_fine_grain_img(fname[dataset]['dev'], dataset=dataset, img_path=fname[dataset]['dev_img'])
            self.prepare_fine_grain_img(fname[dataset]['test'], dataset=dataset, img_path=fname[dataset]['test_img'])
            return
        
        if prepare_scene_level_img == 1:
            print('preparing scene-level img features...')
            model_file = './resnet50_places365.pth.tar'
            model = torchvision.models.__dict__['resnet50'](num_classes=365)
            checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
            model.load_state_dict(state_dict)
            self.cnn_model_name = 'resnet50'
            self.cnn_extractor = nn.Sequential(*list(model.children())[:-1]).to(self.device)
            self.transform_img = transforms.Compose([
                transforms.ToTensor(),
                ])
            self.prepare_scene_level_img(fname[dataset]['train'], dataset=dataset, img_path=fname[dataset]['train_img'])
            self.prepare_scene_level_img(fname[dataset]['dev'], dataset=dataset, img_path=fname[dataset]['dev_img'])
            self.prepare_scene_level_img(fname[dataset]['test'], dataset=dataset, img_path=fname[dataset]['test_img'])
            return
            
        ''' 加载预训练图像模型，用于抽取图像特征 '''
        self.cnn_model_name = cnn_model_name
        self.max_img_len = max_img_len
        self.cnn_extractor = nn.Sequential(*list(cnn_classes[cnn_model_name].children())[:-1]).to(self.device)  # 去除最后一层预测层，抽取图像embedding
        self.transform_img = transforms.Compose([
            transforms.ToTensor(),
            ])
        
        ''' 读取原始数据集，获取token全集，最大序列长度等信息 '''
        text_train, aspect_text_train, max_seq_len_train, max_aspect_len_train = MasadDatasetReader.__read_text__(
            fname[dataset]['train'], dataset=dataset)
        text_dev, aspect_text_dev, max_seq_len_dev, max_aspect_len_dev = MasadDatasetReader.__read_text__(
            fname[dataset]['dev'], dataset=dataset)
        text_test, aspect_text_test, max_seq_len_test, max_aspect_len_test = MasadDatasetReader.__read_text__(
            fname[dataset]['test'], dataset=dataset)
        
        aspect_text = aspect_text_train + " " + aspect_text_dev + " " + aspect_text_test
        text = text_train + " " + text_dev + " " + text_test + " " + aspect_text
        
        if max_seq_len < 0:
            max_seq_len = max_seq_len_train
        if max_aspect_len < 0:
            max_aspect_len = max_aspect_len_train
            
        #print(max_seq_len_train, max_aspect_len_train)  # 3873 1
        #print(max_seq_len_dev, max_aspect_len_dev)  # 902 1
        #print(max_seq_len_test, max_aspect_len_test)  # 1521 1

        
        ''' 根据token_text与max_seq/aspect_len等信息，初始化tokenizer '''
        tokenizer_text = Tokenizer(max_seq_len=max_seq_len, max_aspect_len=max_aspect_len)
        tokenizer_text.fit_on_text(text.lower())
        tokenizer_aspect = Tokenizer(max_seq_len=max_seq_len, max_aspect_len=max_aspect_len)
        tokenizer_aspect.fit_on_text(aspect_text.lower())
        self.aspect2idx = tokenizer_aspect.word2idx
        self.idx2aspect = tokenizer_aspect.idx2word
        self.word2idx = tokenizer_text.word2idx
        self.idx2word = tokenizer_text.idx2word
        print(self.aspect2idx)
        #print(len(tokenizer_text.word2idx))
        #print(len(tokenizer_aspect.word2idx))
        #print(tokenizer_aspect.word2idx)
        
        ''' 基于tokenizer构造Word Embedding矩阵与Aspect Embedding矩阵 '''
        self.embedding_matrix = build_embedding_matrix(tokenizer_text.word2idx, embed_dim, dataset)
        # self.aspect_embedding_matrix = copy.deepcopy(self.embedding_matrix)
        self.aspect_embedding_matrix = build_aspect_embedding_matrix(tokenizer_aspect.word2idx, embed_dim, dataset)
        
        ''' 样本处理 '''
        print('if share word emb with aspect emb, please use aspect_indices_shared, else use asepct_indices!!!')
        print('if share word emb with aspect emb, please use aspect_indices_shared, else use asepct_indices!!!')
        print('if share word emb with aspect emb, please use aspect_indices_shared, else use asepct_indices!!!\n')
        if expanded == 1:
            print('You are going to use Expand-Format MASAD data!!!')
            print('You are going to use Expand-Format MASAD data!!!')
            print('You are going to use Expand-Format MASAD data!!!\n')
            if fine_grain_img == 1 and scene_level_img == 0:  # get fine-grained img features
                print('Involving fine-grained img features ...')
                print('Involving fine-grained img features ...')
                print('Involving fine-grained img features ...\n')
                self.train_data = MasadDataset(
                        self.read_data_expanded_with_FineGrainImg(fname[dataset]['train'], tokenizer_text, tokenizer_aspect, dataset=dataset, img_path=fname[dataset]['train_img'], signal='train'))
                self.dev_data = MasadDataset(
                        self.read_data_expanded_with_FineGrainImg(fname[dataset]['dev'], tokenizer_text, tokenizer_aspect, dataset=dataset, img_path=fname[dataset]['dev_img'], signal='dev'))
                self.test_data = MasadDataset(
                        self.read_data_expanded_with_FineGrainImg(fname[dataset]['test'], tokenizer_text, tokenizer_aspect, dataset=dataset, img_path=fname[dataset]['test_img'], signal='test'))
            elif fine_grain_img == 1 and scene_level_img == 1:  # get fine-grained and scene-level img features
                print('Involving fine-grained and scene-level img features ...')
                print('Involving fine-grained and scene-level img features ...')
                print('Involving fine-grained and scene-level img features ...\n')
                self.train_data = MasadDataset(
                        self.read_data_expanded_with_FineGrainImg_SceneLevelImg(fname[dataset]['train'], tokenizer_text, tokenizer_aspect, dataset=dataset, img_path=fname[dataset]['train_img'], signal='train'))
                self.dev_data = MasadDataset(
                        self.read_data_expanded_with_FineGrainImg_SceneLevelImg(fname[dataset]['dev'], tokenizer_text, tokenizer_aspect, dataset=dataset, img_path=fname[dataset]['dev_img'], signal='dev'))
                self.test_data = MasadDataset(
                        self.read_data_expanded_with_FineGrainImg_SceneLevelImg(fname[dataset]['test'], tokenizer_text, tokenizer_aspect, dataset=dataset, img_path=fname[dataset]['test_img'], signal='test'))
            else:
                self.train_data = MasadDataset(
                        self.read_data_expanded(fname[dataset]['train'], tokenizer_text, tokenizer_aspect, dataset=dataset, img_path=fname[dataset]['train_img'], signal='train'))
                self.dev_data = MasadDataset(
                        self.read_data_expanded(fname[dataset]['dev'], tokenizer_text, tokenizer_aspect, dataset=dataset, img_path=fname[dataset]['dev_img'], signal='dev'))
                self.test_data = MasadDataset(
                        self.read_data_expanded(fname[dataset]['test'], tokenizer_text, tokenizer_aspect, dataset=dataset, img_path=fname[dataset]['test_img'], signal='test'))
        else:
            self.train_data = MasadDataset(
                    self.read_data(fname[dataset]['train'], tokenizer_text, tokenizer_aspect, dataset=dataset, img_path=fname[dataset]['train_img']))
            self.dev_data = MasadDataset(
                    self.read_data(fname[dataset]['dev'], tokenizer_text, tokenizer_aspect, dataset=dataset, img_path=fname[dataset]['dev_img']))
            self.test_data = MasadDataset(
                    self.read_data(fname[dataset]['test'], tokenizer_text, tokenizer_aspect, dataset=dataset, img_path=fname[dataset]['test_img']))
        
        ''' 去除测试集的重复样本 '''
        print('Sample Deduplicating ... ')
        train_sample_ids = {}
        for sample in self.train_data:
            train_sample_ids[sample['id']] = 1
        deduplicate_test_data = []
        for sample in self.test_data:
            if sample['id'] not in train_sample_ids:
                deduplicate_test_data.append(sample)
        self.test_data = deduplicate_test_data
        
        # check fine-grained img feat
        if fine_grain_img == 1:
            for i in range(len(self.train_data)):
                if 'imgs_fine_grain' not in self.train_data[i]:
                    print('fine_grain_imgs not in train sample! idx = %d' % i)
                else:
                    if self.train_data[i]['imgs_fine_grain'].shape != (9, 2048):
                        print('shape error! idx = %d' % i)
                        print('shape = ', self.train_data[i]['imgs_fine_grain'].shape)
            for i in range(len(self.dev_data)):
                if 'imgs_fine_grain' not in self.dev_data[i]:
                    print('fine_grain_imgs not in dev sample! idx = %d' % i)
                else:
                    if self.dev_data[i]['imgs_fine_grain'].shape != (9, 2048):
                        print('shape error! idx = %d' % i)
                        print('shape = ', self.dev_data[i]['imgs_fine_grain'].shape)
            for i in range(len(self.test_data)):
                if 'imgs_fine_grain' not in self.test_data[i]:
                    print('fine_grain_imgs not in test sample! idx = %d' % i)
                else:
                    if self.test_data[i]['imgs_fine_grain'].shape != (9, 2048):
                        print('shape error! idx = %d' % i)
                        print('shape = ', self.test_data[i]['imgs_fine_grain'].shape)
        
        # check scene-level img feat
        if scene_level_img == 1:
            for i in range(len(self.train_data)):
                if 'imgs_scene_level' not in self.train_data[i]:
                    print('imgs_scene_level not in train sample! idx = %d' % i)
            for i in range(len(self.dev_data)):
                if 'imgs_scene_level' not in self.dev_data[i]:
                    print('imgs_scene_level not in dev sample! idx = %d' % i)
            for i in range(len(self.test_data)):
                if 'imgs_scene_level' not in self.test_data[i]:
                    print('imgs_scene_level not in test sample! idx = %d' % i)
                        
            
        print('train_data_num: %d' % len(self.train_data))
        print('dev_data_num: %d' % len(self.dev_data))
        print('test_data_num: %d' % len(self.test_data))
        
        return
        
if __name__ == '__main__':    
    #MasadDatasetReader(dataset='MASAD', embed_dim=300, max_seq_len=25)
    #MasadDatasetReader(dataset='MASAD', embed_dim=300, max_seq_len=25, prepare_fine_grain_img=1)
    #MasadDatasetReader(dataset='MASAD', embed_dim=300, max_seq_len=25, prepare_scene_level_img=1)
    
    # MasadDatasetReader.__data_Counter__(['./dataset_processed/masadTrainData.json', './dataset_processed/masadDevData.json', './dataset_processed/masadTestData.json'])
