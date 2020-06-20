import bisect
import gc
import glob
import random
import re
from copy import deepcopy
import torch
from pattern.en import singularize
from nltk.corpus import wordnet
from nltk import word_tokenize
import nltk

from others.logging import logger


grammar_tweek_negation = {
    'is': '',
    'Is': 'Are',
    'was': 'were',
    'Was': 'Were',
    'are': 'is',
    'Are': 'Is',
    'were': 'was',
    'Were': 'Was',
    'has': 'have',
    'Has': 'Have',
    'have': 'has',
    'Have ': 'Has',
    'do': 'does',
    'Do': 'Does',
    'does': 'do',
    'Does': 'Do',

    'isn\'t': 'aren\'t',
    'Isn\'t': 'Aren\'t',
    'wasn\'t': 'weren\'t',
    'Wasn\'t': 'Weren\'t',
    'aren\'t': 'isn\'t',
    'Aren\'t': 'Isn\'t',
    'weren\'t': 'wasn\'t',
    'Weren\'t': 'Wasn\'t',
    'hasn\'t': 'haven\'t',
    'Hasn\'t': 'Haven\'t',
    'haven\'t': 'hasn\'t',
    'Haven\'t': 'Hasn\'t',
    'don\'t': 'doesn\'t',
    'Don\'t': 'Doesn\'t',
    'doesn\'t': 'don\'t',
    'Doesn\'t': 'Don\'t',
}

grammar_tweek_custom = {

    'on': 'in',
    'On': 'In',
    'in': 'on',
    'In': 'On',
    'at': 'in',
    'At': 'In'

}

semantic_change_simple = {
    'is': 'is not',
    'Is': 'Is not',
    'was': 'was not',
    'Was': 'Was not',
    'are': 'are not',
    'Are': 'Are not',
    'were': 'were not',
    'Were': 'Were not',
    'has': 'has not',
    'Has': 'Has not',
    'have': 'have not',
    'Have ': 'Have not',
    'had': 'had not',
    'Had': 'Had not',
    'do': 'do not',
    'Do': 'Do not',
    'does': 'does not',
    'Does': 'Does not',
    'can': 'can not',
    'Can': 'Can not',
    'could': 'could not',
    'Could': 'Could not',
    'will': 'will not',
    'Will': 'Will not',
    'would': 'would not',
    'Would': 'Would not',
    'should': 'should not',
    'Should': 'Should not',

    'isn\'t': 'is',
    'Isn\'t': 'Is',
    'wasn\'t': 'was',
    'Wasn\'t': 'Was',
    'aren\'t': 'are',
    'Aren\'t': 'Are',
    'weren\'t': 'were',
    'Weren\'t': 'Were',
    'hasn\'t': 'has',
    'Hasn\'t': 'Has',
    'haven\'t': 'have',
    'Haven\'t': 'Have',
    'don\'t': 'do',
    'Don\'t': 'Do',
    'doesn\'t': 'does',
    'Doesn\'t': 'Does',
    'can\'t': 'can',
    'Can\'t': 'Can',
    'couldn\'t': 'could',
    'Couldn\'t': 'Could',
    'won\'t': 'will',
    'Won\'t': 'Will',
    'wouldn\'t': 'would',
    'Wouldn\'t': 'Would',
    'shouldn\'t': 'should',
    'Shouldn\'t': 'Should',
}


class Batch(object):
    def _pad(self, data, pad_id, width=-1):
        if (width == -1):
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data

    def __init__(self, data=None, device=None, is_test=False):
        """Create a Batch from a list of examples."""
        if data is not None:
            self.batch_size = len(data)
            pre_src = [x[0] for x in data]
            pre_tgt = [x[1] for x in data]
            pre_segs = [x[2] for x in data]
            pre_clss = [x[3] for x in data]
            pre_src_sent_labels = [x[4] for x in data]

            src = torch.tensor(self._pad(pre_src, 0))
            tgt = torch.tensor(self._pad(pre_tgt, 0))

            segs = torch.tensor(self._pad(pre_segs, 0))
            mask_src = 1 - (src == 0)
            mask_tgt = 1 - (tgt == 0)


            clss = torch.tensor(self._pad(pre_clss, -1))
            src_sent_labels = torch.tensor(self._pad(pre_src_sent_labels, 0))
            mask_cls = 1 - (clss == -1)
            clss[clss == -1] = 0
            setattr(self, 'clss', clss.to(device))
            setattr(self, 'mask_cls', mask_cls.to(device))
            setattr(self, 'src_sent_labels', src_sent_labels.to(device))


            setattr(self, 'src', src.to(device))
            setattr(self, 'tgt', tgt.to(device))
            setattr(self, 'segs', segs.to(device))
            setattr(self, 'mask_src', mask_src.to(device))
            setattr(self, 'mask_tgt', mask_tgt.to(device))


            if (is_test):
                src_str = [x[-2] for x in data]
                setattr(self, 'src_str', src_str)
                tgt_str = [x[-1] for x in data]
                setattr(self, 'tgt_str', tgt_str)

    def __len__(self):
        return self.batch_size




def load_dataset(args, corpus_type, shuffle, tokenizer):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid", "test"]

    def _lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt_file, len(dataset)))
        for i, data in enumerate(dataset):
            # replace <q> with period
            dataset[i]['tgt_txt'] = re.sub('<q>',' . ', dataset[i]['tgt_txt'])
            dataset[i]['tgt'] = [1] + tokenizer.encode(dataset[i]['tgt_txt']) + [2]
        return dataset

    def _lazy_dataset_loader_with_perturbation(pt_file, corpus_type, tokenizer, perturbation_type):
        dataset = torch.load(pt_file)
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt_file, len(dataset)))
        random.seed(13)
        

        for i, data in enumerate(dataset): # dataset is a list of dictionary with keys = ['src', 'src_sent_labels', 'segs', 'src_txt', 'tgt_txt']  
            dataset[i]['tgt_txt'] = re.sub('<q>',' . ', dataset[i]['tgt_txt'])
            tgt_txt = dataset[i]['tgt_txt'].split()
            original_tgt_txt = deepcopy(tgt_txt)
            
            if perturbation_type == 'semantic':
                change = 0
                tokenized_text = word_tokenize(' '.join(tgt_txt))
                pos_tag = nltk.pos_tag(tokenized_text)
                for pi in range(len(pos_tag)):
                    antonym = ''
                    for syn in wordnet.synsets(pos_tag[pi][0]):
                        for l in syn.lemmas():
                            if l.antonyms():
                                antonym = l.antonyms()[0].name() # get the first antonym of the first lemma
                                break
                        if antonym != '':
                            if change < 2:
                                tokenized_text[pi] = antonym
                                change += 1
                                break
                tgt_txt = tokenized_text

                if tgt_txt == original_tgt_txt:
                    change = 0
                    for k in range(len(tgt_txt)):
                        try:
                            tgt_txt[k] = semantic_change_simple[tgt_txt[k]]
                            change += 1
                        except:
                            pass
                        if change >= 2:
                            break
                        
                dataset[i]['tgt_txt'] = ' '.join(tgt_txt)   
            
            elif perturbation_type == 'syntax':
                sentence_len = len(tgt_txt)
                pos1 = 0
                pos2 = -1
                done = False
                while not done:
                    pos1 += 1
                    pos2 -= 1
                    tgt_txt[pos1] = original_tgt_txt[pos2]
                    tgt_txt[pos2] = original_tgt_txt[pos1]
                    done = True
                    if original_tgt_txt == tgt_txt:
                        done = False
                        
                dataset[i]['tgt_txt'] = ' '.join(tgt_txt)   
 
            elif perturbation_type == 'grammar':
                change = 0
                for k in range(len(tgt_txt)):
                    try:
                        tgt_txt[k] = grammar_tweek_negation[tgt_txt[k]]
                        change += 1
                    except:
                        pass
                    if change >=2 :
                        break

            
                if tgt_txt == original_tgt_txt:
                    change =0
                    for k in range(len(tgt_txt)):
                        try:
                            tgt_txt[k] = grammar_tweek_custom[tgt_txt[k]]
                            change += 1
                        except:
                            pass
                        if change >= 2:
                            break
            
                if tgt_txt == original_tgt_txt:
                    tgt_txt = []
                    change = 0
                    for word in original_tgt_txt:
                        if change >= 2:
                            tgt_txt.append(word)
                        else:
                            new_word = singularize(word)
                            tgt_txt.append(new_word)
                            if new_word != word:
                                change += 1
            
                dataset[i]['tgt_txt'] = ' '.join(tgt_txt)         
            
            elif perturbation_type == 'lead3':
                dataset[i]['tgt_txt'] = ' '.join(dataset[i]['src_txt'][:3])
            
            elif perturbation_type == 'irrelevant':
                ir = open('logs/irrelevant_dict', 'r').readlines()
                ir_dict = {}
                for i in range(len(ir)):
                    if i % 2 ==0:
                        ir_dict[ir[i]] = ir[i+1]
                dataset[i]['tgt_txt'] = ir_dict[dataset[i]['tgt_txt']]       

            dataset[i]['tgt'] = [1] + tokenizer.encode(dataset[i]['tgt_txt']) + [2]
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(args.bert_data_path + '.' + corpus_type + '.[0-9]*.bert.pt'))

    if pts:
        if (shuffle):
            random.shuffle(pts)

        for pt in pts:
            if args.perturbation:
                yield _lazy_dataset_loader_with_perturbation(pt, corpus_type, tokenizer, args.perturbation_type)
            else:
                yield _lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one inputters.*Dataset, simple!
        pt = args.bert_data_path + '.' + corpus_type + '.bert.pt'
        if args.perturbation:
            yield _lazy_dataset_loader_with_perturbation(pt, corpus_type, tokenizer)
        else:
            yield _lazy_dataset_loader(pt, corpus_type)        
        


def abs_batch_size_fn(new, count):
    src, tgt = new[0], new[1]
    global max_n_sents, max_n_tokens, max_size
    if count == 1:
        max_size = 0
        max_n_sents=0
        max_n_tokens=0
    max_n_sents = max(max_n_sents, len(tgt))
    max_size = max(max_size, max_n_sents)
    src_elements = count * max_size
    if (count > 6):
        return src_elements + 1e3
    return src_elements


def ext_batch_size_fn(new, count):
    if (len(new) == 4):
        pass
    src, labels = new[0], new[4]
    global max_n_sents, max_n_tokens, max_size
    if count == 1:
        max_size = 0
        max_n_sents = 0
        max_n_tokens = 0
    max_n_sents = max(max_n_sents, len(src))
    max_size = max(max_size, max_n_sents)
    src_elements = count * max_size
    return src_elements


class Dataloader(object):
    def __init__(self, args, datasets,  batch_size,
                 device, shuffle, is_test):
        self.args = args
        self.datasets = datasets
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.is_test = is_test
        self.cur_iter = self._next_dataset_iterator(datasets)
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)


    def _next_dataset_iterator(self, dataset_iter):
        try:
            # Drop the current dataset for decreasing memory
            if hasattr(self, "cur_dataset"):
                self.cur_dataset = None
                gc.collect()
                del self.cur_dataset
                gc.collect()

            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        return DataIterator(args = self.args,
            dataset=self.cur_dataset,  batch_size=self.batch_size,
            device=self.device, shuffle=self.shuffle, is_test=self.is_test)


class DataIterator(object):
    def __init__(self, args, dataset,  batch_size, device=None, is_test=False,
                 shuffle=True):
        self.args = args
        self.batch_size, self.is_test, self.dataset = batch_size, is_test, dataset
        self.iterations = 0
        self.device = device
        self.shuffle = shuffle

        self.sort_key = lambda x: len(x[1])

        self._iterations_this_epoch = 0
        if (self.args.task == 'abs'):
            self.batch_size_fn = abs_batch_size_fn
        else:
            self.batch_size_fn = ext_batch_size_fn

    def data(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        xs = self.dataset
        return xs






    def preprocess(self, ex, is_test):
        src = ex['src']
        tgt = ex['tgt'][:self.args.max_tgt_len][:-1]+[2]
        src_sent_labels = ex['src_sent_labels']
        segs = ex['segs']
        if(not self.args.use_interval):
            segs=[0]*len(segs)
        clss = ex['clss']
        src_txt = ex['src_txt']
        tgt_txt = ex['tgt_txt']

        end_id = [src[-1]]
        src = src[:-1][:self.args.max_pos - 1] + end_id
        segs = segs[:self.args.max_pos]
        max_sent_id = bisect.bisect_left(clss, self.args.max_pos)
        src_sent_labels = src_sent_labels[:max_sent_id]
        clss = clss[:max_sent_id]
        # src_txt = src_txt[:max_sent_id]



        if(is_test):
            return src, tgt, segs, clss, src_sent_labels, src_txt, tgt_txt
        else:
            return src, tgt, segs, clss, src_sent_labels

    def batch_buffer(self, data, batch_size):
        minibatch, size_so_far = [], 0
        for ex in data:
            if(len(ex['src'])==0):
                continue
            ex = self.preprocess(ex, self.is_test)
            if(ex is None):
                continue
            minibatch.append(ex)
            size_so_far = self.batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def batch(self, data, batch_size):
        """Yield elements from data in chunks of batch_size."""
        minibatch, size_so_far = [], 0
        for ex in data:
            minibatch.append(ex)
            size_so_far = self.batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def create_batches(self):
        """ Create batches """
        data = self.data()
        for buffer in self.batch_buffer(data, self.batch_size * 300):

            if (self.args.task == 'abs'):
                p_batch = sorted(buffer, key=lambda x: len(x[2]))
                p_batch = sorted(p_batch, key=lambda x: len(x[1]))
            else:
                p_batch = sorted(buffer, key=lambda x: len(x[2]))

            p_batch = self.batch(p_batch, self.batch_size)


            p_batch = list(p_batch)
            if (self.shuffle):
                random.shuffle(p_batch)
            for b in p_batch:
                if(len(b)==0):
                    continue
                yield b

    def __iter__(self):
        while True:
            self.batches = self.create_batches()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                batch = Batch(minibatch, self.device, self.is_test)

                yield batch
            return


class TextDataloader(object):
    def __init__(self, args, datasets, batch_size,
                 device, shuffle, is_test):
        self.args = args
        self.batch_size = batch_size
        self.device = device

    def data(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        xs = self.dataset
        return xs

    def preprocess(self, ex, is_test):
        src = ex['src']
        tgt = ex['tgt'][:self.args.max_tgt_len][:-1] + [2]
        src_sent_labels = ex['src_sent_labels']
        segs = ex['segs']
        if (not self.args.use_interval):
            segs = [0] * len(segs)
        clss = ex['clss']
        src_txt = ex['src_txt']
        tgt_txt = ex['tgt_txt']

        end_id = [src[-1]]
        src = src[:-1][:self.args.max_pos - 1] + end_id
        segs = segs[:self.args.max_pos]
        max_sent_id = bisect.bisect_left(clss, self.args.max_pos)
        src_sent_labels = src_sent_labels[:max_sent_id]
        clss = clss[:max_sent_id]
        # src_txt = src_txt[:max_sent_id]

        if (is_test):
            return src, tgt, segs, clss, src_sent_labels, src_txt, tgt_txt
        else:
            return src, tgt, segs, clss, src_sent_labels

    def batch_buffer(self, data, batch_size):
        minibatch, size_so_far = [], 0
        for ex in data:
            if (len(ex['src']) == 0):
                continue
            ex = self.preprocess(ex, self.is_test)
            if (ex is None):
                continue
            minibatch.append(ex)
            size_so_far = simple_batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], simple_batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def create_batches(self):
        """ Create batches """
        data = self.data()
        for buffer in self.batch_buffer(data, self.batch_size * 300):
            if (self.args.task == 'abs'):
                p_batch = sorted(buffer, key=lambda x: len(x[2]))
                p_batch = sorted(p_batch, key=lambda x: len(x[1]))
            else:
                p_batch = sorted(buffer, key=lambda x: len(x[2]))
                p_batch = batch(p_batch, self.batch_size)

            p_batch = batch(p_batch, self.batch_size)

            p_batch = list(p_batch)
            if (self.shuffle):
                random.shuffle(p_batch)
            for b in p_batch:
                if (len(b) == 0):
                    continue
                yield b

    def __iter__(self):
        while True:
            self.batches = self.create_batches()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                batch = Batch(minibatch, self.device, self.is_test)

                yield batch
            return
