import numpy as np
import torch
from gensim.models import Word2Vec


class DataEntry:
    def __init__(self, dataset, sentence, label, meta_data = None, parser = None):
        self.dataset = dataset
        assert isinstance(self.dataset, DataSet)
        self.sentence = sentence
        self.label = label
        if parser is not None:
            self.words = parser.parse(self.sentence)
        else:
            self.words = self.sentence.split()
        self.meta_data = meta_data
        pass

    def init_word_index(self):
        assert self.dataset.vocab is not None, 'Initialize Dataset Vocabulary First'
        self.word_indices = [self.dataset.vocab.start_token]
        for word in self.words:
            self.word_indices.append(self.dataset.vocab.get_token_id(word))
        self.word_indices.append(self.dataset.vocab.end_token)

    def __repr__(self):
        return str(self.word_indices) + '\t' + str(self.label)


class DataSet:
    class Vocabulary:
        def __init__(self):
            self.start_token = 0
            self.end_token = 1
            self.pad_token = 2
            self.unk = 3
            self.index_to_token = {0: "<START>", 1:"<END>", 2:"<PAD>", 3: "<UNK>"}
            self.token_to_index = {"<START>": 0, "<END>": 1, "<PAD>": 2, "<UNK>": 3}
            self.count = 4
            pass

        def get_token_id(self, token):
            if token in self.token_to_index.keys():
                return self.token_to_index[token]
            else:
                return self.unk

        def get_token(self, id):
            assert id < self.count, 'Invalid token ID'
            return self.index_to_token[id]

        def add_token(self, token):
            index = self.get_token_id(token)
            if index != self.unk:
                return index
            else:
                index = self.count
                self.count += 1
                self.index_to_token[index] = token
                self.token_to_index[token] = index
                return index

    def __init__(self, initial_embedding_path=None):
        self.train_entries = []
        self.test_entries = []
        self.vocab = None
        self.initial_embedding_present = (initial_embedding_path is not None)
        if self.initial_embedding_present:
            self.initial_emddings = Word2Vec.load(initial_embedding_path)

    def add_data_entry(self, entry, train_example=True):
        assert isinstance(entry, DataEntry)
        if self.initial_embedding_present:
            entry.wvmodel = self.initial_emddings
        if train_example:
            self.train_entries.append(entry)
        else:
            self.test_entries.append(entry)

    def init_data_set(self, batch_size=32):
        self.build_vocabulary()
        for entry in self.train_entries:
            entry.init_word_index()
        for entry in self.test_entries:
            entry.init_word_index()
        self.batch_size = batch_size
        self.initialize_batch()

    def build_vocabulary(self):
        self.vocab = DataSet.Vocabulary()
        words = {}
        total_words = 0
        for entry in self.train_entries:
            for word in entry.words:
                if word in words.keys():
                    words[word] += 1
                else:
                    words[word] = 1
                total_words += 1
        for entry in self.test_entries:
            for word in entry.words:
                if word in words.keys():
                    words[word] += 1
                else:
                    words[word] = 1
                total_words += 1
        word_freq = [[key, words[key]] for key in words.keys()]
        word_freq = sorted(word_freq, key=lambda x:x[1], reverse=True)
        accepted_words = word_freq
        for word, count in accepted_words:
            self.vocab.add_token(word)
        print('Total Number of Words', total_words)
        print('Unique Words : ', len(words.keys()))
        print('Vocab Size : ', len(accepted_words))

    def get_data_entries_by_id(self, dataset, ids):
        max_seq_len = max([len(dataset[id].word_indices) for id in ids])
        token_indices = []
        masks = []
        labels = []
        token_vectors = []
        for index in ids:
            indices = [self.vocab.pad_token] * max_seq_len
            if self.initial_embedding_present:
                vectors = [np.zeros(shape=self.initial_emddings.vector_size)] * max_seq_len
            mask = [1] * max_seq_len
            for i, w_index in enumerate(dataset[index].word_indices):
                indices[i] = w_index
                mask[i] = 0
                if self.initial_embedding_present:
                    token = self.vocab.get_token(w_index)
                    if token in self.initial_emddings.wv:
                        vectors[i] = self.initial_emddings.wv[token]
                    elif '<UNK>' in self.initial_emddings.wv:
                        vectors[i] = self.initial_emddings.wv['<UNK>']
            token_indices.append(indices)
            masks.append(mask)
            if self.initial_embedding_present:
                token_vectors.append(vectors)
            labels.append(dataset[index].label)
        if not self.initial_embedding_present:
            return torch.LongTensor(np.asarray(token_indices)), \
                   torch.BoolTensor(np.asarray(masks)), \
                   torch.LongTensor(np.asarray(labels))
        else:
            return torch.FloatTensor(np.asarray(token_vectors)), \
                   torch.BoolTensor(np.asarray(masks)), \
                   torch.LongTensor(np.asarray(labels))

    def get_train_dataset_by_ids(self, ids):
        return self.get_data_entries_by_id(self.train_entries, ids)

    def get_test_dataset_by_ids(self, ids):
        return self.get_data_entries_by_id(self.test_entries, ids)

    def initialize_batch(self):
        total = len(self.train_entries)
        indices = np.arange(0,total-1, 1)
        np.random.shuffle(indices)
        self.batch_indices = []
        start = 0
        end = len(indices)
        curr = start
        while curr < end:
            c_end = curr + self.batch_size
            if c_end > end:
                c_end = end
            self.batch_indices.append(indices[curr:c_end])
            curr = c_end

    def get_all_test_examples(self):
        dataset = [None] * len(self.test_entries)
        for i in range(len(self.test_entries)):
            dataset[i] = [self.get_sentence(self.test_entries, i)]
            dataset[i].extend(list(self.get_test_dataset_by_ids([i])))
        return dataset

    def get_all_test_batches(self, batch_size=32):
        dataset = []
        indices = [i for i in range(len(self.test_entries))]
        batch_indices = []
        start = 0
        end = len(indices)
        curr = start
        while curr < end:
            c_end = curr + batch_size
            if c_end > end:
                c_end = end
            batch_indices.append(indices[curr:c_end])
            curr = c_end
        for indices in batch_indices:
            dataset.append(self.get_test_dataset_by_ids(indices))
        return dataset

    def get_next_batch_train_data(self):
        if len(self.batch_indices) == 0:
            self.initialize_batch()
        indices = self.batch_indices[0]
        self.batch_indices = self.batch_indices[1:]
        return self.get_train_dataset_by_ids(indices)

    def get_all_batches(self):
        dataset = []
        np.random.shuffle(self.train_entries)
        self.initialize_batch()
        for indices in self.batch_indices:
            dataset.append(self.get_train_dataset_by_ids(indices))
        return dataset

    def get_selective_batches(self, selection=20):
        dataset = []
        self.initialize_batch()
        for idx, indices in enumerate(self.batch_indices):
            dataset.append(self.get_train_dataset_by_ids(indices))
            if idx == selection:
                break
        return dataset

    def get_test_data(self):
        return self.get_data_entries_by_id(self.test_entries, list(range(len(self.test_entries))))

    def get_complete_train_data(self):
        return self.get_data_entries_by_id(self.train_entries, list(range(len(self.train_entries))))

    def get_sentence(self, entries, i):
        return ' '.join(entries[i].words)
        pass