import json

import torch
from torch.optim import Adam
import argparse
from attention_embedding import AttentionEmbedding
from code_data import DataSet, DataEntry
from tqdm import tqdm
from torch import nn
import numpy as np


def train(model, loss_function, optimizer, dataset, num_epochs, cuda_device=-1):
    model.train()
    softmax = nn.LogSoftmax(dim=-1)
    for epoch in range(num_epochs):
        all_losses = []
        for sequence, mask, label in tqdm(dataset.get_all_batches()):
            model.zero_grad()
            optimizer.zero_grad()
            if cuda_device != -1:
                sequence = sequence.cuda()
                mask = mask.cuda()
                label = label.cuda()
            output, _, _ = model(sequence, mask)
            output = softmax(output)
            batch_loss = loss_function(output, label)
            all_losses.append(batch_loss.cpu().item())
            batch_loss.backward()
            optimizer.step()
        print('After Epoch %d\tTotal Loss : %f' % (epoch + 1, np.average(all_losses)))


def generate_embeddings(model, dataset, output_path=None, cuda_device=-1):
    model.eval()
    vectors = []
    with torch.no_grad():
        for sentence, sequence, mask, label in tqdm(dataset.get_all_test_examples()):
            if cuda_device != -1:
                sequence = sequence.cuda()
                mask = mask.cuda()
            _, embedding, _ = model(sequence, mask)
            embedding = embedding.cpu().data.numpy().tolist()[0]
            data = {
                'code': sentence,
                'label': label[0].item(),
                'embedding': embedding
            }
            vectors.append(data)
    if output_path is not None:
        json.dump(vectors, open(output_path, 'w'))
    return vectors


def main(args):
    inital_emb_path = args.word_to_vec
    dataset = DataSet(initial_embedding_path=inital_emb_path)
    if args.job == 'train':
        train_data = json.load(open(args.train_file))
        for e in train_data:
            entry = DataEntry(dataset, e['code'], e['label'])
            dataset.add_data_entry(entry, train_example=True)
        dataset.init_data_set(batch_size=args.batch_size)
        if inital_emb_path is not None:
            emb_dim = dataset.initial_emddings.vector_size
        else:
            emb_dim = args.hidden_dim
        model = AttentionEmbedding(
            emb_dim=emb_dim, hidden_dim=args.hidden_dim, output_dim=2, external_token_embed=(inital_emb_path is not None))
        loss_function = nn.NLLLoss()
        optimizer = Adam(model.parameters())
        if args.cuda_device != -1:
            model.cuda(device=args.cuda_device)
        train(
            model=model, loss_function=loss_function,
            optimizer=optimizer, dataset=dataset,
            num_epochs=args.num_epochs, cuda_device=args.cuda_device
        )
        torch.save(model, args.model_path)
    elif args.job == 'train_and_generate':
        train_data = json.load(open(args.train_file))
        for e in train_data:
            entry = DataEntry(dataset, e['code'], e['label'])
            dataset.add_data_entry(entry, train_example=True)
        test_data = json.load(open(args.test_file))
        for e in test_data:
            entry = DataEntry(dataset, e['code'], e['label'])
            dataset.add_data_entry(entry, train_example=False)
        dataset.init_data_set(batch_size=args.batch_size)
        if inital_emb_path is not None:
            emb_dim = dataset.initial_emddings.vector_size
        else:
            emb_dim = args.hidden_dim
        model = AttentionEmbedding(
            emb_dim=emb_dim, hidden_dim=args.hidden_dim, output_dim=2,
            external_token_embed=(inital_emb_path is not None))
        loss_function = nn.NLLLoss()
        optimizer = Adam(model.parameters())
        if args.cuda_device != -1:
            model.cuda(device=args.cuda_device)
        train(
            model=model, loss_function=loss_function,
            optimizer=optimizer, dataset=dataset,
            num_epochs=args.num_epochs, cuda_device=args.cuda_device
        )
        model_file = open(args.model_path, 'wb')
        torch.save(model, model_file)
        model_file.close()
        embeddings = generate_embeddings(
            model=model, dataset=dataset, output_path=args.test_output_path, cuda_device=args.cuda_device)

    elif args.job == 'generate':
        test_data = json.load(open(args.test_file))
        for e in test_data:
            entry = DataEntry(dataset, e['code'], e['label'])
            dataset.add_data_entry(entry, train_example=False)
        dataset.init_data_set()
        model_file = open(args.model_path, 'rb')
        model = torch.load(model_file)
        if args.cuda_device != -1:
            model.cuda(device=args.cuda_device)
        embeddings = generate_embeddings(
            model=model, dataset=dataset, output_path=args.test_output_path, cuda_device=args.cuda_device)


def check_argumanets(args):
    assert args.hidden_dim > 0, 'Model dimension has to be positive integer. make sure args.hidden_dim > 0'
    assert args.batch_size > 0, 'Batch size has to be positive integer. make sure args.batch_size > 0'
    if args.job == 'train' or args.job == 'train_and_generate':
        assert args.train_file is not None, \
            'Train file must be provided when args.job == \'train\' or args.job == \'train_and_generate\''
    if args.job == 'generate' or args.job == 'train_and_generate':
        assert args.test_file is not None, \
            'Test file must be provided when args.job == \'generate\' or args.job == \'train_and_generate\''
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--word_to_vec',
                        help='Path of the Word to vector model (optional).', default=None, type=str)
    parser.add_argument('--train_file',
                        help='Path of the train json file (required).', type=str, default=None)
    parser.add_argument('--test_file',
                        help='Path of the train json file (required, if job=\'generate\' or \'train_and_generate\').',
                        type=str)
    parser.add_argument('--test_output_path',
                        help='Path of file where test embedding will be saved (optional).',
                        type=str, default=None)
    parser.add_argument('--hidden_dim', help='Model dimension.', default=256, type=int)
    parser.add_argument('--num_epochs', help='Number of Epochs for training.', type=int, default=50)
    parser.add_argument('--batch_size', help='Batch size for training.', default=16, type=int)
    parser.add_argument('--model_path', help='Path of the trained model (required).', required=True, type=str)
    parser.add_argument('--cuda_device', help='-1 if CPU is used otherwise specific device number.',
                        type=int, default=-1)
    parser.add_argument('--job', choices=['train', 'generate', 'train_and_generate'], type=str, required=True)
    args = parser.parse_args()
    check_argumanets(args)
    main(args)
