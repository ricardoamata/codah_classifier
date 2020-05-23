import random
import numpy as np

np.warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn import Softmax
import torch.optim as optim
from transformers import BertModel, BertConfig, BertTokenizer, load_tf_weights_in_bert
from torch.utils.data import TensorDataset, Dataset, DataLoader, RandomSampler
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam

from codah_pytorch import CodahProcessor, convert_examples_to_features, train_and_validate

def main():

    processor = CodahProcessor("data")
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

    print(">"*12 + " Training with fix dataset " + "<"*12)

    avg_acc = 0

    training_examples = processor.get_train_examples()
    eval_examples = processor.get_dev_examples()

    train_examples = processor.get_train_examples()
    eval_examples = processor.get_dev_examples()
    num_train_examples = len(train_examples)
    train_data = convert_examples_to_features(train_examples, processor.get_labels(), 74, tokenizer)
    train_sampler = RandomSampler(train_data)
    train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=8)

    eval_data = convert_examples_to_features(eval_examples, processor.get_labels(), 74, tokenizer)
    eval_sampler = RandomSampler(eval_data)
    eval_loader = DataLoader(eval_data, sampler=eval_sampler, batch_size=1)

    n_trainings = 0

    for i in range(20):
        print("-"*40)

        model = BertForSequenceClassification.from_pretrained('bert-large-uncased', num_labels=1)
        model.to('cuda')

        results = train_and_validate(model,
                                     train_loader,
                                     eval_loader,
                                     tokenizer,
                                     processor,
                                     6,
                                     1e-5,
                                     16,
                                     num_train_examples,
                                     0.1,
                                     print_every=10,
                                     categories=processor.get_all_categories(),
                                     exclude=False,
                                     use_bert_adam=True,
                                     log_training_info=False)

        if results['eval_accuracy'] >= 0.45:
            avg_acc += results['eval_accuracy']
            n_trainings += 1

    print(f"\n\nAverage accuracy: {avg_acc/n_trainings}")
    print(f"Number of successful trainings: {n_trainings}")
    
    print("\n" + ">"*12 + " Training with random dataset " + "<"*12)
    
   
    avg_acc = 0
    for i in range(20):
        print("-"*40)
        training_examples, eval_examples = processor.get_train_dev_examples()

        train_examples = processor.get_train_examples()
        eval_examples = processor.get_dev_examples()
        num_train_examples = len(train_examples)
        train_data = convert_examples_to_features(train_examples, processor.get_labels(), 74, tokenizer)
        train_sampler = RandomSampler(train_data)
        train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=8)

        eval_data = convert_examples_to_features(eval_examples, processor.get_labels(), 74, tokenizer)
        eval_sampler = RandomSampler(eval_data)
        eval_loader = DataLoader(eval_data, sampler=eval_sampler, batch_size=1)

        model = BertForSequenceClassification.from_pretrained('bert-large-uncased', num_labels=1)
        model.to('cuda')

        results = train_and_validate(model,
                                     train_loader,
                                     eval_loader,
                                     tokenizer,
                                     processor,
                                     6,
                                     1e-5,
                                     16,
                                     num_train_examples,
                                     0.1,
                                     print_every=10,
                                     categories=processor.get_all_categories(),
                                     exclude=False,
                                     use_bert_adam=True,
                                     log_training_info=False)

        if results['eval_accuracy'] >= 0.45:
            avg_acc += results['eval_accuracy']
            n_trainings += 1

    print(f"\n\nAverage accuracy: {avg_acc/n_trainings}")
    print(f"Number of successful trainings: {n_trainings}")

if __name__ == "__main__":
    main()
