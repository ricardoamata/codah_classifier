import _init_paths

import csv
import random
import argparse
import numpy as np

np.warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn import Softmax
import torch.optim as optim
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertAdam  # load_tf_weights_in_bert
from torch.utils.data import TensorDataset, Dataset, DataLoader, RandomSampler

from layout_config import get_config
from datasets.layout_coco import layout_coco
from layout_utils import *
from modules.layout_model import DrawModel

class CodahProcessor():
    """Processor for the CODAH data set."""

    def __init__(self, path="", categories=None, exclude=False): 
        self.id_to_category = list(self.get_all_categories())
        self.categories = {key: id for id, key in enumerate(self.id_to_category)}
        if categories:
            if exclude:
                self.categories = {key: self.categories[key] for key in self.categories if key not in categories}
                if len(self.categories) == 0:
                    raise Exception("there must be at least one category")
            else:
                self.categories = {key: self.categories[key] for key in categories}

        self.path = path

    def get_train_examples(self):
        return self._create_examples(self._read_tsv(self.path+"/train.tsv", quotechar='"'), "train")

    def get_dev_examples(self):
        return self._create_examples(self._read_tsv(self.path+"/test.tsv", quotechar='"'), "dev")

    def get_labels(self):
        return ["0", "1", "2", "3"]

    @staticmethod
    def get_all_categories():
        return {'o', 'n', 'q', 'p', 'i', 'r'}

    def get_train_dev_examples(self, train_size):
        return self._create_train_dev_dataset(self._read_tsv(self.path+"/full_data.tsv", quotechar='"'), train_size)
    
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            category = line[0]
            question = line[1]
            answers = [line[2], line[3], line[4], line[5]]
            label = str(int(line[6]))
            examples.append(
                [(guid, question, answers[i], label, self.categories[category]) for i in range(4)]
            )
        return examples

    def _create_train_dev_dataset(self, lines, train_size):
        ex_categories = {key: [] for key in self.categories}
        random.shuffle(lines)
        for line in lines:
            ex_categories[line[0]].append(line)
        
        train_data = []
        dev_data = []
        for key in self.categories:
            cat_size = len(ex_categories[key])
            train_data.extend(ex_categories[key][:int(train_size*cat_size)])
            dev_data.extend(ex_categories[key][int(train_size*cat_size):])
        
        train_examples = self._create_examples(train_data, "train")
        dev_examples = self._create_examples(dev_data, "dev")

        return train_examples, dev_examples

    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if line[0] in self.categories:
                    lines.append(line)
            return lines[:-1]


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, db, num_b=4):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    input_ids = []
    attention_masks = []
    seg_ids = []
    words_ids = [] # for abstract scene representation
    words_lens = []
    labels = []
    categories = []
    for (ex_index, example) in enumerate(examples):
        input_id = []
        attention_mask = []
        seg_id = []
        word_ids = []
        word_lens = []
        for i in range(num_b):
            
            tokens_a = tokenizer.tokenize(example[i][1])
            tokens_b = tokenizer.tokenize(example[i][2])
            
            tokens = ['[CLS]'] + tokens_a + ['[SEP]']
            seg_mask = [0] * len(tokens)
            tokens += tokens_b + ['[SEP]']
            seg_mask += [1] * len(tokens_b + ['[SEP]'])
            
            in_id = tokenizer.convert_tokens_to_ids(tokens)
            in_mask = [1] * len(in_id)
            padding = [0] * (max_seq_length - len(in_id))
            in_id += padding
            seg_mask += padding
            in_mask += padding

            full_sent = example[i][1] + ' ' + example[i][2]
            
            w_ids, w_lens = db.encode_sentence(full_sent)            

            
            # it's posible to use this tokenizer method
            # but it's broken in some versions of the 
            # transformers package
            # encoded_dict = tokenizer.encode_plus(
            #     example[i][1],
            #     text_pair=example[i][2],
            #     add_special_tokens=True,
            #     max_length=max_seq_length,
            #     pad_to_max_length=True,
            #     return_attention_mask=True,
            # )

            input_id.append(in_id)
            attention_mask.append(in_mask)
            seg_id.append(seg_mask)
            word_ids.append(w_ids)
            word_lens.append(w_lens)


        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        seg_ids.append(seg_id)
        words_ids.append(word_ids)
        words_lens.append(word_lens)
        labels.append([label_map[example[0][3]]]*4)
        categories.append([example[0][4]]*4)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_masks = torch.tensor(attention_masks, dtype=torch.long)
    seg_ids = torch.tensor(seg_ids, dtype=torch.long)
    words_ids = torch.tensor(words_ids, dtype=torch.long)
    words_lens = torch.tensor(words_lens, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    categories = torch.tensor(categories, dtype=torch.long)

    return TensorDataset(input_ids, attention_masks, seg_ids, words_ids, words_lens, labels, categories)


class CodahClassifier(nn.Module):

    def __init__(self, model_type, db,  freeze_bert=False):
        super(CodahClassifier, self).__init__()
        # instantiating BERT model object
        size = model_type.split('-')[1]

        if size == 'large':
            output_dim = 1024
        else:
            output_dim = 768
 
        self.bert_layer = BertForSequenceClassification.from_pretrained(model_type, num_labels=1)
        
        # freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False
        
        self.asn = DrawModel(db)

        for p in self.asn.parameters():
            p.requires_grad = False
        

        self.obj_cls = nn.Linear(db.cfg.output_cls_size, 1)
        
        self.att_layer = nn.Linear(db.cfg.num_scales*2, 1)
        self.att_cls = nn.Linear(db.cfg.grid_size[0] ** 2, 1)

        self.pos_cls = nn.Linear(db.cfg.grid_size[0] ** 2, 1)

        self.relu = nn.ReLU()

    def forward(self, input_ids, seg_ids, input_mask, word_ids, word_lens):

        bert_logits = self.bert_layer(input_ids, seg_ids, input_mask)

        inf_outs, _ = self.asn.inference(word_ids, word_lens, -1, 2.0, 0, None)
        
        obj_logits, coord_logits, attri_logits, _, _, _ = inf_outs

        obj_logits = torch.sum(obj_logits, dim=1)
        obj_logits = self.obj_cls(obj_logits)

        coord_logits = torch.sum(coord_logits, dim=1)
        coord_logits = self.pos_cls(coord_logits)

        attri_logits = torch.sum(attri_logits, dim=1)
        attri_logits = self.att_layer(torch.transpose(attri_logits, 1, 2))
        attri_logits = self.relu(attri_logits[:, :, 0])
        attri_logits = self.att_cls(attri_logits)

        return torch.sum(torch.cat([bert_logits, obj_logits, coord_logits, attri_logits], dim=1), dim=1)

def accuracy(out, labels):
    _, indices = torch.max(out, dim=1)
    return torch.sum((indices == labels).float()) / out.shape[0]
    
def np_accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def train_and_validate(model, 
                       train_loader,
                       eval_loader,
                       tokenizer,
                       processor,
                       max_eps, 
                       lr, 
                       batch_size,
                       num_train_examples,
                       warmup,
                       print_every=10,
                       use_bert_adam=False,
                       log_training_info=True):
    torch.cuda.empty_cache()
    tr_loss = 0
    nb_tr_steps = 1

    criterion = nn.CrossEntropyLoss()
    
    if use_bert_adam:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
        t_total = int( (float(num_train_examples)/batch_size) * max_eps )
        opti = BertAdam(optimizer_grouped_parameters,
                        lr=lr,
                        warmup=warmup,
                        t_total=t_total)
    else:    
        opti = optim.Adam(model.parameters(), lr=lr)

    categories = set(processor.categories.keys())

    # training
    if log_training_info: 
        print("***** Running training *****")
        print(f"  Epochs = {max_eps}\n")
        print(f"  Num examples = {num_train_examples}")
        print(f"  Learning rate = {lr}")
        print(f"  Batch size = {batch_size}")
        print(f"  Categories = {categories if categories != CodahProcessor.get_all_categories() else 'all'}\n")

    model.train()
    for ep in range(max_eps):
        tr_loss = 0
        for step, batch in enumerate(train_loader):
            # clear gradients
            model.zero_grad()
            
            # reshape and reduce the second dimension
            # pull label from training data, set aside for softmax
            n_batches = batch[0].shape[0]
            batch = [ids.view(ids.shape[0] * 4, -1) for ids in batch]

            # feedforward and loss calculation
            # batch = tuple(t.cuda() for t in batch)
            input_ids, input_mask, segment_ids, word_ids, word_lens, label_ids, _ = batch
            logits = model.forward(input_ids.cuda(),
                                   segment_ids.cuda(), 
                                   input_mask.cuda(),
                                   word_ids.cuda(),
                                   word_lens.cuda())  #, label_ids) label removed to skip softmax in model
            logits = logits.view(-1, 4)  # reshape to (:, 4)
            loss = criterion(logits, label_ids.view(n_batches, 4)[:, 0].cuda())
            loss.backward()
            tr_loss += loss.item()
            nb_tr_steps += 1

            # optimization step
            opti.step()

            if (step + 1) % print_every == 0 and log_training_info:
                acc = accuracy(logits, label_ids.view(n_batches, 4)[:, 0].cuda())
                print("Iteration {} of epoch {} complete. Loss : {} Accuracy : {}".format(step+1, ep+1, loss.item(), acc))

    # evaluation

    # eval_examples = processor.get_dev_examples()

    
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    eval_category_acc = {key: 0 for key in categories}
    nb_eval_category_steps = {key: 0 for key in categories}
    model.eval()
    for input_ids, input_mask, segment_ids, word_ids, word_lens, label_ids, category_ids in eval_loader:
        input_ids = input_ids.view(input_ids.shape[0] * 4, -1).cuda()
        input_mask = input_mask.view(input_mask.shape[0] * 4, -1).cuda()
        segment_ids = segment_ids.view(segment_ids.shape[0] * 4, -1).cuda()
        word_ids = word_ids.view(word_ids.shape[0] * 4, -1).cuda()
        word_lens = word_lens.view(word_lens.shape[0] * 4, -1).cuda()
        label_ids = label_ids[:, 0].cuda()
        category_ids = category_ids[:, 0]
      
        with torch.no_grad():  
            logits = model.forward(input_ids, segment_ids, input_mask, word_ids, word_lens).view(-1, 4)
 
        tmp_eval_loss = criterion(logits, label_ids)
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        tmp_eval_accuracy = np_accuracy(logits, label_ids)
        
        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy
        eval_category_acc[processor.id_to_category[category_ids[0]]] += tmp_eval_accuracy
        nb_eval_category_steps[processor.id_to_category[category_ids[0]]] += 1

        nb_eval_examples += label_ids.shape[0]
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples

    result = {  'eval_loss': eval_loss,
                'eval_accuracy': eval_accuracy,
                'tr_loss': tr_loss / nb_tr_steps}

    print("\n***** Eval results *****")
    for key in sorted(result.keys()):
        print(f"{key} = {str(result[key])}")

    print("\nresults by question category")
    for key in categories:
        eval_category_acc[key] /= nb_eval_category_steps[key]
        print(f"{key} = {eval_category_acc[key]}")
    
    return result


def main():
    parser = argparse.ArgumentParser()    

    parser.add_argument("--codah_dir",
                        type=str,
                        required=True,
                        help="The input data dir. Should contain train.tsv and dev.tsv files for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--train_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--train_size",
                        default=0.8,
                        type=float,
                        help="Percentage of the data use for training.")
    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=6,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--categories",
                        type=str,
                        default="all",
                        help='String with the categories to be included or excluded separated by "-" ej: "o-i-q-n".')
    parser.add_argument("--exclude_categories",
                        default=False,
                        action="store_true",
                        help="the categories listed in `--categories` will be excluded,  must not be use if all categories are listed.")
    parser.add_argument("--local_model",
                       default=False,
                       action="store_true",
                       help="This is to load the bert model from a local ckpt tensorflow index instead of downloading it.")
    parser.add_argument("--use_pooled",
                       default=False,
                       action="store_true",
                       help="Use the pooler output instead of the normal cls.")
    parser.add_argument("--bert_dir",
                       type=str,
                       help="The directori to load ckpt index of bert model.")
    parser.add_argument("--use_bert_adam",
                       default=False,
                       action="store_true",
                       help="Use build in BertAdam class instead of Adam.")


 
    args = parser.parse_args()

    print(args)

    if args.categories == "all":
        categories = CodahProcessor.get_all_categories()
    else:
        categories = set(args.categories.split('-'))

   
    cfg, _ = get_config()
    cfg.cuda =  True
    transformer = volume_normalize('background')
    db = layout_coco(cfg, split='train', transform=transformer)
 
    processor = CodahProcessor(path=args.codah_dir, categories=categories, exclude=args.exclude_categories)
    print(" Initializing tokenizer ")
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    print(" Creating train and dev datasets ")
    train_examples, eval_examples = processor.get_train_dev_examples(args.train_size)
    num_train_examples = len(train_examples)
    train_data = convert_examples_to_features(train_examples, processor.get_labels(), args.max_seq_length, tokenizer, db)
    train_sampler = RandomSampler(train_data)
    train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    eval_data = convert_examples_to_features(eval_examples, processor.get_labels(), args.max_seq_length, tokenizer, db)
    eval_sampler = RandomSampler(eval_data)
    eval_loader = DataLoader(eval_data, sampler=eval_sampler, batch_size=1)

    print(" Initializing bert model ")

    # model = CodahClasifier(model_type=args.bert_model,
    #                          from_tf=args.local_model,
    #                          tf_dir=args.model_dir,
    #                          use_pooled_output=args.use_pooled, 
    #                          freeze_bert=False).cuda()

    # model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=1)
    model = CodahClassifier(args.bert_model, db)
    model.cuda()
    train_and_validate(model, 
                        train_loader, 
                        eval_loader, 
                        tokenizer, 
                        processor, 
                        args.num_train_epochs, 
                        args.learning_rate, 
                        args.train_batch_size,
                        num_train_examples,
                        args.warmup,
                        print_every=10,
                        use_bert_adam=args.use_bert_adam)

if __name__ == "__main__":
    main()
