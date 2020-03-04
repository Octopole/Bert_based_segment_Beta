from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
from .bert import modeling
from .bert import optimization
from .bert import tokenization
import tensorflow as tf
#from sklearn.metrics import f1_score,precision_score,recall_score
from tensorflow.python.ops import math_ops
from . import tf_metrics
import pickle
import re
import time
import numpy as np
import copy

flags = tf.flags

FLAGS = flags.FLAGS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ROOTDIR = os.path.dirname(os.path.abspath(__file__))

flags.DEFINE_string(
    "label2id", ROOTDIR+"/output",
    "The searching-helper dir."
)

flags.DEFINE_string(
    "data_dir", ROOTDIR+"/predict",
    "The input datadir.",
)

flags.DEFINE_string(
    "bert_config_file", ROOTDIR+"/bert_config.json",
    "The config json file corresponding to the pre-trained BERT model."
)

flags.DEFINE_string(
    "task_name", "NER", "The name of the task to train."
)

flags.DEFINE_string(
    "output_dir", ROOTDIR+"/predict",
    "The output directory where the model checkpoints will be written."
)

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", ROOTDIR+"/output",
    "Initial checkpoint (usually from a pre-trained BERT model)."
)

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text."
)

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization."
)

flags.DEFINE_bool(
    "do_train", False,
    "Whether to run training."
)
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", True,"Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0, "Total number of training epochs to perform.")



flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_string("vocab_file", "vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")
tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")
flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids,):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        #self.label_mask = label_mask


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        """Reads a BIO data."""
        with open(input_file, encoding="utf-8") as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contends = line.strip()
                word = line.strip().split(' ')[0]
                label = line.strip().split(' ')[-1]
                if contends.startswith("-DOCSTART-"):
                    words.append('')
                    continue
                # if len(contends) == 0 and words[-1] == '。':
                if len(contends) == 0:
                    l = ' '.join([label for label in labels if len(label) > 0])
                    w = ' '.join([word for word in words if len(word) > 0])
                    lines.append([l, w])
                    words = []
                    labels = []
                    continue
                words.append(word)
                labels.append(label)
            return lines


class NerProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.txt")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.txt")), "dev"
        )

    def get_test_examples(self,data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.txt")), "test")


    def get_labels(self):
        # prevent potential bug for chinese text mixed with english text
        # return ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]","[SEP]"]
        return ["B-DN","B-Extra","B-FlangeType","B-Material","B-Name","B-PN","B-Specification","B-Standard","B-Thick","B-OrderNumber","I-OrderNumber","I-DN","I-Extra","I-FlangeType","I-Material","I-Name","I-PN","I-Specification","I-Standard","I-Thick","O","X","[CLS]","[SEP]"]

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples

def extlabel(input):
    return (input[0], input[2:])

def write_tokens(tokens,mode):
    if mode=="test":
        path = os.path.join(FLAGS.output_dir, "token_"+mode+".txt")
        wf = open(path, 'a', encoding="utf-8")
        for token in tokens:
            if token!="**NULL**":
                wf.write(token+'\n')
        wf.close()

def convert_single_example(ex_index, example, label_map, max_seq_length, tokenizer,mode):
    textlist = example.text.split(' ')
    labellist = example.label.split(' ')
    tokens = []
    labels = []
    # print(textlist)
    for i, word in enumerate(textlist):
        token = tokenizer.tokenize(word)
        # print(token)
        tokens.extend(token)
        label_1 = labellist[i]
        # print(label_1)
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:
                labels.append("X")
        # print(tokens, labels)
    # tokens = tokenizer.tokenize(example.text)
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]
        labels = labels[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(label_map["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    ntokens.append("[SEP]")
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    #label_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")
        #label_mask.append(0)
    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    #assert len(label_mask) == max_seq_length

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        #tf.logging.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))
    #print("+--------------+")
    #print(input_ids)
    #print("+--------------+")
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        #label_mask = label_mask
    )
    write_tokens(ntokens,mode)
    #return feature
    zl = []
    for i in input_mask:
        zl += [0]
    input_mask = [zl, input_mask]
    zl = []
    for i in segment_ids:
        zl += [0]
    segment_ids = [zl, segment_ids]
    zl = []
    for i in input_ids:
        zl += [0]
    input_ids = [zl, input_ids]

    '''print(input_ids)
    print(input_mask)
    print(segment_ids)
    print(label_ids)
    '''
    return {"input_ids": input_ids, "input_mask": input_mask, "segment_ids": segment_ids, "label_ids": label_ids}


def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings):

    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )

    output_layer = model.get_sequence_output()

    hidden_size = output_layer.shape[-1].value

    output_weight = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02)
    )
    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer()
    )
    with tf.variable_scope("loss"):
        if is_training:
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        output_layer = tf.reshape(output_layer, [-1, hidden_size])
        logits = tf.matmul(output_layer, output_weight, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        logits = tf.reshape(logits, [-1, FLAGS.max_seq_length, 25])
        # mask = tf.cast(input_mask,tf.float32)
        # loss = tf.contrib.seq2seq.sequence_loss(logits,labels,mask)
        # return (loss, logits, predict)
        ##########################################################################
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_sum(per_example_loss)
        probabilities = tf.nn.softmax(logits, axis=-1)
        predict = tf.argmax(probabilities,axis=-1)
        return (loss, per_example_loss, logits,predict)
        ##########################################################################

def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        #label_mask = features["label_mask"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        (total_loss, per_example_loss, logits, predicts) = create_model(
            bert_config, is_training, input_ids, input_mask,segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)
        tvars = tf.trainable_variables()
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()
                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        tf.logging.info("**** Trainable Variables ****")

        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            #tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, logits):
            # def metric_fn(label_ids, logits):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                precision = tf_metrics.precision(label_ids,predictions,25,[1],average="macro")
                recall = tf_metrics.recall(label_ids,predictions,25,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22],average="macro")
                f = tf_metrics.f1(label_ids,predictions,25,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22],average="macro")
                #
                return {
                    "eval_precision":precision,
                    "eval_recall":recall,
                    "eval_f": f,
                    #"eval_loss": loss,
                }
            eval_metrics = (metric_fn, [per_example_loss, label_ids, logits])
            # eval_metrics = (metric_fn, [label_ids, logits])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode = mode,predictions= predicts,scaffold_fn=scaffold_fn
            )
        return output_spec
    return model_fn


def create_example(lines, set_type):
    examples = []
    for (i, line) in enumerate(lines):
        guid = "%s-%s" % (set_type, i)
        text = tokenization.convert_to_unicode(line[1])
        label = tokenization.convert_to_unicode(line[0])
        examples+=[InputExample(guid=guid, text=text, label=label)]
    return examples

def linepurify(line):
    stopword = [" ", ",", "，", "\t", "\n", "(", ")", "（", "）","+"]
    output = ""
    for i in line:
        if i not in stopword:
            output += i
        else:
            output += "_"
    return output
    #return line.strip()

def input_based_convert_examples_to_features(
    inputstr, label_map, max_seq_length, tokenizer, buffer_path, mode=None
):
    lines = []
    words = ""
    labels = ""
    for letter in inputstr:
        words += letter+" "
        labels += "O "
    lines += [[labels.strip(), words.strip()]]
    lines += [["", ""]]

    examples = create_example(lines, "test")
    writer = tf.python_io.TFRecordWriter(buffer_path+"/predict.tf_record")
    feature = convert_single_example(6, examples[0], label_map, max_seq_length, tokenizer, mode)

    return feature

def record_based_input_fn_builder(record, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        #"label_ids": tf.VarLenFeature(tf.int64),
        #"label_mask": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = record
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d
    return input_fn

def serving_input_fn():
    inputs = {
        'label_ids': tf.placeholder(dtype=tf.int64, shape=[None], name='label_ids'),
        'input_ids': tf.placeholder(dtype=tf.int64, shape=[None,128], name='input_ids'),
        'input_mask': tf.placeholder(dtype=tf.int64, shape=[None,128], name='input_mask'),
        'segment_ids': tf.placeholder(dtype=tf.int64, shape=[None,128], name='segment_ids'),
    }
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

def dupicate_killer(inplist):
    outputlist = []
    for index in range(len(inplist)):
        checker=True
        for item in inplist[:index]+inplist[index+1:]:
            if item.rfind(inplist[index])!=-1:
                if item!=inplist[index]:
                    checker=False
        if checker and inplist[index] not in outputlist:
            outputlist += [inplist[index]]

    return outputlist


class BertSeg:
    def __init__(self, searchrootdir):
        #os.environ['AUTOGRAPH_VERBOSITY'] = '10'
        self.SH_dir=searchrootdir
        self.data_dir=self.SH_dir+"/predict"
        self.bert_config_file=self.SH_dir+"/bert_config.json"
        self.task_name="NER"
        self.output_dir=self.SH_dir+"/predict"
        self.init_checkpoint=self.SH_dir+"/output"
        self.do_lower_case=True
        self.max_seq_length=128
        self.do_train=False
        self.use_tpu=False
        self.do_eval=False
        self.do_predict=True
        self.train_batch_size=32
        self.eval_batch_size=8
        self.predict_batch_size=8
        self.learning_rate=5e-5
        self.num_train_epochs=3.0
        self.warmup_proportion=0.1
        self.save_checkpoints_steps=1000
        self.iterations_per_loop=1000
        self.vocab_file=self.SH_dir+"/vocab.txt"
        self.master=None
        self.num_tpu_cores=8

        tf.logging.set_verbosity(tf.logging.INFO)
        self.processors = {
            "ner": NerProcessor
        }

        self.bert_config = modeling.BertConfig.from_json_file(self.bert_config_file)

        if self.max_seq_length > self.bert_config.max_position_embeddings:
            raise ValueError(
                "Cannot use sequence length %d because the BERT model "
                "was only trained up to sequence length %d" %
                (self.max_seq_length, self.bert_config.max_position_embeddings))

        self.task_name = self.task_name.lower()
        if self.task_name not in self.processors:
            raise ValueError("Task not found: %s" % (self.task_name))
        self.processor = self.processors[self.task_name]()

        self.label_list = self.processor.get_labels()
        self.label_map = {}
        for (i, label) in enumerate(self.label_list,1):
            self.label_map[label] = i

        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=self.vocab_file, do_lower_case=self.do_lower_case)
        self.tpu_cluster_resolver = None

        self.is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

        self.run_config = tf.contrib.tpu.RunConfig(
            cluster=self.tpu_cluster_resolver,
            master=self.master,
            model_dir=self.output_dir,
            save_checkpoints_steps=self.save_checkpoints_steps,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=self.iterations_per_loop,
                num_shards=self.num_tpu_cores,
                per_host_input_for_training=self.is_per_host))

        self.train_examples = None
        self.num_train_steps = None
        self.num_warmup_steps = None

        self.model_fn = model_fn_builder(
            bert_config=self.bert_config,
            num_labels=len(self.label_list)+1,
            init_checkpoint=self.init_checkpoint,
            learning_rate=self.learning_rate,
            num_train_steps=self.num_train_steps,
            num_warmup_steps=self.num_warmup_steps,
            use_tpu=self.use_tpu,
            use_one_hot_embeddings=self.use_tpu)

        self.estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=self.use_tpu,
            model_fn=self.model_fn,
            config=self.run_config,
            train_batch_size=self.train_batch_size,
            eval_batch_size=self.eval_batch_size,
            predict_batch_size=self.predict_batch_size)

        with open(self.SH_dir+'/output/label2id.pkl','rb') as rf:
            self.label2id = pickle.load(rf)
            self.id2label = {value:key for key,value in self.label2id.items()}

        self.predictor = tf.contrib.predictor.from_estimator(self.estimator, serving_input_fn)

    def line_predict(self, line, outputdic):
        oriline = copy.deepcopy(line)

        line = linepurify(line)
        print("purified line: "+line)
        print("")
        record = input_based_convert_examples_to_features(
            line, self.label_map, self.max_seq_length, self.tokenizer, self.output_dir, mode="test")
        self.p = self.predictor(record)
        return_labels = []
        for key in self.p["output"].tolist()[1]:
            if key != 0:
                v = self.id2label[key]
                #print(v)
                return_labels += [v]

        return_labels = return_labels[1:]

        typebuf = ""
        contbuf = ""
        for ord in range(len(line)):
            tag = extlabel(return_labels[ord])
            if tag[0] == "B":
                if contbuf != "" and typebuf != "":
                    outputdic[typebuf]+=[contbuf]
                typebuf = tag[1]
                contbuf = oriline[ord][0]
            elif tag[0] == "I":
                if typebuf != "":
                    typebuf = tag[1]
                    contbuf += oriline[ord][0]
                else:
                    typebuf = tag[1]
                    contbuf = oriline[ord][0]
            else:
                if contbuf != "" and typebuf != "":
                    outputdic[typebuf]+=[contbuf]
                typebuf = ""
                contbuf = ""
        if contbuf != "" and typebuf != "":
            outputdic[typebuf]+=[contbuf]
        return outputdic

    def predict(self, inputstr):
        time1 = time.time()
        print("+------------------------------+")
        print("original line: " + inputstr)
        print("")
        outputdic = {"OrderNumber":[],"Name":[],"Standard":[],"FlangeType":[],"Material":[],"DN":[],"PN":[],"Specification":[],"Thick":[],"Extra":[]}
        stopper = [" ",",", "，"]
        #outputdic = self.line_predict(inputstr, outputdic)

        while(len(inputstr)>=128):
            seperater = 0
            for i in stopper:
                k = inputstr[:128].rfind(i)
                if k > seperater:
                    seperater = k

            preline = inputstr[:seperater]
            outputdic = self.line_predict(preline, outputdic)
            postline = inputstr[seperater:]
            inputstr = postline
        outputdic = self.line_predict(inputstr, outputdic)
        for key in outputdic:
            buflist = dupicate_killer(outputdic[key])
            outputdic[key] = buflist
        time2 = time.time()
        print(outputdic)
        print("time cost: "+ str(time2-time1))
        print("+------------------------------+")

        return outputdic
