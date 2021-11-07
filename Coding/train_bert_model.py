from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
import os
import re
from configuration import ConfigFile


class TrainBertModel:

    def __init__(self):
        if not os.path.isdir(ConfigFile.OUTPUT_DIR):
            os.makedirs(ConfigFile.OUTPUT_DIR)

    def load_directory_data(self, directory):
        """
                        Method Name: load_directory_data
                        Description: BLoad all files from a directory in a DataFrame
                                                    """
        data = {"sentence": [], "sentiment": []}
        for file_path in os.listdir(directory):
            with tf.gfile.GFile(os.path.join(directory, file_path), "r") as f:
                data["sentence"].append(f.read())
                data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
        return pd.DataFrame.from_dict(data)

    def load_dataset(self, directory):
        """
                        Method Name: load_dataset
                        Description: Merge positive and negative examples, add a polarity column and shuffle.
                                                    """
        pos_df = self.load_directory_data(os.path.join(directory, "pos"))
        neg_df = self.load_directory_data(os.path.join(directory, "neg"))
        pos_df["polarity"] = 1
        neg_df["polarity"] = 0
        return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)

    def download_and_load_datasets(self, force_download=True):
        """
        Method Name: download_and_load_datasets
        Description: Download and process the dataset files
        """

        if force_download:
            dataset = tf.keras.utils.get_file(
                fname="aclImdb.tar.gz",
                origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
                extract=True)
        else:
            dataset = tf.keras.utils.get_file(
                fname="aclImdb.tar.gz",
                origin="/home/paul/PycharmProjects/moviebert/aclImdb",
                extract=False)

        train_df = self.load_dataset(os.path.join(os.path.dirname(dataset),
                                                  "aclImdb", "train"))
        test_df = self.load_dataset(os.path.join(os.path.dirname(dataset),
                                                 "aclImdb", "test"))

        return train_df, test_df

    def processDataInBertFormat(self, train, test):

        """
        Method Name: processDataInBertFormat
        Description: Download and process the dataset files
        """
        train = train.sample(5000)
        test = test.sample(5000)
        # Use the InputExample class from BERT's run_classifier code to create examples from the data
        train_InputExamples = train.apply(lambda x: bert.run_classifier.InputExample(guid=None,
                                                                                     # Globally unique ID for bookkeeping, unused in this example
                                                                                     text_a=x[ConfigFile.DATA_COLUMN],
                                                                                     text_b=None,
                                                                                     label=x[ConfigFile.LABEL_COLUMN]),
                                          axis=1)

        test_InputExamples = test.apply(lambda x: bert.run_classifier.InputExample(guid=None,
                                                                                   text_a=x[ConfigFile.DATA_COLUMN],
                                                                                   text_b=None,
                                                                                   label=x[ConfigFile.LABEL_COLUMN]),
                                        axis=1)

        return train_InputExamples, test_InputExamples

    def create_tokenizer_from_hub_module(self):
        """
                Method Name: create_tokenizer_from_hub_module
                Description: Creating the Tokenizer for each words using tensorflow Hub
                """
        with tf.Graph().as_default():
            bert_module = hub.Module(ConfigFile.BERT_MODEL_HUB)
            tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
            with tf.Session() as sess:
                vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                      tokenization_info["do_lower_case"]])

        return bert.tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=do_lower_case)

    def tokenizeSentences(self, tokenizer, sentence):
        tokenizedSent = tokenizer.tokenize(sentence)

        return tokenizedSent

    def convertDataIninpFeatures(self, tokenizer, train_InputExamples, test_InputExamples):

        """
                        Method Name: convertDataIninpFeatures
                        Description: Convert our train and test features to InputFeatures that BERT understands.
                        """
        train_features = bert.run_classifier.convert_examples_to_features(train_InputExamples, ConfigFile.label_list,
                                                                          ConfigFile.MAX_SEQ_LENGTH, tokenizer)
        test_features = bert.run_classifier.convert_examples_to_features(test_InputExamples, ConfigFile.label_list,
                                                                         ConfigFile.MAX_SEQ_LENGTH,
                                                                         tokenizer)

        return train_features, test_features

    def create_model(self, is_predicting, input_ids, input_mask, segment_ids, labels,
                     num_labels):

        """
        Method Name: create_model
        Description: Creates a classification model
                                    """
        bert_module = hub.Module(
            ConfigFile.BERT_MODEL_HUB,
            trainable=True)
        bert_inputs = dict(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids)
        bert_outputs = bert_module(
            inputs=bert_inputs,
            signature="tokens",
            as_dict=True)

        output_layer = bert_outputs["pooled_output"]

        hidden_size = output_layer.shape[-1].value

        # Create our own layer to tune for politeness data.
        output_weights = tf.get_variable(
            "output_weights", [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer())

        with tf.variable_scope("loss"):
            # Dropout helps prevent overfitting
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            # Convert labels into one-hot encoding
            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

            predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
            # If we're predicting, we want predicted labels and the probabiltiies.
            if is_predicting:
                return predicted_labels, log_probs

            # If we're train/eval, compute loss between predicted and actual label
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)
            return loss, predicted_labels, log_probs

    # model_fn_builder actually creates our model function
    # using the passed parameters for num_labels, learning_rate, etc.
    def model_fn_builder(self, num_labels, learning_rate, num_train_steps,
                         num_warmup_steps):

        """
                Method Name: model_fn_builder
                Description: Creates our model function
                Return: model_fn` closure for TPUEstimator
        """

        def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
            """The `model_fn` for TPUEstimator."""

            input_ids = features["input_ids"]
            input_mask = features["input_mask"]
            segment_ids = features["segment_ids"]
            label_ids = features["label_ids"]

            is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)

            # TRAIN and EVAL
            if not is_predicting:

                (loss, predicted_labels, log_probs) = self.create_model(
                    is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

                train_op = bert.optimization.create_optimizer(
                    loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

                # Calculate evaluation metrics.
                def metric_fn(label_ids, predicted_labels):
                    accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
                    f1_score = tf.contrib.metrics.f1_score(
                        label_ids,
                        predicted_labels)
                    auc = tf.metrics.auc(
                        label_ids,
                        predicted_labels)
                    recall = tf.metrics.recall(
                        label_ids,
                        predicted_labels)
                    precision = tf.metrics.precision(
                        label_ids,
                        predicted_labels)
                    true_pos = tf.metrics.true_positives(
                        label_ids,
                        predicted_labels)
                    true_neg = tf.metrics.true_negatives(
                        label_ids,
                        predicted_labels)
                    false_pos = tf.metrics.false_positives(
                        label_ids,
                        predicted_labels)
                    false_neg = tf.metrics.false_negatives(
                        label_ids,
                        predicted_labels)
                    return {
                        "eval_accuracy": accuracy,
                        "f1_score": f1_score,
                        "auc": auc,
                        "precision": precision,
                        "recall": recall,
                        "true_positives": true_pos,
                        "true_negatives": true_neg,
                        "false_positives": false_pos,
                        "false_negatives": false_neg
                    }

                eval_metrics = metric_fn(label_ids, predicted_labels)

                if mode == tf.estimator.ModeKeys.TRAIN:
                    return tf.estimator.EstimatorSpec(mode=mode,
                                                      loss=loss,
                                                      train_op=train_op)
                else:
                    return tf.estimator.EstimatorSpec(mode=mode,
                                                      loss=loss,
                                                      eval_metric_ops=eval_metrics)
            else:
                (predicted_labels, log_probs) = self.create_model(
                    is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

                predictions = {
                    'probabilities': log_probs,
                    'labels': predicted_labels
                }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        # Return the actual model function in the closure
        return model_fn

    def prepareTraingParams(self, train_features):

        """
                        Method Name: prepareTraingParams
                        Description: Preparing the Training parameters example train and warmup steps from batch size
                        Return: Number of checkpoint steps to save
                """

        num_train_steps = int(len(train_features) / ConfigFile.BATCH_SIZE * ConfigFile.NUM_TRAIN_EPOCHS)
        num_warmup_steps = int(num_train_steps * ConfigFile.WARMUP_PROPORTION)

        #
        run_config = tf.estimator.RunConfig(
            model_dir=ConfigFile.OUTPUT_DIR,
            save_summary_steps=ConfigFile.SAVE_SUMMARY_STEPS,
            save_checkpoints_steps=ConfigFile.SAVE_CHECKPOINTS_STEPS)

        model_fn = self.model_fn_builder(
            num_labels=len(ConfigFile.label_list),
            learning_rate=ConfigFile.LEARNING_RATE,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps)

        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config,
            params={"batch_size": ConfigFile.BATCH_SIZE})

        # Create an input function for training. drop_remainder = True for using TPUs.
        train_input_fn = bert.run_classifier.input_fn_builder(
            features=train_features,
            seq_length=ConfigFile.MAX_SEQ_LENGTH,
            is_training=True,
            drop_remainder=False)

        return estimator, train_input_fn, num_train_steps

    def trainModel(self, estimator, train_input_fn, num_train_steps):

        """
                                Method Name: trainModel
                                Description: Train the  Bert model
                                Return: Number of checkpoint steps to save
                        """

        print(f'Beginning Training!')
        current_time = datetime.now()
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
        print("Training took time ", datetime.now() - current_time)

        return estimator

    def evaluateModel(self, test_features, estimator):

        """
                                        Method Name: evaluateModel
                                        Description: Evaluate our model using the test dataset
                                        Return: Confusion matrix and accuracy
                                """

        test_input_fn = run_classifier.input_fn_builder(
            features=test_features,
            seq_length=ConfigFile.MAX_SEQ_LENGTH,
            is_training=False,
            drop_remainder=False)
        estimator.evaluate(input_fn=test_input_fn, steps=None)

    def executeProcessing(self, sentence):

        """
       Method Name: executeProcessing
       Description: Evaluate our model using the test dataset
    """

        train, test = self.download_and_load_datasets()

        train_InputExamples, test_InputExamples = self.processDataInBertFormat(train, test)
        tokenizer = self.create_tokenizer_from_hub_module()
        self.tokenizeSentences(tokenizer, sentence)

        train_features, test_features = self.convertDataIninpFeatures(tokenizer, train_InputExamples,
                                                                      test_InputExamples)

        estimator, train_input_fn, num_train_steps = self.prepareTraingParams(train_features)
        estimator = self.trainModel(estimator, train_input_fn, num_train_steps)

# if __name__ == "__main__":
#     trnMdlObj = TrainBertModel()
#     trnMdlObj.executeProcessing()
