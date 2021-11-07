from bert import run_classifier
import tensorflow as tf
import tensorflow_hub as hub
import bert
from configuration import ConfigFile
from bert import optimization


class Prediction:

    def serving_input_receiver_fn(self):
        """
        Method Name: serving_input_receiver_fn
        Description:Creates an serving_input_receiver_fn for BERT
        """

        unique_ids = tf.placeholder(tf.int32, [None], name="unique_ids")
        input_ids = tf.placeholder(tf.int32, [None, ConfigFile.MAX_SEQ_LENGTH], name="input_ids")
        input_mask = tf.placeholder(tf.int32, [None, ConfigFile.MAX_SEQ_LENGTH], name="input_mask")
        segment_ids = tf.placeholder(tf.int32, [None, ConfigFile.MAX_SEQ_LENGTH], name="segment_ids")
        label_ids = tf.placeholder(tf.int32, [None], name="label_ids")
        return tf.estimator.export.build_raw_serving_input_receiver_fn(
            {
                "unique_ids": unique_ids,
                "input_ids": input_ids,
                "input_mask": input_mask,
                "segment_ids": segment_ids,
                "label_ids": label_ids,
            }
        )()

        return serving_input_receiver_fn

    def create_model(self, is_predicting, input_ids, input_mask, segment_ids, labels,
                     num_labels):

        """
               Method Name: create_model
               Description:Creates a classification model BERT
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

        # Use "pooled_output" for classification tasks on an entire sentence.
        # Use "sequence_outputs" for token-level output.
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
                Description:Returns `model_fn` closure for TPUEstimator
                """

        def model_fn(features, labels, mode, params):

            """
                            Method Name: model_fn
                            Description: RThe `model_fn` for TPUEstimator.
                            """

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

    def get_estimator(self, num_train_steps, num_warmup_steps):

        """
        Method Name: get_estimator
        Description: Builds BERT model_fn and tf.estimator.RunConfig from a param set
        and returns an estimator object
        Arguments.
                                    """


        # Specify outpit directory and number of checkpoint steps to save
        run_config = tf.estimator.RunConfig(
            model_dir=ConfigFile.OUTPUT_DIR,
            save_summary_steps=ConfigFile.SAVE_SUMMARY_STEPS,
            save_checkpoints_steps=ConfigFile.SAVE_CHECKPOINTS_STEPS)

        model_fn = self.model_fn_builder(
            num_labels=len(ConfigFile.label_list),
            learning_rate=ConfigFile.LEARNING_RATE,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps)

        # estimator = tf.estimator.Estimator(
        #     model_fn=model_fn, config=run_config, params={"batch_size": kwargs.get("train_batch_size")}
        # )
        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config,
            params={"batch_size": ConfigFile.BATCH_SIZE})
        return estimator



    def create_tokenizer_from_hub_module(self):

        """
                Method Name: create_tokenizer_from_hub_module
                Description: Builds BERT model_fn and tf.estimator.RunConfig from a param set
                and returns an estimator object
                Arguments.
                                            """

        with tf.Graph().as_default():
            bert_module = hub.Module(ConfigFile.BERT_MODEL_HUB)
            tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
            with tf.Session() as sess:
                vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                      tokenization_info["do_lower_case"]])

        return bert.tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=do_lower_case)

    def getPrediction(self, in_sentences, tokenizer, estimator):
        labels = ["Negative", "Positive"]
        input_examples = [run_classifier.InputExample(guid="", text_a=x, text_b=None, label=0) for x in
                          in_sentences]  # here, "" is just a dummy label
        input_features = run_classifier.convert_examples_to_features(input_examples, ConfigFile.label_list,
                                                                     ConfigFile.MAX_SEQ_LENGTH,
                                                                     tokenizer)
        predict_input_fn = run_classifier.input_fn_builder(features=input_features,
                                                           seq_length=ConfigFile.MAX_SEQ_LENGTH,
                                                           is_training=False, drop_remainder=False)
        predictions = estimator.predict(predict_input_fn)
        return [(labels[prediction['labels']]) for prediction in predictions]

    def executeProcessing(self, sentence_list):
        tokenizer = self.create_tokenizer_from_hub_module()
        # This two params will change if you are changing the training data
        num_train_steps = 468
        num_warmup_steps = 46
        estimator = self.get_estimator(num_train_steps, num_warmup_steps)
        # serving_input_receiver_fn = self.serving_input_receiver_fn()
        estimator.export_saved_model(ConfigFile.OUTPUT_DIR, self.serving_input_receiver_fn)
        predictions = self.getPrediction(sentence_list, tokenizer, estimator)
        return predictions

