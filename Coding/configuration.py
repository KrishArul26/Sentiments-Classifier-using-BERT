class ConfigFile:

    # Training parameters
    DATA_COLUMN = 'sentence'
    LABEL_COLUMN = 'polarity'
    # label_list is the list of labels, i.e. True, False or 0, 1 or 'dog', 'cat'
    label_list = [0, 1]
    # This is a path to an uncased (all lowercase) version of BERT
    BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
    # We'll set sequences to be at most 128 tokens long.
    MAX_SEQ_LENGTH = 128

    # Compute train and warmup steps from batch size
    # These hyperparameters are copied from this colab notebook (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-5
    NUM_TRAIN_EPOCHS = 1.0
    # Warmup is a period of time where hte learning rate
    # is small and gradually increases--usually helps training.
    WARMUP_PROPORTION = 0.1
    # Model configs
    SAVE_CHECKPOINTS_STEPS = 100
    SAVE_SUMMARY_STEPS = 100

    OUTPUT_DIR = 'models'
