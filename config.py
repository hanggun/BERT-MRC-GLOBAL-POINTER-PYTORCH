class config:

    pretrained_model_path = './bert_pretrained/'
    train_data_path = "../ner_data/train_data.txt"
    dev_data_path = "../ner_data/test_data1.txt"
    test_data_path = "../ner_data/test_data1.txt"
    label_path = './label2id.json'

    loss_type = 'ce'

    max_seq_len = 64
    hidden_size = 768
    dropout_rate = 0.2
    num_labels = 8
    continue_train = False

    batch_size = 8
    gradient_accumulation_steps = 1
    max_epoches = 3
    lr = 3e-5
    other_lr = 2e-4
    weight_decay = 0.01
    adam_epsilon = 1e-6
    warmup_proportion = 0.1

    log_every = 200
    valid_epoch = 1
    model_save_dir = './model_results_new/'