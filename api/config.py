class CFG:
    class data:
        batch_size = 16
        lr = 1e-3  # 5e-5
        epochs = 10
        epsilon = 1e-8
        MAX_LEN = 128  # max sentence length
        seed_val = 42  # random seed
        k_folds = 10
        hidden_size = 768  # hidden layer size (embedding size) for feedforward net
        PATH = "./saved_models/hs.pth"

        # defaults for CNN
        dropout = 0.4
        Ks = [3, 4, 5, 6, 7]
        kernel_num = 8  # number of filters for each conv layer
        input_shape = [-1, 1, 128, 768]
