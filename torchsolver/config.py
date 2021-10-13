import torch


class Config:
    def __init__(self, **kwargs):
        self.output_dir = "output"
        self.pretrained_file = None

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_device = torch.cuda.device_count()

        self.data_name = None
        self.data_args = Dict()

        self.train_data_name = None
        self.train_data_args = Dict()
        self.train_loader_args = Dict()

        self.val_data_name = None
        self.val_data_args = Dict()
        self.val_loader_args = Dict()

        self.do_val = True

        self.model_name = None
        self.model_args = Dict()

        self.loss_name = None
        self.loss_args = Dict()

        self.optimizer_name = None
        self.optimizer_args = Dict()

        self.scheduler_name = None
        self.scheduler_args = Dict()

        self.lr = 0.001
        self.batch_size = 32
        self.epochs = 20
        self.pretrained_file = None

        for k, v in kwargs.items():
            self.__setattr__(k, v)

    def build(self, **kwargs):
        if self.train_data_name is None:
            self.train_data_name = self.data_name

        if len(self.train_data_args) == 0 and len(self.data_args) > 0:
            self.train_data_args = self.data_args.copy()

        if self.val_data_name is None:
            self.val_data_name = self.data_name

        if len(self.val_data_args) == 0 and len(self.data_args) > 0:
            self.val_data_args = self.data_args.copy()

        if "lr" not in self.optimizer_args:
            self.optimizer_args.lr = self.lr

        if "batch_size" not in self.train_loader_args:
            self.train_loader_args.batch_size = self.batch_size

        if "shuffle" not in self.train_loader_args:
            self.train_loader_args.shuffle = True

        if "batch_size" not in self.val_loader_args:
            self.val_loader_args.batch_size = self.batch_size

        for k, v in kwargs.items():
            self.__setattr__(k, v)

        return self


class GANConfig(Config):
    def __init__(self, **kwargs):
        self.g_net_name = None
        self.g_net_args = Dict()

        self.d_net_name = None
        self.d_net_args = Dict()

        self.g_optimizer_name = None
        self.g_optimizer_args = Dict()

        self.d_optimizer_name = None
        self.d_optimizer_args = Dict()

        super().__init__(**kwargs)

    def build(self, **kwargs):
        super().build(**kwargs)

        if self.g_optimizer_name is None and self.optimizer_name is not None:
            self.g_optimizer_name = self.optimizer_name

        if len(self.g_optimizer_args) == 0 and len(self.optimizer_args) != 0:
            self.g_optimizer_args = self.optimizer_args.copy()

        if self.d_optimizer_name is None and self.optimizer_name is not None:
            self.d_optimizer_name = self.optimizer_name

        if len(self.d_optimizer_args) == 0 and len(self.optimizer_args) != 0:
            self.d_optimizer_args = self.optimizer_args.copy()

        self.optimizer_name = None
        self.model_name = None

        return self


class Dict(dict):
    def __getattr__(self, item):
        return self.get(item, None)

    def __setattr__(self, key, value):
        self[key] = value
