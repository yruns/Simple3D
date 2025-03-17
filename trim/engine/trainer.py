import weakref

from trim.callbacks.misc import *
from trim.engine.defaults import Mode
from trim.utils import dist
from trim.utils.events import EventStorage


class TrainerBase(object):
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        self.scaler = None
        self.logger = None
        self.wandb = None
        self.callbacks = []
        self.stroage: EventStorage
        self.comm_info = dict()
        self.best_metric_value = float("-inf")
        self.best_metric_epoch = -1
        self.output_dir = None

        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = None
        self.data_iterator = None
        self.debug = False

    def training_step(self, batch_data, batch_index):
        pass

    def test_step(self, batch_data, batch_index):
        pass

    def on_training_epoch_start(self):
        if dist.get_world_size() > 1:
            self.train_loader.sampler.set_epoch(self.epoch)
        for callback in self.callbacks:
            callback.on_training_epoch_start()

    def on_training_epoch_end(self):
        for callback in self.callbacks:
            callback.on_training_epoch_end()
        self.storage.reset_histories()  # reset histories (required)
        self.wandb.save(os.path.join(self.output_dir, "train.log"))

    def on_training_step_start(self):
        for callback in self.callbacks:
            callback.on_training_step_start()

    def on_training_step_end(self):
        for callback in self.callbacks:
            callback.on_training_step_end()

    def on_training_phase_start(self):
        for callback in self.callbacks:
            callback.on_training_phase_start()

    def on_training_phase_end(self):
        for callback in self.callbacks:
            callback.on_training_phase_end()
        self.wandb.finish()

    def configure_callbacks(self):
        for callback in self.callbacks:
            assert isinstance(callback, CallbackBase)
            callback.trainer = weakref.proxy(self)

    def configure_optimizers(self):
        pass

    def configure_model(self):
        pass

    def configure_dataloader(self):
        pass

    def configure_criterion(self):
        pass

    def configure_scaler(self):
        pass

    def configure_wandb(self):
        pass

    def setup(self, mode=Mode.TRAIN):
        self.configure_wandb()
        self.configure_model()
        self.configure_dataloader()
        self.configure_optimizers()
        self.configure_criterion()
        self.configure_scaler()
        self.configure_callbacks()

        dist.synchronize()

    def fit(self):
        self.setup()
        with EventStorage() as self.storage:
            self.on_training_phase_start()
            self.logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
            self.model.zero_grad()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                self.model.train()
                if dist.is_distritubed():
                    self.train_loader.sampler.set_epoch(self.epoch)
                if self.debug:
                    from itertools import islice
                    self.data_iterator = enumerate(islice(self.train_loader, 5))
                    self.val_loader = list(islice(self.val_loader, 10))
                else:
                    self.data_iterator = enumerate(self.train_loader)
                self.on_training_epoch_start()
                # => run_epoch
                for batch_index, batch_data in self.data_iterator:
                    self.comm_info["iter"] = batch_index
                    self.on_training_step_start()
                    self.training_step(batch_data, batch_index)
                    self.on_training_step_end()
                # => after epoch
                self.on_training_epoch_end()
            # => after train
            self.on_training_phase_end()

    def test(self):
        self.setup(mode=Mode.VAL)
        with EventStorage() as self.storage:
            self.logger.info(">>>>>>>>>>>>>>>> Start Testing >>>>>>>>>>>>>>>>")

            self.on_training_phase_start()
            self.on_training_epoch_start()
            # self.on_training_step_start()
            # self.on_training_step_end()
            self.on_training_epoch_end()
            self.on_training_phase_end()

            self.logger.info(">>>>>>>>>>>>>>>> End Testing >>>>>>>>>>>>>>>>")
