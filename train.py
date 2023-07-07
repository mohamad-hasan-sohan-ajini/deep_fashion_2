from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from data.data_pl import DeepFashion2DataModule
from models.model_pl import TransformerModelPL

datamodule = DeepFashion2DataModule(
    '/data/DeepFashion2/train',
    '/data/DeepFashion2/validation',
    batch_size=24,
)
datamodule.setup()
model = TransformerModelPL()
logger = TensorBoardLogger('.')
checkpoint_callback = ModelCheckpoint(
    save_top_k=3,
    monitor='class_accuracy_w0',
    mode='max',
    save_last=True,
    every_n_train_steps=1000,
)
lr_callback = LearningRateMonitor('step')
trainer = Trainer(
    # gpus=0,
    # precision=16,
    max_epochs=1000,
    callbacks=[checkpoint_callback, lr_callback],
    accumulate_grad_batches=1,
    log_every_n_steps=100,
    logger=logger,
)
trainer.fit(model, datamodule)
