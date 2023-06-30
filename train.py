from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from data.data_pl import DeepFashion2DataModule
from models.model_pl import TransformerModelPL

datamodule = DeepFashion2DataModule(
    '/data/DeepFashion2/train',
    '/data/DeepFashion2/validation',
    batch_size=256,
)
datamodule.setup()
model = TransformerModelPL()
checkpoint_callback = ModelCheckpoint(
    save_top_k=3,
    monitor='class_accuracy_w0',
    mode='max',
    save_last=True,
    every_n_train_steps=1_000,
)
lr_callback = LearningRateMonitor('step')
trainer = Trainer(
    # gpus=0,
    max_epochs=100,
    callbacks=[checkpoint_callback, lr_callback],
    accumulate_grad_batches=1,
    # precision=16,
    # resume_from_checkpoint='lightning_logs/version_2/checkpoints/last.ckpt'
)
trainer.fit(model, datamodule)
