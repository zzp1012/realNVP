import os, torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchvision.utils import save_image, make_grid
import numpy as np

# import internal libs
from model.realnvp import RealNVP
from data.utils import logit_transform
from utils import get_logger
from config import SCALE_REG

def train(save_path: str,
          device: torch.device,
          train_split: Subset,
          val_split: Subset,
          flow: RealNVP,
          batch_size: int,
          lr: float,
          momentum: float,
          decay: float,
          epochs: int,
          sample_size: int,
          scale_reg: float = SCALE_REG) -> None:
    """train realNVP model.

    Args:
        save_path: path to save model.
        device: device to use.
        data_info: data information.
        train_split: train dataset.
        val_split: validation dataset.
        flow: realNVP model.
        batch_size: batch size.
        lr: learning rate.
        momentum: momentum.
        decay: learning rate decay.
        epochs: epochs.
        sample_size: number of samples to generate.
    
    Returns:
        None
    """
    logger = get_logger(__name__)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # define the dataloader
    train_loader = DataLoader(train_split,
        batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_split,
        batch_size=batch_size, shuffle=False, num_workers=2)

    # define the optimizer
    optimizer = optim.Adamax(flow.parameters(), lr=lr, betas=(momentum, decay), eps=1e-7)

    # start
    for epoch in range(1, epochs+1):
        logger.info(f"####Epoch {epoch}...")

        # train
        flow.train()
        for batch_idx, data_ in enumerate(train_loader, 1):
            x, _ = data_
            # log-determinant of Jacobian from the logit transform
            x, log_det = logit_transform(x)
            x = x.to(device)
            log_det = log_det.to(device)

            # log-likelihood of input minibatch
            log_ll, weight_scale = flow(x)
            log_ll = (log_ll + log_det).mean()

            # add L2 regularization on scaling factors
            loss = -log_ll + scale_reg * weight_scale

            # update the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                logger.info('[%d/%d]\tloss: %.3f\tlog-ll: %.3f' % \
                    (batch_idx*batch_size, len(train_loader.dataset), loss.item(), log_ll.item()))

        # evaluation
        flow.eval()
        with torch.no_grad():
            for batch_idx, data_ in enumerate(val_loader, 1):
                x, _ = data_
                # log-determinant of Jacobian from the logit transform
                x, log_det = logit_transform(x)
                x = x.to(device)
                log_det = log_det.to(device)

                # log-likelihood of input minibatch
                log_ll, weight_scale = flow(x)
                log_ll = (log_ll + log_det).mean()

                # add L2 regularization on scaling factors
                loss = -log_ll + scale_reg * weight_scale

            # sample from the model
            samples = flow.sample(sample_size)
            samples, _ = logit_transform(samples, reverse=True)
            image_path = os.path.join(save_path, f"samples")
            os.makedirs(image_path, exist_ok=True)
            save_image(make_grid(samples),
                       os.path.join(image_path, f"{epoch}.png"))

        # save the model
        if epoch % 10 == 0 or epoch == epochs:
            torch.save(flow.state_dict(), os.path.join(save_path, f"model_epoch{epoch}.pt"))
            logger.info(f'Saved model to {save_path}')

    logger.info('Training finished at epoch %d.' % epoch)