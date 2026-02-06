from model_vanilla import subsequent_mask
import torch
from torch.amp import autocast, GradScaler

import time

class Batch:
    # Unit of training data

    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank> The padding
        self.src = src # Original sentences
        self.src_mask = (src != pad).unsqueeze(-2) # [Batch, 1, Len]

        if tgt is not None:
            self.tgt = tgt[:, :-1] # Nothing to predict at the last place
            self.tgt_y = tgt[:, 1:] # Nothing to be predicted at the 1st
            self.tgt_mask = self.make_std_mask(self.tgt, pad)

            # Count the total valid tokens -> Normalize lengths 
            self.ntokens = self.tgt_y.sum().item()
            # Detach -> Not use .data anymore

    @staticmethod
    def make_std_mask(tgt, pad):
        m = (tgt != pad).unsqueeze(-2) 
        tgt_mask = (m & subsequent_mask(tgt.size(-1))).type_as(m)
        return tgt_mask
    
class TrainState:
    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total of examples
    tokens: int = 0  # total of tokens


def train_one_epoch(data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
    train_state=TrainState(),):


    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0

    scaler = GradScaler()
    for i, batch in enumerate(data_iter):

        with autocast(dtype= torch.bfloat16): # brain floating -> More EXP bits
            out = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
            loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)

            scaler.scale(loss_node).backward()

            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens

            if (i + 1) % accum_iter == 0:
                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                
                n_accum += 1
                train_state.accum_step += 1
            
            total_loss += loss.item() 

        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        
        return total_loss / total_tokens, train_state



    

    

