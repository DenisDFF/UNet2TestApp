from torch.optim import AdamW
from diffusers.optimization import get_cosine_schedule_with_warmup

def create_optimizer_and_scheduler(model, config, train_dataloader):
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )
    return optimizer, lr_scheduler