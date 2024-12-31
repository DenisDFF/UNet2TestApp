class Config:
    # Dataset configuration
    dataset_name = "huggan/smithsonian_butterflies_subset"
    image_size = 128  
    train_batch_size = 16 
    eval_batch_size = 8 
    learning_rate = 1e-4  
    lr_warmup_steps = 500 
    num_epochs = 10  
    save_image_epochs = 1  
    save_model_epochs = 1  
    mixed_precision = "bf16" 
    gradient_accumulation_steps = 1 
    output_dir = "./output" 
    push_to_hub = False  
    seed = 42  
    hub_model_id = None  