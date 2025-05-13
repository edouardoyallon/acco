#SBATCH --nodes=2                # Request 2 nodes
#SBATCH --gres=gpu:1             # 1 GPU per node
#SBATCH --ntasks-per-node=1      # 1 task per node

acco-ft:
    group_by_length : false
    batch_size: 4
    n_grad_accumulation: 4 
    learning_rate : 1e-5
    weight_decay: 0.0
    adam_beta1: 0.9
    adam_beta2: 0.95
    nb_steps_tot: 50000
    dataloader_num_workers : 1
    dataloader_pin_memory : True
    dataloader_persistent_workers : True
    label_smoothing_factor : 0
    max_length : 512
    scheduler_name : 'cosine'
    warmup : 0
    save: False
    use_mixed_precision: True
    n_warmup_steps: 0
    run_baseline_ddp: False # True for DDP
    method_name: 'acco' # ‘ddp’ for DDP
    #gradient_accumulation_steps : 1 # add for DDP
    eval: True
    eval_step: 10
    run_expe_slow: False
    const_len_batch: False
    finetune: True
