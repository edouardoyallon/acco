import os
import time
from collections.abc import Mapping
from typing import Optional
import datasets
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import (
    DataLoader,
    RandomSampler,
)
from torch.utils.tensorboard import SummaryWriter
from utils.hostli import expand_hostlist
from utils.logs_utils import ArgDict
from utils.trainer_utils import LabelSmoother
from transformers import DataCollatorForLanguageModeling

class DecoupledTrainerBase(object):
    """
    The 'Decoupled Data Parallel' base trainer.
    """

    def __init__(
            self,
            model=None,
            tokenizer=None,
            train_dataset=None,
            eval_dataset=None,
            args=None,
            log=None,
            text_column_name="text",
            preprocess_dataset_fn=None,
            run_name = '',
    ):
        # Initialization of constants
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.args = args
        self.batch_size = self.args.batch_size
        self.text_column_name = text_column_name
        self.label_smoothing_factor = self.args.label_smoothing_factor

        self.nb_grad_tot = self.args.nb_steps_tot  # self.args.n_batch_per_epoch * self.args.n_epoch_if_1_worker

        # Initialization of the logs
        self.log = log
        self.epoch = 0

        self.initialize_com()
        
        # if a tensorboard dir doesn't exist, makes it
        tensorboard_dir = os.getcwd() + '/tensorboard/'
        if self.rank == 0:
            if not os.path.exists(tensorboard_dir):
                os.makedirs(tensorboard_dir, exist_ok=True)
        # initialize a tensorboard writer
        path_tensorboard = tensorboard_dir +f'/{run_name}/'+str(self.id_run)
        self.writer = SummaryWriter(path_tensorboard)

        if self.args.label_smoothing_factor != 0:
            self.label_smoother = LabelSmoother(
                epsilon=self.args.label_smoothing_factor
            )
        else:
            self.label_smoother = None

        # func to use for preprocessing data
        self.preprocess_dataset_fn = preprocess_dataset_fn

        # split data into shards
        self.prepare_data()
        # Initialization of dataloader
        
        def tokenize_data(element):
            #Concat all text id seperated with eos id
            #return sequences of fixed size (discard last seq if len()< max)
            outputs = tokenizer(element[text_column_name], truncation=True , max_length = args.max_length)
            # return attention mask + input_ids
            return outputs
        
        def tokenize_data_const_len(element):
            #Concat all text id seperated with eos id
            #return sequences of fixed size (discard last seq if len()< max)
            outputs = tokenizer(element[text_column_name], truncation=False)
            input_batch = []
            input_ids_concat = []
            context_length = args.max_length
            for input_ids in outputs["input_ids"]:
                input_ids.append(tokenizer.eos_token_id)
                input_ids_concat+=input_ids
            batch_size = len(input_ids_concat)//context_length
            input_batch = torch.tensor(input_ids_concat[:batch_size*context_length]).reshape(batch_size, context_length)
            # no pad => no attention_mask
            return {"input_ids": input_batch }

    
        if self.preprocess_dataset_fn is not None:
            self.train_dataset = self.train_dataset.map(
                self.preprocess_dataset_fn, batched=True
            )
            if self.eval_dataset is not None:
                self.eval_dataset = self.eval_dataset.map(
                    self.preprocess_dataset_fn, batched=True
                )
        if 'input_ids' not in self.train_dataset.column_names:
            if self.args.const_len_batch:
                self.train_dataset = self.train_dataset.map(
                    tokenize_data_const_len, batched=True, remove_columns=self.train_dataset.column_names, num_proc=self.args.dataloader_num_workers
                )
                if self.eval_dataset is not None:
                    self.eval_dataset = self.eval_dataset.map(
                        tokenize_data_const_len, batched=True, remove_columns=self.eval_dataset.column_names, num_proc=self.args.dataloader_num_workers
                )
            else:
                self.train_dataset = self.train_dataset.map(
                    tokenize_data, batched=True, remove_columns=self.train_dataset.column_names, num_proc=self.args.dataloader_num_workers
                )
                if self.eval_dataset is not None:
                    self.eval_dataset = self.eval_dataset.map(
                        tokenize_data, batched=True, remove_columns=self.eval_dataset.column_names, num_proc=self.args.dataloader_num_workers
                )
            

        self.train_dataloader = self.get_train_dataloader()
        if self.eval_dataset is not None:
            self.eval_dataloader = self.get_eval_dataloader()
        
    def _data_collator(batch):
        return {'input_ids':torch.stack([torch.LongTensor(seq['input_ids']) for seq in batch])}
    

    def initialize_com(self):
        # get distributed configuration from Slurm environment
        self.node_id = os.environ["SLURM_NODEID"]
        self.rank = int(os.environ["SLURM_PROCID"])
        self.id_run = os.environ["SLURM_JOBID"]
        self.local_rank = int(os.environ["SLURM_LOCALID"])
        self.world_size = int(os.environ["SLURM_NTASKS"])
        # get node list from slurm
        hostnames = expand_hostlist(os.environ["SLURM_JOB_NODELIST"])
        self.n_nodes = len(hostnames)
        # get IDs of reserved GPU
        self.gpu_ids = os.environ["SLURM_STEP_GPUS"].split(",")
        # define MASTER_ADD & MASTER_PORT, used to define the distributed communication environment
        self.master_addr = hostnames[0]
        self.master_port = 12346 + int(
            min(self.gpu_ids)
        )  # to avoid port conflict on the same node
        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
        
        if self.rank == 0:
            self.log.info(f">>> Training on {self.n_nodes} nodes and {self.world_size} GPUs")
        self.log.info(
            "- Process {} corresponds to GPU {} of node {}".format(
                self.rank, self.local_rank, self.node_id
            )
        )
        
        # loads the weights into a 1D tensor to ease handling of parameters
        self.dtype = torch.bfloat16 if self.args.use_mixed_precision else torch.float32
        if self.args.run_baseline_ddp:
            # loads model into default precision if ddp is used
            self.model.to(self.local_rank, dtype=torch.float32)
        else:
            self.model.to(self.local_rank, dtype=self.dtype)
        # gather the params of the model into a 1D tensor
        self.params = self.get_weights()
        # loads the client's parameters into the model
        self.set_weights(self.params)
        
        # initialize the process group for communications
        self.process_group = dist.init_process_group(
            backend="nccl", rank=self.rank, world_size=self.world_size
        )
        # initialize model weights by performing a first all-reduce
        dist.all_reduce(self.params, group=self.process_group, op=dist.ReduceOp.AVG)

        
    def prepare_data(self):
        if (
                self.train_dataset is not None
                and isinstance(self.train_dataset, torch.utils.data.IterableDataset)
                and self.args.group_by_length
        ):
            raise ValueError(
                "the `--group_by_length` option is only available for `Dataset`, not `IterableDataset"
            )

        if self.train_dataset is not None:
            self.train_dataset = self.train_dataset.shard(
                num_shards=self.world_size, index=self.rank
            )
        if self.eval_dataset is not None:
            self.eval_dataset = self.eval_dataset.shard(
                num_shards=self.world_size, index=self.rank
            )


    def get_train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset

        if self.args.const_len_batch:
            data_collator = DecoupledTrainerBase._data_collator
        else:
            data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        dataloader_params = {
            "batch_size": self.args.batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "drop_last": True,
            "pin_memory_device": f"cuda:{self.local_rank}",
        }
        dataloader_params["sampler"] = RandomSampler(self.train_dataset)
        return DataLoader(train_dataset, **dataloader_params)
    

    def get_eval_dataloader(self) -> DataLoader:
        eval_dataset = self.eval_dataset
        if self.args.const_len_batch:
            data_collator = DecoupledTrainerBase._data_collator
        else:
            data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        dataloader_params = {
            "batch_size": self.args.batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "drop_last": True,
            "pin_memory_device": f"cuda:{self.local_rank}",
        }
        return DataLoader(eval_dataset, **dataloader_params)


    def _prepare_input(self, data):
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.local_rank, "non_blocking": True}
            return data.to(**kwargs)
        return data

    def _prepare_inputs(self, inputs):
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(**inputs)

        if labels is not None:
            loss = self.label_smoother(outputs, labels, shift_labels=True)

        else:

            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    @torch.no_grad()
    def get_weights(self):
        """
        Wrapper around nn.utils.parameters_to_vector.
        Given a nn.Module, returns a 1D tensor containing all of its parameters.
        """
        return nn.utils.parameters_to_vector(self.model.parameters())
    

    @torch.no_grad()
    def set_weights(self, weights):
        """
        Wrapper around nn.utils.vector_to_parameters.
        Given a 1D tensor containing a nn.Module parameters,
        loads the parameters into the nn.Module.
        """
        nn.utils.vector_to_parameters(weights, self.model.parameters())
        
        
    @torch.no_grad()    
    def set_grads(self,grads):
        """
        Wrapper around nn.utils.vector_to_parameters.
        Given a 1D tensor containing a nn.Module parameters,
        loads the grads into the nn.Module parameters.grads.
        """

        # Pointer for slicing the vector for each parameter
        pointer = 0
        for param in self.model.parameters():
            grad = param.grad
            # The length of the parameter
            num_grad = grad.numel()
            # Slice the vector, reshape it, and replace the old data of the parameter
            grad.data = grads[pointer:pointer + num_grad].view_as(grad).data
            # Increment the pointer
            pointer += num_grad
            

    @torch.no_grad()
    def get_grads(self,):
        """
        Wrapper around nn.utils.parameters_to_vector.
        Given a nn.Module, returns a 1D tensor containing all of its parameters.grad.
        """
        vec = []
        for param in self.model.parameters():
            vec.append(param.grad.view(-1))
        return torch.cat(vec)