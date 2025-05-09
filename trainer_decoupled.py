import os
import math
import time
import torch
import torch.distributed as dist
import multiprocessing as mp
from threading import Thread
from torch.optim import AdamW
from omegaconf import OmegaConf
from transformers import get_scheduler #, AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer
from utils.logs_utils import print_training_evolution, log_to_tensorboard, create_dict_result, save_result
from trainer_base import DecoupledTrainerBase


#@torch.compile
def gradient_step(
    dtype,
    model,
    count_grad_local,
    inputs,
    loss_static,
    n_grad_acc_ddp,
):
    with torch.autocast(device_type='cuda', dtype=dtype):
        # forward pass
        if 'labels' in inputs.keys():
            outputs = model(**inputs)

        else:
            outputs = model(**inputs, labels=inputs['input_ids'])
        
        loss = outputs[0]/n_grad_acc_ddp
    # backward to accumulate grads
    loss.backward()
    count_grad_local.add_(1)
     # register the loss
    loss_static.fill_(loss*n_grad_acc_ddp)


#@torch.compile
def update_buffers_step(
    params,
    com_buffer,
    len_params,
    count_grad_this_round,
    count_grad_local,
    count_after_init=-1,
):
    # update the local params with the latest all-reduced ones
    params.copy_(com_buffer[:len_params])
    # update the com buffer with the accumulated gradients computed
    # in the last computation round
    com_buffer[:len_params].copy_(params.grad)
    # put the number of local grads in the comm buffer
    count_grad_this_round.copy_(count_grad_local)
    # reset the grad buffer and grad count every other step
    if count_after_init == -1 or (count_after_init >-1 and count_after_init%2 == 0):
        # zero grad the local gradients
        params.grad.fill_(0)
        # reinit the local grad one
        count_grad_local.fill_(0)
    

#@torch.compile
def communication_step(
    rank,
    size_slice,
    size_local_slice,
    process_group,
    count_grad_this_round,
    com_buffer,
    params_opt,
    sharded_optimizer,
    scheduler,
    count_after_init=-1,
):
    if count_after_init >-1 and count_after_init%2 == 0:
        buffer_temp = params_opt.detach().clone()
        if count_after_init > 0:
            step_temp = sharded_optimizer.state_dict()['state'][0]['step'].detach().clone()
            exp_avg_temp = sharded_optimizer.state_dict()['state'][0]['exp_avg'].detach().clone()
            exp_avg_sq_temp = sharded_optimizer.state_dict()['state'][0]['exp_avg_sq'].detach().clone()
    # 1. all-reduce the grad_counts so that we know how many grads have been performed in total
    op = dist.all_reduce(count_grad_this_round, group=process_group, op=dist.ReduceOp.SUM, async_op=True)
    # 2. Reduce scatter the accumulated gradients (inside com buffer)
    dist.reduce_scatter_tensor(
        com_buffer[rank*size_slice : (rank+1)*size_slice],
        com_buffer,
        group=process_group,
        op=dist.ReduceOp.SUM,
    )
    # 3. loads the grads of the params to optimize
    params_opt.grad.copy_(com_buffer[rank*size_slice : rank*size_slice+size_local_slice])
    # 4. averages the grads
    op.wait()
    params_opt.grad.mul_(1/count_grad_this_round)
    # 5. take the sharded optimizer step
    sharded_optimizer.step()
    # take the right number of scheduler steps
    if count_after_init == -1 or (count_after_init >-1 and count_after_init%2 == 1):
        scheduler.step()
        scheduler._step_count += count_grad_this_round[0] - 1
    # 6. copy the updated params into the communication buffer
    com_buffer[rank*size_slice : rank*size_slice+size_local_slice].copy_(params_opt)
    # 7. all gather the params
    dist.all_gather_into_tensor(
        com_buffer,
        com_buffer[rank*size_slice : (rank+1)*size_slice],
        group=process_group,
    )
    if count_after_init >-1 and count_after_init%2 == 0:
        params_opt.copy_(buffer_temp)
        del buffer_temp
        if count_after_init > 0:
            sharded_optimizer.state_dict()['state'][0]['step'].copy_(step_temp)
            sharded_optimizer.state_dict()['state'][0]['exp_avg'].copy_(exp_avg_temp)
            sharded_optimizer.state_dict()['state'][0]['exp_avg_sq'].copy_(exp_avg_sq_temp)
            del exp_avg_temp
            del exp_avg_sq_temp
            del step_temp
        else:
            # if we are at step 0, we restart from an empty dict at state
            sharded_optimizer.state_dict()['state'] = {}
        
    

def com_routine(nb_grad_tot,
                com_stream,
                com_event,
                barrier_com_update,
                count_grad_tot,
                cuda_device,
                com_finished,
                rank,
                size_slice,
                size_local_slice,
                process_group,
                count_grad_this_round,
                com_buffer,
                params_opt,
                sharded_optimizer,
                scheduler,
                count_after_init,
               ):
    com_stream.wait_stream(torch.cuda.default_stream(cuda_device))
    with torch.cuda.stream(com_stream):
        while count_grad_tot.value < nb_grad_tot:
            # perform the communication steps and optimizer steps
            communication_step(
                rank,
                size_slice,
                size_local_slice,
                process_group,
                count_grad_this_round,
                com_buffer,
                params_opt,
                sharded_optimizer,
                scheduler,
                count_after_init.value,
            )
            # record the event
            com_event.record(stream=com_stream)
            com_event.synchronize()
            com_finished.value = 1
            # wait the end of a computation and update
            barrier_com_update.wait()

class DecoupledTrainer(DecoupledTrainerBase):
    """
    The 'Decoupled Trainer' with sharded optimizers.
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
        super().__init__(
            model,
            tokenizer,
            train_dataset,
            eval_dataset,
            args,
            log,
            text_column_name,
            preprocess_dataset_fn,
            run_name,
        )
        
        # create static inputs for the cuda graph and use them to prepare the grad vector
        #self.input_ids_static = torch.randint(low=0, high=self.args.vocab_size, size=(self.batch_size, self.args.max_length), device=self.local_rank)

        self.loss_static = torch.zeros(1, device=self.local_rank, dtype=torch.float32)
        # create a data iterator
        self.train_iterator = iter(self.train_dataloader)
        if self.eval_dataset is not None:
            self.eval_iterator = iter(self.eval_dataloader)
        # for ddp runs, filled with ones so it won't affect the training of non-ddp runs
        self.n_grad_acc_ddp = 1 #torch.ones(1, device=self.local_rank, dtype=torch.float32).squeeze()
        
        if self.args.run_baseline_ddp:
            self.prepare_ddp()
        else:
            self.prepare_grads()
            self.prepare_buffer_com()
            self.prepare_opt()

            # initalize a cuda event for the "end of all-reduce" as well as "end of update"
            self.com_event = torch.cuda.Event(blocking=True)
            self.update_event = torch.cuda.Event(blocking=True)
            # initialize the two concurrent streams
            self.com_stream = torch.cuda.Stream(device=self.local_rank)
            self.grad_stream = torch.cuda.Stream(device=self.local_rank)
            self.device = torch.device(self.local_rank)
            
            
    def prepare_ddp(self,):
        
        self.ddp_model = DDP(self.model, process_group=self.process_group)
        self.optimizer = ZeroRedundancyOptimizer(
            self.ddp_model.parameters(),
            optimizer_class=AdamW,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
        )
        self.scheduler = get_scheduler(
            self.args.scheduler_name,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.warmup,
            num_training_steps=self.nb_grad_tot,
        )
        
        
    def prepare_buffer_com(self,):
        self.len_params = len(self.params)
        self.log.info(f'Worker {self.rank} training {self.len_params} parameters')
        # a slice is of size len_params/word_size
        # if world_size doesn't divide len_params,
        # the last slice will contain some zeros.
        self.size_slice = math.ceil(self.len_params / self.world_size)
        buffer_size = self.size_slice*self.world_size
        # if we are not the last slice, or the len or params is divided by the slice size
        if self.rank < self.world_size -1 or self.len_params % self.size_slice == 0:
            # then our slice is "fully" filled
            self.size_local_slice = self.size_slice
        # else, it means we are the last slice and it is not a full one
        else:
            # its size is the rest of the division of len_params by size slice
            self.size_local_slice = self.len_params % self.size_slice 
        self.com_buffer = torch.zeros(buffer_size, device=self.local_rank, dtype=self.dtype)
       # fills the buffer with the params or the grads
        if self.args.n_warmup_steps > 0:
            self.com_buffer[:self.len_params].copy_(self.params)
            init_count = 0
        else:
            self.com_buffer[:self.len_params].copy_(self.params.grad)
            init_count = 1
        # init at 0 the count of gradients
        self.count_grad_this_round = torch.zeros(1, device=self.local_rank, dtype=torch.int).fill_(init_count)
        
    
    def prepare_grads(self):
        
        # loads the next batch of data
        inputs = self.load_next_batch_into_static_memory()
        with torch.autocast(device_type='cuda', dtype=self.dtype):
            # forward pass
            if 'labels' in inputs.keys():
                outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs, labels = inputs['input_ids'])
            loss = outputs[0]
        # backward to accumulate grads
        loss.backward()
        # get grads
        grad_vector = self.get_grads()
        # make sure this 1D tensor is the one used for the model's grads
        self.set_grads(grad_vector)
        # points towards it in params.grad
        self.params.grad = grad_vector
        if self.args.n_warmup_steps > 0:
            # zeros the dummy grad
            self.params.grad.fill_(0)
        
        
    def prepare_opt(self,):
        self.params_opt = torch.zeros(self.size_local_slice, device=self.local_rank, dtype=torch.float32)
        # copy the value of params into the slice
        self.params_opt.copy_(self.params[self.rank*self.size_slice : self.rank*self.size_slice + self.size_local_slice])
        self.params_opt.grad = torch.zeros(self.size_local_slice, device=self.local_rank, dtype=torch.float32)
        # fix torch compile errors, see https://github.com/pytorch/pytorch/issues/107076
        lr_tensor = torch.tensor(self.args.learning_rate, requires_grad=False)
        self.sharded_optimizer = AdamW(
            [self.params_opt],
            lr=lr_tensor,
            capturable=True,
            weight_decay=self.args.weight_decay,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
        )
        self.scheduler = get_scheduler(
            self.args.scheduler_name,
            optimizer=self.sharded_optimizer,
            num_warmup_steps=self.args.warmup,
            num_training_steps=self.nb_grad_tot,
        )
    
    
    def warmup_steps(self, n_warmup_steps):
        warmup_stream = torch.cuda.Stream(device=self.local_rank)
        warmup_stream.wait_stream(torch.cuda.default_stream(self.device))
        with torch.cuda.stream(warmup_stream):
            for k in range(n_warmup_steps):
                # copy the buffer's params in the params tensor
                self.params.copy_(self.com_buffer[:self.len_params])
                # use gradient accumulation
                for step in range(self.args.n_grad_accumulation):
                    # loads the next batch of data
                    inputs = self.load_next_batch_into_static_memory()
                    # compute the gradients
                    gradient_step(
                        self.dtype,
                        self.model,
                        self.count_grad_local,
                        inputs,
                        self.loss_static,
                        self.n_grad_acc_ddp,
                    )
                # put them in the com buffer
                update_buffers_step(
                    self.params,
                    self.com_buffer,
                    self.len_params,
                    self.count_grad_this_round,
                    self.count_grad_local,
                )
                # average gradients and perform an optimizer's step
                communication_step(
                    self.rank,
                    self.size_slice,
                    self.size_local_slice,
                    self.process_group,
                    self.count_grad_this_round,
                    self.com_buffer,
                    self.params_opt,
                    self.sharded_optimizer,
                    self.scheduler,
                )
        torch.cuda.default_stream(self.device).wait_stream(warmup_stream)
        # to initialize the next round of com, need to compute grads and put them in the buffer
        # 1. copy the buffer's params in the params tensor
        self.params.copy_(self.com_buffer[:self.len_params])
        # use gradient accumulation
        for step in range(self.args.n_grad_accumulation):
            # 2. loads the next batch of data
            inputs = self.load_next_batch_into_static_memory()
            # 3. compute the grads
            gradient_step(
                self.dtype,
                self.model,
                self.count_grad_local,
                inputs,
                self.loss_static,
                self.n_grad_acc_ddp,
            )
        # 4. update the buffers
        update_buffers_step(
            self.params,
            self.com_buffer,
            self.len_params,
            self.count_grad_this_round,
            self.count_grad_local,
            count_after_init=-2,
        )
          
    
    def load_next_batch_into_static_memory(self):
        try:
            # gather the next batch of data
            inputs = next(self.train_iterator)
        except StopIteration:
            # When the epoch ends, start a new epoch.
            self.train_iterator = iter(self.train_dataloader)
            inputs = next(self.train_iterator)
        # to local device
        inputs = {k: v.to(device=self.local_rank) for k, v in inputs.items()}
        # to static memory
        return inputs
        
    @torch.no_grad()
    def eval_loop(self):
        self.eval_iterator = iter(self.eval_dataloader)
        self.model.eval()
        loss = []
        for inputs in self.eval_iterator:
            # to local device
            inputs = {k: v.to(device=self.local_rank) for k, v in inputs.items()}
            with torch.autocast(device_type='cuda', dtype=self.dtype): 
                if 'labels' in inputs.keys():
                    outputs = self.model(**inputs)
                else:
                    outputs = self.model(**inputs, labels=inputs['input_ids'])    
            loss.append(outputs[0].cpu())
        self.model.train()
        print(torch.mean(torch.tensor(loss)))
        return torch.mean(torch.tensor(loss))
    
    
    def train(self):
        self.t_last_epoch = time.time()
        self.t_beg = time.time()
        
        if self.args.method_name == 'acco':
            self.train_acco()
        elif self.args.method_name == 'ddp':
            self.train_ddp()
        elif self.args.method_name == 'dpu':
            self.train_dpu()
        else:
            raise ValueError("You must select one of the following method_name: 'acco', 'ddp', 'dpu'")
    
    def train_acco(self):
        last_eval = 0
        time_checkpoint = time.time()
        t_begin = time.time()
        # performs n_warmup steps SEQUENTIALLY
        if self.args.n_warmup_steps > 0:
            self.count_grad_local = torch.zeros(1, device=self.local_rank, dtype=torch.int)
            self.warmup_steps(self.args.n_warmup_steps)
            grad_init = self.world_size*(self.args.n_warmup_steps + 1)*self.args.n_grad_accumulation
        else:
            self.count_grad_local = torch.ones(1, device=self.local_rank, dtype=torch.int)
            grad_init = 0
        # init the counts after warmup
        count_grad_tot = mp.Value('i', grad_init)
        count_com = self.args.n_warmup_steps
        count_after_init = mp.Value('i', 0)
        # to sync CPU and GPU at the end of grad step
        end_of_grad = torch.cuda.Event()
        self.com_finished = mp.Value('i', 0)
        # to sync com thread and update
        barrier_com_update = mp.Barrier(2)
        # launch the communication routine in a separate thread
        com_thread = Thread(
            target=com_routine,
            args=(
                self.nb_grad_tot,
                self.com_stream,
                self.com_event,
                barrier_com_update,
                count_grad_tot,
                self.device,
                self.com_finished,
                self.rank,
                self.size_slice,
                self.size_local_slice,
                self.process_group,
                self.count_grad_this_round,
                self.com_buffer,
                self.params_opt,
                self.sharded_optimizer,
                self.scheduler,
                count_after_init,
            ),
        )
        com_thread.start()
        # grad step in the grad stream
        self.grad_stream.wait_stream(torch.cuda.default_stream(self.device))
        with torch.cuda.stream(self.grad_stream):
            while count_grad_tot.value < self.nb_grad_tot:
                # use gradient accumulation
                for step in range(self.args.n_grad_accumulation):
                    # loads next batch in memory
                    inputs = self.load_next_batch_into_static_memory()
                    # perform a grad step
                    gradient_step(
                        self.dtype,
                        self.model,
                        self.count_grad_local,
                        inputs,
                        self.loss_static,
                        self.n_grad_acc_ddp,
                    )
                # waits the end of the grad step
                end_of_grad.record(stream=self.grad_stream)
                end_of_grad.synchronize()
                # if the com finished
                if self.com_finished.value:
                    # reset the bool
                    self.com_finished.value = 0
                    # update the current count grad tot
                    if count_after_init.value%2 == 1:
                        count_grad_tot.value += self.count_grad_this_round[0]
                    # updates the buffers
                    update_buffers_step(
                        self.params,
                        self.com_buffer,
                        self.len_params,
                        self.count_grad_this_round,
                        self.count_grad_local,
                        count_after_init.value,
                    )
                    # record the end of the update
                    self.update_event.record(stream=self.grad_stream)
                    self.update_event.synchronize()
                    # update the com count and reinit the local grad one
                    count_com += 1
                    count_after_init.value += 1
                    # tells the communication process to continue
                    barrier_com_update.wait()
                    barrier_com_update.reset()




                    if self.rank==0:
                        # evalloop
                        eval_loss = None
                        if (self.args.eval == True) and (self.eval_dataset is not None):
                            if count_grad_tot.value - last_eval > self.args.eval_step:
                                eval_loss = self.eval_loop()
                                last_eval = count_grad_tot.value
                        
                        # logs the evolution of training
                        delta_step_for_log=10
                        log_to_tensorboard(
                            self.writer,
                            count_com//2, # the nb of opt step is every 2 coms
                            count_grad_tot.value,
                            self.rank,
                            self.loss_static,
                            eval_loss,
                            self.t_beg,
                            delta_step_for_log,
                            self.epoch,
                        )
                        # print the evolution in the logs
                        self.epoch, self.t_last_epoch = print_training_evolution(
                            self.log,
                            count_grad_tot.value,
                            count_com,
                            delta_step_for_log,
                            self.rank,
                            self.t_beg,
                            self.t_last_epoch,
                            self.loss_static,
                            self.epoch,
                        )

                        if self.args.save:
                            total_time_last_checkpoint = time.time() - time_checkpoint



                            if total_time_last_checkpoint >= 1800:  # 1800 seconds = 30 minutes
                                # Save the checkpoint
                                # Reset self.time_checkpoint to the current time for the next comparison
                                time_checkpoint = time.time()
                                path_checkpoint = os.getcwd() + "/checkpoints/"
                                if not os.path.exists(path_checkpoint):
                                    os.mkdir(path_checkpoint)
                                path_save_model = path_checkpoint + self.id_run + "_model_"+str(count_grad_tot.value)+".pt"
                                #path_save_optim = path_checkpoint + self.id_run + "_optim_"+str(count_grad_tot.value)+".pt"
                                torch.save(self.model.state_dict(), path_save_model)
                                #torch.save(self.sharded_optimizer.state_dict(), path_save_optim)
            # waits the end of the communication thread            
            com_thread.join()
            # LOGS for the end of training
            t_end = time.time()
            total_time = t_end - self.t_beg
            # save results
            if self.rank == 0:
                dict_args = OmegaConf.to_container(self.args, resolve=True)
                dict_result = create_dict_result(
                    dict_args,
                    self.world_size,
                    self.n_nodes,
                    torch.cuda.get_device_name(),
                    total_time,
                    self.id_run,
                    self.loss_static,
                )
                save_result(os.getcwd() + "/results.csv", dict_result)
                # save model
                path_checkpoint = os.getcwd() + "/checkpoints/"
                if not os.path.exists(path_checkpoint):
                    os.mkdir(path_checkpoint)
                path_save_model = path_checkpoint + self.id_run + "_model.pt"
                torch.save(self.model.state_dict(), path_save_model)
    
                
 



    def train_dpu(self):
        last_eval = 0
        t_begin = time.time()
        # performs n_warmup steps SEQUENTIALLY
        if self.args.n_warmup_steps > 0:
            self.count_grad_local = torch.zeros(1, device=self.local_rank, dtype=torch.int)
            self.warmup_steps(self.args.n_warmup_steps)
            grad_init = self.world_size*(self.args.n_warmup_steps + 1)*self.args.n_grad_accumulation
        else:
            self.count_grad_local = torch.ones(1, device=self.local_rank, dtype=torch.int)
            grad_init = 0
        # init the counts after warmup
        count_grad_tot = mp.Value('i', grad_init)
        count_com = self.args.n_warmup_steps
        while count_grad_tot.value < self.nb_grad_tot:
            
            # use gradient accumulation
            for step in range(self.args.n_grad_accumulation):
                # loads next batch in memory
                inputs = self.load_next_batch_into_static_memory()
                    # perform a grad step
                gradient_step(
                    self.dtype,
                    self.model,
                    self.count_grad_local,
                    inputs,
                    self.loss_static,
                    self.n_grad_acc_ddp,
                )
        
            # average stale gradients and perform an optimizer's step
            communication_step(
                self.rank,
                self.size_slice,
                self.size_local_slice,
                self.process_group,
                self.count_grad_this_round,
                self.com_buffer,
                self.params_opt,
                self.sharded_optimizer,
                self.scheduler,
            )
            # copy the buffer's params in the params tensor
            #self.params.copy_(self.com_buffer[:self.len_params])
            # update the current count grad tot
            count_grad_tot.value += self.count_grad_this_round[0]
            
            # updates the buffers
            update_buffers_step(
                self.params,
                self.com_buffer,
                self.len_params,
                self.count_grad_this_round,
                self.count_grad_local,
                count_com,
            )
            
            # update the com count and reinit the local grad one
            count_com += 1

            # logs
            if self.rank==0:
                # evalloop
                eval_loss = None
                if (self.args.eval == True) and (self.eval_dataset is not None):
                    if count_grad_tot.value - last_eval > self.args.eval_step:
                        eval_loss = self.eval_loop()
                        last_eval = count_grad_tot.value
        # logs the evolution of training
                delta_step_for_log=10
                log_to_tensorboard(
                    self.writer,
                    count_com,
                    count_grad_tot.value,
                    self.rank,
                    self.loss_static,
                    eval_loss,
                    self.t_beg,
                    delta_step_for_log,
                    self.epoch,
                )
                # print the evolution in the logs
                self.epoch, self.t_last_epoch = print_training_evolution(
                    self.log,
                    count_grad_tot.value,
                    count_com,
                    delta_step_for_log,
                    self.rank,
                    self.t_beg,
                    self.t_last_epoch,
                    self.loss_static,
                    self.epoch,
                )
                if self.args.save:
                    total_time_last_checkpoint = time.time() - time_checkpoint
                    if total_time_last_checkpoint >= 1800:  # 1800 seconds = 30 minutes
                        # Save the checkpoint
                        # Reset self.time_checkpoint to the current time for the next comparison
                        time_checkpoint = time.time()
                        path_checkpoint = os.getcwd() + "/checkpoints/"
                        if not os.path.exists(path_checkpoint):
                            os.mkdir(path_checkpoint)
                        path_save_model = path_checkpoint + self.id_run + "_dpu_model_"+str(count_grad_tot.value)+".pt"
                        torch.save(self.model.state_dict(), path_save_model)

        # LOGS for the end of training
        t_end = time.time()
        total_time = t_end - t_begin
        # save results
        if self.rank == 0:
            dict_result = create_dict_result(
                self.args,
                self.world_size,
                self.n_nodes,
                torch.cuda.get_device_name(),
                total_time,
                self.id_run,
                self.loss_static,
            )
            save_result(os.getcwd() + "/results.csv", dict_result)
            # save model
            path_checkpoint = os.getcwd() + "/checkpoints/"
            if not os.path.exists(path_checkpoint):
                os.mkdir(path_checkpoint)
            path_save_model = path_checkpoint + self.id_run + "dpu_model.pt"
            torch.save(self.model.state_dict(), path_save_model)

    def train_ddp(self):
        
        # create a data iterator
        time_checkpoint = time.time()
        self.train_iterator = iter(self.train_dataloader)
        self.count_grad_local = torch.zeros(1, device=self.local_rank, dtype=torch.int)
        self.n_grad_acc_ddp = self.args.n_grad_accumulation
        # init the count of batches
        last_eval = 0
        count_grad_tot = 0
        count_com = 0
        while count_grad_tot < self.nb_grad_tot:
            # use gradient accumulation
            for step in range(self.args.n_grad_accumulation):
                # loads next batch in memory
                inputs = self.load_next_batch_into_static_memory()
                # perform a grad step
                gradient_step(
                    self.dtype,
                    self.ddp_model,
                    self.count_grad_local,
                    inputs,
                    self.loss_static,
                    self.n_grad_acc_ddp,
                )
            # optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
            # increment the counts
            self.scheduler._step_count += self.world_size*self.args.n_grad_accumulation - 1
            count_grad_tot += self.world_size*self.args.n_grad_accumulation
            count_com += 1
            # logs
            if self.rank==0:
                # evalloop
                eval_loss = None
                if (self.args.eval == True) and (self.eval_dataset is not None):
                    if count_grad_tot - last_eval > self.args.eval_step:
                        eval_loss = self.eval_loop()
                        last_eval = count_grad_tot
                # logs the evolution of training
                delta_step_for_log=10
                log_to_tensorboard(
                    self.writer,
                    count_com,
                    count_grad_tot.value,
                    self.rank,
                    self.loss_static,
                    eval_loss,
                    self.t_beg,
                    delta_step_for_log,
                    self.epoch,
                )
                # print the evolution in the logs
                self.epoch, self.t_last_epoch = print_training_evolution(
                    self.log,
                    count_grad_tot,
                    count_com,
                    delta_step_for_log,
                    self.rank,
                    self.t_beg,
                    self.t_last_epoch,
                    self.loss_static,
                    self.epoch,
                )
                if self.args.save:
                    total_time_last_checkpoint = time.time() - time_checkpoint
                    if total_time_last_checkpoint >= 1800:  # 1800 seconds = 30 minutes
                        # Save the checkpoint
                        # Reset self.time_checkpoint to the current time for the next comparison
                        time_checkpoint = time.time()
                        path_checkpoint = os.getcwd() + "/checkpoints/"
                        if not os.path.exists(path_checkpoint):
                            os.mkdir(path_checkpoint)
                        path_save_model = path_checkpoint + self.id_run + "_ddp_model_"+str(count_grad_tot)+".pt"
                        #path_save_optim = path_checkpoint + self.id_run + "_ddp_optim_"+str(count_grad_tot)+".pt"
                        torch.save(self.model.state_dict(), path_save_model)
                        #torch.save(self.optimizer.state_dict(), path_save_optim)
        # LOGS for the end of training
        t_end = time.time()
        total_time = t_end - self.t_beg
        # save results
        if self.rank == 0:
            dict_args = OmegaConf.to_container(self.args, resolve=True)
            dict_result = create_dict_result(
                dict_args,
                self.world_size,
                self.n_nodes,
                torch.cuda.get_device_name(),
                total_time,
                self.id_run,
                self.loss_static,
            )
            
            save_result(os.getcwd() + "/results.csv", dict_result)
            # save model
            path_checkpoint = os.getcwd() + "/checkpoints/"
            if not os.path.exists(path_checkpoint):
                os.mkdir(path_checkpoint)
            path_save_model = path_checkpoint + self.id_run + "_ddp_model.pt"
            torch.save(self.model.state_dict(), path_save_model)


        