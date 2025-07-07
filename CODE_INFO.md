## Tools/Software being used

- The repository uses [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/common/trainer.html)
  - PL is basically a lightweight wrapper over raw pytorch that helps with:
    - Distributed training
    - Provides simple to use callbacks (see `base_model_trainer.py`)
    - Annoying cuda (`.to('cuda')`) calls
    - Speeds some things up
    - Templatizing code base, all code bases will have a similar structure. 
    - Automates the training loop, no need to call `backwards` and `zero_grad`
- Using weights and biases (WandB) as the logger
  - The wandb logger has been configured to log A TON of info including: hparams (see info tab), console logs (see logs tab, which includes model params, hparams, size of model, model layers/arch, etc), checkpoints (super helpful to not manually dig through folders!), the number of gpus, parameters/gradients, etc.
  - You can also store logs to debug in `logs/console.log` if you are not using wandb (`--no_wandb`)
- Bash and slurm scripts. Slurm scripts are basically bash scripts with an extra header on top specifying stuff about which hardware, allocation, and amount of time to use. 


## Debugging and Other Helpful Things

- There is a dataloader_debugger that you can invoke with the `--debug_dataloader` flag that calls `bash job_scripts/debug/debug_dataloader.sh`.
- Unfortunately, not all hparams (or arguments) are implemented for all cases so I highly recommend using ctrl f to your advantage and potentially checking if a new hparam is implemented. This repo has evolved a lot over time.
- For debugging quicklt see the argparse arguments in `train_model.py` under the DEBUGGING header
  - Several of these can be used to increase the speed of debugging as well as debug NANs/dataloader issues/fitting issues/unused parameters (detect_anomaly, debug_dataloader, overfit_batches, debug_unused_parameters)
  - For example: limit_train_batches, limit_val_batches, limit_test_batches can be used to iterate over a smaller version of the dataset
  - Several of these are supported out of the box using the [Pytorch Lightning Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html)
- [abbreviations.md](./abbreviations.md) contains, you guessed it, several abbreviations used in code throughout the repo!
- By default WandB captures the logs in the logs tab in the UI which can be useful.
- `slurm_executor.sh` is extremely useful for saving the slurm scripts used/built. If you ever find that you are launching a lot of slurm jobs and want to save the actual job script code I recommend using it.
- If CPU ram is leaking over time, chances are its because either code saving the comp graph or because of a dataloader (i.e. an HF dataloader). Check if it occurs with a dummy/synthetic dataloader, if it occurs with non multi-GPU, etc. If it is CPU RAM it is probably data; if it is GPU RAM is probably tensors.

## Known Minor Issues/TODOs

- We have not yet implemented the KV cache for inference. Also, the EBT inference is not optimized at all (you no longer need the ebt parallelized prediction implementation, etc.)
- There is no preprocessing support for anything in vision yet (i.e. pre-encoding).
- there is a small very minor bug with the learning rate of non model parameters (they will all decay to the same learning rate value of other parameters).
- Let me know of any other issues and feel free to add issues on github :)!