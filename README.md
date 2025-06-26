# Energy-Based Transformers

## Setup

- Make sure a conda env is being used. You can create it by doing `conda create -n pytorch_2.4 python=3.12` and then `pip install -r requirements.txt`
  - There is also `gh200_requirements.txt` and `loose_requirements.txt` which are `requirements.txt` for GH200s and without nvidia, pytorch, and triton packages respectively. These were helpful for setup on Nvidia's new GH200's.
  - You can also create a conda environment using the `environment.yml`
- Set the HF_HOME env variable so that your data/models cache to a desired directory (e.g. for me this is /[path_to_work_dir]/.cache)
  - Possibly set the HF_TOKEN env variable (may not need it)
- [Login](https://docs.wandb.ai/ref/cli/wandb-login) to wandb (basically just `pip install wandb` (if you didn't install requirements yet) and `wandb login` inside of that environment.)
- For the video (VID) dataset setup please see the README at [/data/vid/](/data/vid/README.md) for dataset installation and FFPROBE installation; similarly for VID inference setup please see the README at [/inference/vid/](/inference/vid/README.md)

 

## Running Code

- Start by running a job script. There are two ways to do this:
  - Running a bash script directly (recommended for interactive session or debugging) `bash job_scripts/nlp/pretrain/ebt_s1.sh`
  - Running a bash script using slurm executor (recommended on HPC) `bash slurm_executor.sh reference_a100 job_scripts/nlp/pretrain/ebt_s1.sh`
    - This method has a mandatory param (in this case `reference_a100`) which tells slurm_executor.sh how to build the slurm script. The available params are currently "reference_a100" (for your reference :). You could also just add a slurm header to these bash scripts and execute scripts using sbatch as is more standard, but this `slurm_executor.sh` is super helpful for keeping code modular
      - To add a HPC config type for slurm_executor.sh please see the reference script [job_scripts/slurm_headers/reference_a100.slurm](job_scripts/slurm_headers/reference_a100.slurm) and add the script name to [slurm_executor.sh](slurm_executor.sh)
      - If you need to override the default config in these scripts you can add the corresponding line in the .sh script and it will use the most recent value 
  - **The key params in these job scripts are, aside from the many hparams and gpus/cpus, the RUN_NAME, MODEL_NAME, and MODEL_SIZE**. Make sure to ctrl d (edit 3 things at once) when changing these to change the log names as well in addition to the RUN_NAME. The model name and size are both parsed from this so make sure to set them (with a hyphen) correctly! The model size *magically* automatically sets the numbers of layers, attention heads, embed dim, etc. :)
  - Also make sure you set the wandb run information properly (entity and project)
  - If you want to do multinode, you may need to set `ntasks = ngpus` (depends on slurm config it seems) and run the code using `srun python filename.py` instead (see the [job_scripts/nlp/pretrain/ebt_s1_mn.sh](job_scripts/nlp/pretrain/ebt_s1_mn.sh) file, very little multinode training was used for this paper, hence the lack of exploration of multinode code in the codebase). *You may also need to disable GPU binding in slurm headers (i.e. dont have `#SBATCH --gpu-bind=verbose,closest`)*, more on that [here](https://lightning.ai/docs/pytorch/stable/clouds/cluster_advanced.html).
  - There is also an example minimal training loop in `example_code/minimal_nlp_training_loop.py` that is easier to digest and has the minimum code for training language models with the Transformer++ vs EBTs. This however is not an exact implementation and won't produce observed results in the paper

### Pretraining Code
- The pretraining scripts are located under the specific modality desired (i.e. NLP, IMG) in the `job_scripts` dir. For example, the NLP pretraining script in [job_scripts/nlp/pretrain/ebt_s1.sh](job_scripts/nlp/pretrain/ebt_s1.sh) is the System 1 EBT used for many experiments in the paper.
  - There are several similar scripts for System 2 EBTs and for other modalities in the  `job_scripts` folder
  - If you are training your own EBT, I recommend starting with these hyperparameters and then tweaking them from there, as EBT hyperparameters are very sensitive as discussed in the paper.


### Inference Code
- The inference scripts are located under the specific modality desired and under the inference subdirectory. For example, the NLP inference script in [job_scripts/nlp/inference/ebt.sh](job_scripts/nlp/inference/ebt.sh) is useful for EBTs. The biggest difference between pretraining and inference is the use of a pretrained checkpoint with `--only_test_model_ckpt`, `--only_test` for telling the trained to not train and just test, as well as `--execution_mode "inference"` which controls the "mode" of the training loop. Everything else is relatively standard and can be used to do more complex inference procedures, such as EBT System 2 Thinking with self-verification (`infer_generated_samples` and `infer_ebt_advanced`)
  - If you are looking for more fine-grained control over these it's possible to use `--only_test` without using `--execution_mode "inference"` (for example if all you want to do is calculate the perplexity of language models). You can see more on this in the `test_step` of `base_model_trainer.py` as well as in `train_model.py`
  - Most of the other hparams in the script dont matter as they will be inherited from the trained ckpt (this logic is in `train_model.py`).
  - If using these make sure to fill in the `your/model/ckpt` as an actual .ckpt file


## General Code Flow

- The executed job script will send all the hparams to `train_model.py`. `train_model.py` will do things like set up distributed training, determine the number of GPUs, determine model config/size, and alter the behavior based off the specified hparams (note that I use the term hparams to vary widely refer to anything that controls the behavior of the script. i.e. some hparams could be `modality`, `wandb_project`, `log_filename`, etc.)
- `train_model.py` will call `base_model_trainer.py` (unless you are doing something such as debugging the dataloader or something miscellaneous). This is the pytorch lightning trainer, that is responsible for all training loop behavior, val behavior, testing, dataset setup, logging, creating the optimizer and lr scheduler, etc. Feel free to check it out.
  - The most important lines in base_model_trainer.py are the `eval_step` function as well as the instantiation of `self.model`.
  - This file is also where the datasets are created, particularly the `setup` function.
- After the instantiation of `self.model`, different models will be created. Some examples of these can be seen in `.model/nlp`. These are also pytorch lightning modules, that way lightning handles all the gpu calls, distributed things, etc.
- After this, `train_model.py` will do something such as `trainer.fit()` which will actually start training!
- Generally, you should need to change little in train_model.py and base_model_trainer.py, aside from maybe some new args, metrics, and adding datasets. The biggest changes usually need to be made when adding new models and their architectures
- If all you want is to get the model code and put it in your own training loop, just refer to the `model/` directory which has the model architectures, model forward/loss calculations, and other miscellaneous utils that models use.



## Tools/Software being used

- The repository uses [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/common/trainer.html)
  - I highly recommend reading and watching all the videos in the docs, they are AMAZING and that's coming from sone who usually hates docs.
  - PL is basically a lightweight wrapper over raw pytorch that helps with:
    - Distributed training
    - Provides simple to use callbacks (see `base_model_trainer.py`)
    - Annoying cuda (`.to('cuda')`) calls
    - Speeds some things up
    - Templatizing code base, all code bases will have a similar structure. 
    - Automates the training loop, no need to call `backwards` and `zero_grad`
- Using weights and biases (wandb) as the logger, if you haven't used this watch a tutorial online is a super simple but super amazing logger
  - The wandb logger has been configured to log A TON of info including: hparams (see info tab), console logs (see logs tab, which includes model params, hparams, size of model, model layers/arch, etc), checkpoints (super helpful to not manually dig through folders!), num gpus, etc
  - Also store logs to debug in `logs/console.log` if not using wandb (no_wandb)
- Bash and slurm scripts. Slurm scripts are basically bash scripts with an extra header on top specifying stuff about which hardware, allocation, and amount of time to use. 


## Debugging and Other Helpful Things

- Try to always download data using the dataloader_debugger (i.e. `bash job_scripts/debug/debug_dataloader.sh` so that data is not downloaded using multiple processes which can corrupt data)
- Not all hparams (args) are implemented for all cases so I highly recommend using ctrl f to your advantage and potentially checking if an hparam is implemented. This repo has evolved over time so some of the hparams were from testing over a year ago and have not been used since
- see the argparse params in `train_model.py` under the DEBUGGING header (as well as in other areas)
  - several of these can be used to increase speed of debugging as well as debug nans/dataloader issues/fitting issues/unused parameters (detect_anomaly, debug_dataloader, overfit_batches, debug_unused_parameters (which involves some effort in base_model_trainer.py))
  - limit_train_batches, limit_val_batches, limit_test_batches for example can be used to iterate over a smaller version of the dataset
  - several of these are supported out of the box with one line using the [lightning trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html) (once again, docs are amazing, highly recommend reading)
- abbreviations.md contains, you guessed it, several abbreviations used in code throughout the repo!
- as mentioned earlier, `logs/debug.log` and `logs/console.log` get populated with logs of what happens, so you can use these to debug what is happening/track print statements easily, etc
- `slurm_executor.sh` is extremely useful for saving the slurm scripts used/built. If you ever find that you are launching a lot of slurm scripts and want to save the actual script code used use it (helpful if runs keep failing and dont want to have to change params). With the "none" option this also serves to just save the bash script being used.
- If CPU ram is leaking over time, chances are its because either code saving the comp graph or because of a dataloader (i.e. an HF dataloader). Check if it occurs with a dummy dataloader, if it occurs with non multi GPU, etc. If is CPU RAM is probably data. If is GPU ram is probably tensors.

## Known Issues/TODOs (feel free to make a PR to fix!)
- not using KV cache for inference (not implemented). also the ebt inference is not optimized at all (you can reduce the ebt parallelization implementation, etc)
- there is no preprocessing support for CV stuff yet (i.e. pre-encoding)
- there is a small very minor bug with the learning rate of non model parameters (they will all decay to the same lr value of other parameters)
- let me know of any others and feel free to add issues on github!

