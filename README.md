# CX3 & Batch Job

Start Up

1. Connect to Zacalar / Imperial-WPA

2. SSH login to CX3
   - Method 1: PowerShell (not x86!!)
     - `ssh yl9422@login-b.cx3.hpc.ic.ac.uk` + pw
   - Method 2: VSCode local
     - Terminal (PowerShell) : `ssh yl9422@login-b.cx3.hpc.ic.ac.uk` + pw
     - JupyterHub Remote File Explorer
   - Method 3: VSCode remote
     - Remote-SSH : pw
     - Terminal (bash home) : `export PS1="[\W]\$ "`

3. Load env

   ```bash
   eval "$(~/miniforge3/bin/conda shell.bash hook)"
   conda activate M4R
   ```

4. Start new Jupyter Hub session [https://jupyter.cx3.rcs.ic.ac.uk](https://jupyter.cx3.rcs.ic.ac.uk/)

5. Run batch job on Computational Node (instead of Login Node)

   [1] Offline

   (1) PBS(Portable Batch System) script

   > 3 accessible directories:
   >
   > - `$HOME` : initial dir
   >
   > - `$PBS_O_WORKDIR` : job submission dir
   > - `$TMPDIR` : temp dir, created by node, on local disk of node, accessible by node only

   ```bash
   #!/bin/bash									# Interpreter = bash
   #PBS -l select=1:ncpus=1:mem=1gb			# LIST:nodes:cores:memory
   #PBS -l walltime=00:01:00					# LIST: max runtime
   #PBS -N hello_world							# JOB NAME
   #PBS -o /rds/general/user/yl9422/home/files	# OUTPUT PATH
   #PBS -e /rds/general/user/yl9422/home/files	# ERROR PATH
   eval "$(~/miniforge3/bin/conda shell.bash hook)"
   conda activate M4R
   source ~/.bashrc
   source /rds/general/user/yl9422/home/files/M4R-Elly/MyScripts/discord-notif/discord_notif.sh	# Discord Notif
   
   cd /rds/general/user/yl9422/home/files/M4R-Elly/MyScripts/first-test-job
   
   echo "hi from job echo"
   
   python test.py
   
   ```
   
   (2) Run in Terminal
   
   ```bash
   qsub py_script_job.sh	# submit job
   # returns unique job ID: `XXXXXX.pbs-7`
   
   qstat -u yl9422		# monitor user job
   # Status(S): Running(R), Queued(Q), Held(H), Error(E)
   
   qstat -Q [Queue Name]	# monitor queue
   
   qdel XXXXXX.pbs-7		# delete job
   ```
   
   (3) Job Output
   
   ​	Output file: `XXXXXXX.pbd-7.OU`
   
   ​	Error file: `XXXXXXX.pbd-7.ER`
   
   [2] Interactive session (for debugging)
   
   ```bash
   # import ipdb; ipdb.set_trace() in PyScript
   (M4R) [elly] $ qsub -I -l select=1:ncpus=2:mem=8gb -l walltime=02:00:00
   [yl9422@cx3-1-4 ~]$ qsub job_script.sh
   [yl9422@cx3-1-4 ~]$ exit
   (M4R) [elly]$ 
   ```

### Shell Commands

paths

```bash
$ pwd		# print working directory
$ cd		# go to home dir (or `cd ~`)
# /rds/general/user/yl9422/home
$ cd ..		# go to parent dir
$ cd ../..	# go to grandparent dir
$ cd -		# go to previous dir
```

### Package requitements

- Python
- ipykernel
- huggingface_hub[cli]
- requirements.txt in StatQA repo
- ...

### Resources Requests

| Scenario                        | Recommended Requests                |
| :------------------------------ | :---------------------------------- |
| Small Scripts / Data Processing | `select=1:ncpus=2:mem=8gb`          |
| Download Models                 | `select=1:ncpus=4:mem=32gb`         |
| API Model (e.g. GPT)            | `select=1:ncpus=4:mem=16gb`         |
| Local Model (e.g. Llama)        | `select=1:ncpus=8:mem=64gb:ngpus=1` |







# Reproducing Results

### Add New Models

##### I. Download Models

- [Hugging Face token](https://huggingface.co/settings/tokens)

- Model Folder: /rds/general/user/yl9422/home/files/models

| Model               | Open/Closed source | Desc  |
| ------------------- | ------------------ | ----- |
| LLaMA‑2 7B          | Open               | Paper |
| LLaMA‑2 13B         | Open               | Paper |
| LLaMA‑3 8B          | Open               | Paper |
| LLaMA‑3 8B Instruct | Open               | Paper |
| GPT‑3.5‑Turbo       | Closed             | Paper |
| GPT‑4               | Closed             | Paper |
| GPT‑4o              | Closed             | Paper |

1. edit file `MyScripts/download-models/download_models_job.sh`

2. run batch job

   ```bash
   cd /rds/general/user/yl9422/home/files/M4R-Elly/MyScripts/download-models
   qsub download_models_job.sh
   qstat -u yl9422
   ```

##### II. Edit Model Paths

1. find file `StatQA/Evaluation/llama_evaluation.py`

2. search for `model_path`

3. add / change to

   ```python
   elif model_type == '2_7b':
       model_path = "/rds/general/user/yl9422/home/files/models/Llama-2-7b-chat-hf"
       parallel_num = 1    # we only have 1 GPU
   ```

### Prompt Organisation 

(Only Once unless you change the prompt)

```bash
cd /rds/general/user/yl9422/home/files/M4R-Elly/StatQA/Script
sh prompt_organization.sh
```

> [!] An error occurred during prompt organization: [Errno 2] No such file or directory: 'Data/Integrated Dataset/Balanced Benchmark/Balanced Benchmark train.csv'
>
> The code is trying to rebuild the training set from raw data, but the raw data are missing. This step is time and token expensive, and the final processed training set is already provided in StatQA repo, so just use that instead.

OUTPUT: 5 files

- `StatQA/Data/Integrated Dataset/Dataset with Prompt/Test Set/` except for zero-shot-CoT

### Evaluation

1. edit file `StatQA/Script/llama_exp.sh`

2. run batch job (4hrs)

   ```bash
   qsub /rds/general/user/yl9422/home/files/M4R-Elly/MyScripts/reproduce-results/llama_exp_job.sh
   qstat -u yl9422
   ```

OUTPUT: 5 files per model

- `StatQA/Model Answer/Origin Answer/`

### Analysis

```bash
cd /rds/general/user/yl9422/home/files/M4R-Elly/StatQA/Script
sh answer_analysis.sh
```

OUTPUT:

- `StatQA/Model Answer/Processed Answer/`
- `StatQA/Model Answer/Processed Answer (for task confusion analysis)/`
- `StatQA/Model Answer/Task Performance/`
- modif on `StatQA/Model Answer/Original Answer/` except for stats-prompt ones
