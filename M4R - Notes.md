## M4R - CX3 & Batch Job

Github repo: [https://github.com/kellywzhang/llm-statistics-m4r](https://github.com/kellywzhang/llm-statistics-m4r)

#### Start Up

1. Connect to Zacalar / Imperial-WPA
2. SSH login to CX3

   - Method 1: PowerShell (not x86)
     - `ssh yl9422@login-b.cx3.hpc.ic.ac.uk` + password
   - Method 2: VSCode local
     - Terminal (PowerShell) : `ssh yl9422@login-b.cx3.hpc.ic.ac.uk` + password
     - JupyterHub Remote File Explorer extension
   - Method 3: VSCode remote
     - Remote-SSH extension: password
     - Terminal (bash home) : `export PS1="[\W]\$ "`
3. Load env

   ```shell
   eval "$(~/miniforge3/bin/conda shell.bash hook)"
   conda activate M4R
   ```
4. Start new Jupyter Hub session [https://jupyter.cx3.rcs.ic.ac.uk](https://jupyter.cx3.rcs.ic.ac.uk/)
5. Run batch job

   (1) ==Create "job_script.sh"==

   > 3 accessible directories:
   >
   > - `$HOME` : initial dir
   > - `$PBS_O_WORKDIR` : job submission dir
   > - `$TMPDIR` : temp dir, created by node, on local disk of node, accessible by node only
   >

   ```shell
   #!/bin/bash									# Interpreter = bash
   #PBS -l select=1:ncpus=1:mem=1gb			# LIST:nodes:cores:memory
   #PBS -l walltime=00:01:00					# LIST: max runtime
   #PBS -N hello_world							# JOB NAME
   #PBS -o /rds/general/user/yl9422/home/files	# OUTPUT PATH
   #PBS -e /rds/general/user/yl9422/home/files	# ERROR PATH
   cd $PBS_O_WORKDIR							# Set working dir

   echo "hi from job echo"						# Actual Command

   eval "$(~/miniforge3/bin/conda shell.bash hook)" # Run PyScript
   conda activate M4R
   python py_script.py
   ```

   (2) Run in Terminal

   ```shell
   $ qsub job_script.sh	# submit job
   # returns unique job ID: `XXXXXX.pbs-7`

   $ qstat -u yl9422		# monitor job
   # Status(S): Running(R), Queued(Q), Held(H), Error(E)

   $ qdel XXXXXX.pbs-7		# delete job
   ```

   (3) Job Output

   - Output file: `XXXXXXX.pbd-7.OU`
   - Error file: `XXXXXXX.pbd-7.ER`

#### Shell Commands

paths

```shell
$ pwd		# print working directory
$ cd		# go to home dir (or `cd ~`)
# /rds/general/user/yl9422/home
$ cd ..		# go to parent dir
$ cd ../..	# go to grandparent dir
$ cd -		# go to previous dir
```

Interactive session for debugging

```shell
# import ipdb; ipdb.set_trace() in PyScript
(M4R) [elly] $ qsub -I -l select=1:ncpus=2:mem=8gb -l walltime=02:00:00
[yl9422@cx3-1-4 ~]$ qsub job_script.sh
[yl9422@cx3-1-4 ~]$ exit
(M4R) [elly]$ 
```

Interactive session for temporary GPU

```shell
(M4R) [elly] $ qsub -I -l select=1:ncpus=4:ngpus=2:mem=32gb -l walltime=02:00:00
[yl9422@cx3-20-6 ~]$ eval "$(~/miniforge3/bin/conda shell.bash hook)"
[yl9422@cx3-20-6 ~]$ conda activate M4R
(M4R) [yl9422@cx3-20-6 ~]$ cd /rds/general/user/yl9422/home/files/StatQA
(M4R) [yl9422@cx3-20-6 StatQA]$ python StatsQA/Evaluation/llama_evaluation.py --model_type "2_7b" --trick "zero-shot"
```

#### Hugging Face

```shell
hf auth whoami    #if you have logged in
hf auth login    # if you haven't logged in
hf download meta-llama/Llama-2-7b-chat-hf --local-dir llama-2-7b-chat-hf
```

#### Package requitements

- Python
- ipykernel
- ...
