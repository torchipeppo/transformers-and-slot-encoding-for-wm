# Transformers and slot encoding for sample efficient physical world modeling

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Configuration

All configuration and hyperparameters are handled through [hydra](https://hydra.cc/docs/intro/) configuration files, found in the `config` directory for human-friendly reading. To change the configuration, either change the config files or provide overrides through the command line.

The experiments in the paper were run with the configuration found in these files.

## Training and evaluation

### 0. Preliminary

Download the PHYRE video dataset by following these instructions: https://github.com/HaozhiQi/RPIN/blob/master/docs/PHYRE.md#11-download-our-dataset

Notes:
- The zip file is ~15G and the whole dataset is ~100G
- Do not change the directory structure

Then, find the file in `config/path_constants/TEMPLATE.yaml` and rename it to `path_constants.yaml`. This will be the way to provide data paths.

### 1. Train tokenizer

Fill in the following fields in `path_constants.yaml`:
- `logs_dir`: the base directory where Tensorboard will write its logs (a unique subdirectory for each run is automatically created inside this)
- `phyre_video_dataset`: one directory or a list of directories containing videos from the PHYRE dataset, and nothing else

Then train the tokenizer with this command:

```train
python train_tokenizer.py
```

The trained tokenizer will be found in the newly-created `outputs` directory. Find the subdirectories corresponding to the date and time the command was launched at. Use the file ending in `last.pt`.

### 2. Train world model

Fill in the following field in `path_constants.yaml` *(in addition to the previous ones)*:
- `phyre_pretrained_tokenizer`: path to the pretrained tokenizer produced in the previous step

Then train the FPTT world model with this command:

```train
python train_three_transformers.py
```

As before, the trained model will be found in the `outputs` directory.

### 3. Perform task success classification task

Fill in the following field in `path_constants.yaml` *(in addition to the previous ones)*:
- `phyre_pretrained_3transf`: path to the trained FPTT model produced in the previous step

Then run this command:

```eval
python eval_3transf_worldmodeling.py
```

Note the location of the corresponding Tensorboard log, it will be needed for "Producing results" (see section below).

## Baselines

For the STEVE baseline, follow steps 2 and 3 with the following scripts and `path_constants` field instead:
- `train_steve.py`
- `phyre_pretrained_steve`
- `eval_steve_worldmodeling.py`

For the decoder-only baseline, follow steps 2 and 3 with the following scripts and `path_constants` field instead:
- `train_decoder_only.py`
- `phyre_pretrained_decodonly`
- `eval_decodonly_worldmodeling.py`

## Producing results

### A. Preliminary steps
1. **Move to the `analysis` directory.**
2. Use the `downloading/download-tensorboard-data.py` script to generate CSV files corresponding to the Tensorboard logs of interest
3. Edit the `RUN_BY_ARCHS` dictionary in `common.py` to point at your CSV locations

### B1. Plots
To produce figures equivalent to Figure 5 in the paper: 
1. Run the `make_plots.py` script
2. Find the plots in `../plots`

### B2. Sample efficiency
To produce quantitative results equivalent to Table 1 in the paper:
1. Run the script `for-the-sample-efficiency.py`. This computation has hyperparameters which can be set directly from the last line of the script.
2. The results are printed out directly to the terminal. `mean` is the average result, `sem` is the standard error of the mean, `len` is the number of suns that cleared the threshold.
