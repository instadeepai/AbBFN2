# AbBFN2: A flexible antibody foundation model based on Bayesian Flow Networks

Welcome to the inference code of AbBFN2, a state-of-the-art model for antibody sequence generation and labelling.

AbBFN2 allows for flexible task adaptation by virtue of its ability to condition the generative process on an arbitrary subset of variables. Further, since AbBFN2 is based on the Bayesian Flow Network paradigm, it can jointly model both discrete and continuous variables. Using this architecture, we provide a rich syntax which can be used to interact with the model. Regardless of conditioning information, the model generates all 45 "data modes" at inference time and arbitrary conditioning can be used to define specific tasks.

## Prerequisites
- Docker installed on your system
- Sufficient computational resources (TPU/GPU recommended)
- Basic understanding of antibody structure and sequence notation

## Installation

### Hardware Configuration
First, configure your accelerator in the Makefile:
```bash
ACCELERATOR = GPU  # Options: CPU, TPU, or GPU
```

Note: Multi-host inference is not supported in this release. Please use single-host settings only.

### Building the Docker Image
Run the following command to build the AbBFN2 Docker image:
```bash
make build
```
This process typically takes 5-20 minutes depending on your hardware.


### For Apple Silicon users
Build the conda environment instead directly using:
```bash
conda env create -f environment.yaml
conda activate abbfn2
```

## Usage

AbBFN2 supports three main generation modes, each with its own configuration file in the `experiments/configs/` directory.

In addition to the mode-specific settings, configuration files contain options for loading model weights. By default (`load_from_hf: true`), weights are downloaded from Hugging Face. Optionally, if you have the weights locally, set `load_from_hf: false` and provide the path in `model_weights_path` (e.g., `/app/params.pkl`).

### 1. Unconditional Generation
Generate novel antibody sequences without any constraints. AbBFN2 will generate natural-like antibody sequences matching its training distribution. Note that the metadata labels are also predictions made by the model. For a discussion of the accuracy of these labels, please refer to the AbBFN2 manuscript.

Configuration (`unconditional.yaml`):
```yaml
cfg:
  sampling:
    num_samples_per_batch: 10   # Number of sequences per batch
    num_batches: 1              # Number of batches to generate
  sample_fn:
    num_steps: 300              # Number of sampling steps (recommended: 300-1000)
```

Run:
```bash
make unconditional # or python experiments/unconditional.py for Apple Silicon users.
```

### 2. Conditional Generation/Inpainting
Generate antibody sequences conditioned on specific attributes. Conditional generation highlights the flexibility of AbBFN2 and allows it to be task adaptible depending on the exact conditioning data. While any arbitrary combination is possible, conditional generation is mostly to be used primarily when conditioning on full sequences (referred to as sequence labelling in the manuscript), partial sequences (sequence inpainting), partial sequences and metadata (sequence design), metadata only (conditional de novo generation). For categorical variables, the set of of possible values is found in `src/abbfn2/data_mode_handler/oas_paired/constants.py`. For genes and CDR lengths, only values that appear at least 100 times in the training data are valid. When conditioning on species, human, mouse, or rat can be chosen.

**Disclaimer**: _As discussed in the manuscript, the flexibility of AbBFN2 requires careful consideration of the exact combination of conditioning information for effective generation. For instance, conditioning on a kappa light chain locus V-gene together with a lambda locus J-gene family is unlikely to yield samples of high quality. Such paradoxical combinations can also exist in more subtle ways. Due to the space of possible conditioning information, we have only tested a small subset of such combinations._

Configuration (`inpaint.yaml`):
```yaml
cfg:
  input:
    num_input_samples: 2        # Number of input samples
    dm_overwrites:              # Specify values of the data modes
      h_cdr1_seq: GYTFTSHA
      h_cdr2_seq: ISPYRGDT
      h_cdr3_seq: ARDAGVPLDY
  sampling:
    inpaint_fn:
      num_steps: 300       # Number of sampling steps (recommended: 300-1000)
    mask_fn:
      data_modes:               # Specify which data modes to condition on
        - "h_cdr1_seq"
        - "h_cdr2_seq"
        - "h_cdr3_seq"
```

Run:
```bash
make inpaint # or python experiments/inpaint.py for Apple Silicon users.
```

### 3. Sequence Humanization
Convert non-human antibody sequences into humanized versions. This workflow is designed to run a sequence humanisation experiment given a paired, non-human starting sequence. AbBFN2 will be used to introduce mutations to the framework regions of the starting antibody, possibly using several recycling iterations. During sequence humanisation, appropriate human V-gene families to target will also be chosen, but can be manually set by the user too.

Briefly, the humanisation workflow here uses the conditional generation capabilities of AbBFN2 in a sample recycling approach. At each iteration, further mutations are introduced, using a more aggressive starting strategy that is likely to introduce a larger number of mutations. As the sequence becomes more human under the model, fewer mutations are introduced at subsequent steps. Please note that we have found that in most cases, humanisation is achieved within a single recycling iteration. If the model introduces a change to the CDR loops, which can happen in rare cases, these are removed. For a detailed description of the humanisation workflow, please refer to the AbBFN2 manuscript. 

Please also note that while we provide the option to manually select V-gene families here, this workflow allows the model to select more appropriate V-gene families during inference. Therefore, the final V-gene families may differ from the initially selected ones. Please also note that due to the data that AbBFN2 is trained on, humanisation will be most reliable when performed on murine or rat sequences. Sequences from other species have not been tested.

Configuration (`humanization.yaml`):
```yaml
cfg:
  input:
    l_seq: "DIVLTQSPASLAVSLGQRATISCKASQSVDYDGHSYMNWYQQKPGQPPKLLIYAASNLESGIPARFSGSGSGTDFTLNIHPVEEEDAATYYCQQSDENPLTFGTGTKLELK"
    h_seq: "QVQLQQSGPELVKPGALVKISCKASGYTFTSYDINWVKQRPGQGLEWIGWIYPGDGSIKYNEKFKGKATLTVDKSSSTAYMQVSSLTSENSAVYFCARRGEYGNYEGAMDYWGQGTTVTVSS"
    # h_vfams: null # Optionally, set target v-gene families
    # l_vfams: null
  sampling:
    recycling_steps: 10         # Number of recycling steps (recommended: 5-12)
    inpaint_fn:
      num_steps: 500            # Number of sampling steps (recommended: 300-1000)
```

Run:
```bash
make humanization # or python experiments/humanization.py Apple Silicon users.
```

## Data Modes

The data modes supported by AbBFN2 are:

```
# NB for sequence data modes, a wide range of unusual lengths are possible due to these lengths having been observed in the training data.
# During testing, we have limited our exploration to more common sequence lengths.
# All sequence regions are defined using the IMGT scheme.

"h_fwr1_seq" (string) # 20 amino acids, length between (18, 41), inclusive
"h_fwr2_seq" (string) # 20 amino acids, length between (6, 30), inclusive
"h_fwr3_seq" (string) # 20 amino acids, length between (29, 58), inclusive
"h_fwr4_seq" (string) # 20 amino acids, length between (3, 12), inclusive
"h_cdr1_seq" (string) # 20 amino acids, length between (1, 22), inclusive
"h_cdr2_seq" (string) # 20 amino acids, length between (1, 25), inclusive
"h_cdr3_seq" (string) # 20 amino acids, length between (2, 58), inclusive

"l_fwr1_seq" (string) # 20 amino acids, length between (18, 36), inclusive
"l_fwr2_seq" (string) # 20 amino acids, length between (11, 27), inclusive
"l_fwr3_seq" (string) # 20 amino acids, length between (25, 48), inclusive
"l_fwr4_seq" (string) # 20 amino acids, length between (3, 13), inclusive
"l_cdr1_seq" (string) # 20 amino acids, length between (1, 20), inclusive
"l_cdr2_seq" (string) # 20 amino acids, length between (1, 16), inclusive
"l_cdr3_seq" (string) # 20 amino acids, length between (1, 27), inclusive

# Possible values provided in src/abbfn2/data_mode_handler/oas_paired/constants.py
"h1_length" (int)
"h2_length" (int)
"h3_length" (int)
"l1_length" (int)
"l2_length" (int)
"l3_length" (int)

# Possible values provided in src/abbfn2/data_mode_handler/oas_paired/constants.py
"hv_gene"  (string)
"hd_gene"  (string)
"hj_gene"  (string)
"lv_gene"  (string)
"lj_gene"  (string)
"hv_family" (string)
"hd_family"  (string)
"hj_family"  (string)
"lv_family"  (string)
"lj_family"  (string)
"species" (string) # one of "human", "rat", "mouse"
"light_locus" (string) # one of "K", "L"
"tap_psh" (float) # [72.0, 300.0]
"tap_pnc" (float) # [0.0, 10.0]
"tap_ppc" (float) # [0.0, 7.5]
"tap_sfvcsp" (float) # [-55.0, 55.0]
"tap_psh_flag" (string) # "red", "amber", "green" 
"tap_pnc_flag" (string) # "red", "amber", "green" 
"tap_ppc_flag" (string) # "red", "amber", "green" 
"tap_sfvcsp_flag" (string) # "red", "amber", "green" 
"h_v_identity" (float) # [64.0, 100.0]
"h_d_identity" (float) # [74.0, 100.0]
"h_j_identity" (float) # [74.0, 100.0]
"l_v_identity" (float) # [66.0, 100.0]
"l_j_identity" (float) # [77.0, 100.0]
```

# Loading Weights




## Citation
If you use AbBFN2 in your research, please cite our work:
```
[Citation information to be added]
```
