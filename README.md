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

The data modes supported by AbBFN2 are detailed below.

##### Heavy-Chain IMGT Regions

| Field         | Type   | Region (IMGT)           | Description                                | Length Range (AA) |
|---------------|--------|-------------------------|--------------------------------------------|-------------------|
| `h_fwr1_seq`  | string | FWR1                    | Framework region 1                         | 18 – 41           |
| `h_fwr2_seq`  | string | FWR2                    | Framework region 2                         | 6 – 30            |
| `h_fwr3_seq`  | string | FWR3                    | Framework region 3                         | 29 – 58           |
| `h_fwr4_seq`  | string | FWR4                    | Framework region 4                         | 3 – 12            |
| `h_cdr1_seq`  | string | CDR1                    | Complementarity-determining region 1       | 1 – 22            |
| `h_cdr2_seq`  | string | CDR2                    | Complementarity-determining region 2       | 1 – 25            |
| `h_cdr3_seq`  | string | CDR3                    | Complementarity-determining region 3       | 2 – 58            |

##### Light-Chain IMGT Regions

| Field         | Type   | Region (IMGT)           | Description                                | Length Range (AA) |
|---------------|--------|-------------------------|--------------------------------------------|-------------------|
| `l_fwr1_seq`  | string | FWR1                    | Framework region 1                         | 18 – 36           |
| `l_fwr2_seq`  | string | FWR2                    | Framework region 2                         | 11 – 27           |
| `l_fwr3_seq`  | string | FWR3                    | Framework region 3                         | 25 – 48           |
| `l_fwr4_seq`  | string | FWR4                    | Framework region 4                         | 3 – 13            |
| `l_cdr1_seq`  | string | CDR1                    | Complementarity-determining region 1       | 1 – 20            |
| `l_cdr2_seq`  | string | CDR2                    | Complementarity-determining region 2       | 1 – 16            |
| `l_cdr3_seq`  | string | CDR3                    | Complementarity-determining region 3       | 1 – 27            |

##### CDR Length Metrics

Possible values provided in [src/abbfn2/data_mode_handler/oas_paired/constants.py](src/abbfn2/data_mode_handler/oas_paired/constants.py).


| Field       | Type | Description                     |
|-------------|------|---------------------------------|
| `h1_length` | int  | CDR1 length (heavy chain)       |
| `h2_length` | int  | CDR2 length (heavy chain)       |
| `h3_length` | int  | CDR3 length (heavy chain)       |
| `l1_length` | int  | CDR1 length (light chain)       |
| `l2_length` | int  | CDR2 length (light chain)       |
| `l3_length` | int  | CDR3 length (light chain)       |

##### Gene and Family Annotations

Possible values provided in [src/abbfn2/data_mode_handler/oas_paired/constants.py](src/abbfn2/data_mode_handler/oas_paired/constants.py).

| Field         | Type   | Description                        |
|---------------|--------|------------------------------------|
| `hv_gene`     | string | V gene segment (heavy)            |
| `hd_gene`     | string | D gene segment (heavy)            |
| `hj_gene`     | string | J gene segment (heavy)            |
| `lv_gene`     | string | V gene segment (light)            |
| `lj_gene`     | string | J gene segment (light)            |
| `hv_family`   | string | V gene family (heavy)             |
| `hd_family`   | string | D gene family (heavy)             |
| `hj_family`   | string | J gene family (heavy)             |
| `lv_family`   | string | V gene family (light)             |
| `lj_family`   | string | J gene family (light)             |
| `species`     | string | One of “human”, “rat”, “mouse”    |
| `light_locus` | string | One of “K” (kappa) or “L” (lambda)|

##### TAP Physicochemical Metrics

| Field              | Type   | Description                                 | Range           |
|--------------------|--------|---------------------------------------------|-----------------|
| `tap_psh`          | float  | Patch hydrophobicity                        | 72.0 – 300.0    |
| `tap_pnc`          | float  | Proportion of non-covalent contacts         | 0.0 – 10.0      |
| `tap_ppc`          | float  | Proportion of polar contacts                | 0.0 – 7.5       |
| `tap_sfvcsp`       | float  | Surface-exposed variable-chain charge score | –55.0 – 55.0    |
| `tap_psh_flag`     | string | Hydrophobicity flag                         | “red“ / “amber“ / “green“ |
| `tap_pnc_flag`     | string | Non-covalent contacts flag                  | “red“ / “amber“ / “green“ |
| `tap_ppc_flag`     | string | Polar contacts flag                         | “red“ / “amber“ / “green“ |
| `tap_sfvcsp_flag`  | string | Charge score flag                           | “red“ / “amber“ / “green“ |


##### V- and J- Identity Scores

| Field           | Type   | Description                       | Range (%)     |
|-----------------|--------|-----------------------------------|---------------|
| `h_v_identity`  | float  | Heavy-chain V segment identity    | 64.0 – 100.0  |
| `h_d_identity`  | float  | Heavy-chain D segment identity    | 74.0 – 100.0  |
| `h_j_identity`  | float  | Heavy-chain J segment identity    | 74.0 – 100.0  |
| `l_v_identity`  | float  | Light-chain V segment identity    | 66.0 – 100.0  |
| `l_j_identity`  | float  | Light-chain J segment identity    | 77.0 – 100.0  |

## Citation
If you use AbBFN2 in your research, please cite our work:
```
[Citation information to be added]
```
