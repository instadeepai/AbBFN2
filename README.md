# AbBFN2: A flexible antibody foundation model based on Bayesian Flow Networks

Welcome to the inference code of AbBFN2, a state-of-the-art model for antibody sequence generation.

## Overview
[Overview to be added]

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

## Usage

AbBFN2 supports three main generation modes, each with its own configuration file in the `experiments/configs/` directory.

### 1. Unconditional Generation
Generate novel antibody sequences without any constraints.

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
make unconditional
```

### 2. Inpainting
Generate antibody sequences conditioned on specific genetic attributes.

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
make inpaint
```

### 3. Sequence Humanization
Convert non-human antibody sequences into humanized versions.

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
make humanization
```

## Data Modes

The data modes supported by AbBFN2 are:

```
"h_fwr1_seq" (string)
"h_fwr2_seq" (string)
"h_fwr3_seq" (string)
"h_fwr4_seq" (string)
"h_cdr1_seq" (string)
"h_cdr2_seq" (string)
"h_cdr3_seq" (string)

"l_fwr1_seq" (string)
"l_fwr2_seq" (string)
"l_fwr3_seq" (string)
"l_fwr4_seq" (string)
"l_cdr1_seq" (string)
"l_cdr2_seq" (string)
"l_cdr3_seq" (string)

"h1_length" (int)
"h2_length" (int)
"h3_length" (int)
"l1_length" (int)
"l2_length" (int)
"l3_length" (int)

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
"species" (string)
"light_locus" (string)
"tap_psh" (float)
"tap_pnc" (float)
"tap_ppc" (float)
"tap_sfvcsp" (float)
"tap_psh_flag" (string)
"tap_pnc_flag" (string)
"tap_ppc_flag" (string)
"tap_sfvcsp_flag" (string)
"h_v_identity" (float)
"h_d_identity" (float)
"h_j_identity" (float)
"l_v_identity" (float)
"l_j_identity" (float)
```


## Citation
If you use AbBFN2 in your research, please cite our work:
```
[Citation information to be added]
```

## License
[License information to be added]
