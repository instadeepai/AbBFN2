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
Generate antibody sequences conditioned on specific CDR regions.

Configuration (`inpaint.yaml`):
```yaml
cfg:
  input:
    num_input_samples: 2        # Number of input samples
    dm_overwrites:              # Specify values of the data modes to condition on
      h_cdr1_seq: GYTFTSHA
      h_cdr2_seq: ISPYRGDT
      h_cdr3_seq: ARDAGVPLDY
  sampling:
    inpaint_fn:
      num_steps: 300-1000       # Number of sampling steps (recommended: 300-1000)
    mask_fn:
      data_modes:               # Specify which regions to condition on
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
    l_seq: "EVKLQQSGPGLVTPSQSLSITCTVSGFSLSDYGVHWVRQSPGQGLEWLGVIWAGGGTNYNSALMSRKSISKDNSKSQVFLKMNSLQADDTAVYYCARDKGYSYYYSMDYWGQGTSVTVSS"
    h_seq: "DIETLQSPASLAVSLGQRATISCRASESVEYYVTSLMQWYQQKPGQPPKLLIFAASNVESGVPARFSGSGSGTNFSLNIHPVDEDDVAMYFCQQSRKYVPYTFGGGTKLEIK"
  sampling:
    recycling_steps: 10         # Number of recycling steps
    inpaint_fn:
      num_steps: 500            # Number of sampling steps
```

Run:
```bash
make humanization
```

## Citation
If you use AbBFN2 in your research, please cite our work:
```
[Citation information to be added]
```

## License
[License information to be added]
