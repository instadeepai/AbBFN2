# AbBFN2: A flexible antibody foundation model based on Bayesian Flow Networks
Welcome to the inference code of AbBFN2.

## Overview
\<Insert awesome description about BFNs and Antibodies >

## Getting Started
To get started, you need to modify Makefile to use your accelerator. For example, if you are using a GPU, you would modify the Makefile:

```
ACCELERATOR = GPU
```

You can choose between CPU, TPU and GPU accelerators. Please note that multi-host inference is not supported by this code release, and you should therefore restrict your hardware usage to single-host settings. Once you have configured your accelerator, simply run:

```
make build
```

to build the abbfn docker image. We typically find that this step takes 5-20 minutes to run, depending on your hardware.

## Experiments
AbBFN2 has three generation modes.

### Unconditional Generation
Set up the 'unconditional.yaml' config file with the desired number of samples per batch and number of batches. 
Set the number of sampling steps.

Run:
```
make unconditional
```

### Inpainting
Set up the 'inpaint.yaml' config file with the number of input samples and the number of samples per batch.
Set the number of sampling steps.
Set the values custom values for the antibody in cfg.input.dm_overwrites, and set the data modes to condition on cfg.sampling.mask_fn.data_modes

Run:
```
make inpaint
```

### Sequence Humanization
Set up the 'humanization.yaml' config file with the entire VL and VH sequences (l_seq and h_seq, respectively). 
Set the number of sampling steps.
Set the number of recycling steps.

Run:
```
make humanization
```

## Cite our work
