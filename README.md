# AbBFN2: A flexible antibody foundation model based on Bayesian Flow Networks
Welcome to the inference code of AbBFN2.

## Overview
\<Insert cool description about Antibodies and BFNs>

## Getting Started - (Copied from protein-sequence-bfn)
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
```
make unconditional
```

### Conditional Genearion
```
make inpaint
```

### Sequence Humanization
```
make humanization
```


## Cite our work
