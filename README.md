# Combining neural networks and decision trees

This repo contains a demo for the following technical report ([arXiv](https://arxiv.org/abs/1702.08481)):

```
@misc
{
	nets-and-trees,
	author = {Nenad Marku\v{s} and Ivan Gogi\'c and Igor S. Pand\v{z}i\'c and J\"{o}rgen Ahlberg},
	title = {{Memory-Efficient Global Refinement of Decision-Tree Ensembles and its Application to Face Alignment}},
	year = {2017},
	eprint = {arXiv:1702.08481}
}
```

Basically, the report discusses some modifications of the method introduced by Ren et al. (CVPR2014), "Face Alignment at 3000 FPS via Regressing Local Binary Features".
However, instead to face alignment, these ideas are here applied to MNIST classification.
This approach can illustrate the basic idea of the report in a relatively few lines of code.

## Experiment 1

The code for this experiment can be found in the folder `basic/`.
Here we use the trees in the usual way:
the features for each tree are sampled from the whole 32x32 patch.

After compiling `tree.c` (use the provided `makefile`), learn the trees (256 trees of depth 12):

	th lrntrees.lua

Next, learn a neural network on top of the tree outputs:

	th lrnnn.lua --type rrr

You can try other architectures by specifying `--type nn` or `--type full`.

Note that all architectures easily fit the training data.
However, the generalization performance is not spectacular in these experiments (~3% test error).

## Experiment 2

The code for this experiment can be found in the folder `treecnv/`.
Here we apply the trees in a *convolutional* way.
This leads to *significant* gains in accuracy.

Compile the `treecnv` library:

	cc treecnv.c -o treecnv.so -I ~/torch/install/include -O3 -fopenmp -lm -fpic -shared

Next, run the demo:

	th lrnnn.lua

After several training epochs (30 or so), the test error should fall well below 1%.
This shows that exploiting spatial symmetry improves results (it serves as a strong regularizer?).

## License

MIT.

Also note that this repo contains code modified from [here](https://github.com/torch/demos/tree/master/train-a-digit-classifier).

## Contact

For any additional information contact me at <nenad.markus@fer.hr>.