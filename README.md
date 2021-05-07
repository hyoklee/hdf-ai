# HDF-AI Tools and Information Center

This is a central hub for exchanging AI solutions in HDF5.
It also hosts sample AI data / model files in HDF5.

## Data
* [Cat vs. Non-Cat](https://www.floydhub.com/deeplearningai/datasets/cat-vs-noncat/1/train_catvnoncat.h5)
* [Core ML Specification](https://apple.github.io/coremltools/coremlspecification/)
* [Joke Generator](https://info.microsoft.com/ww-Thankyou-ADeepDiveintoServerlessApplications.html) (Jump to 14:40 in the video.)
  * [textgenrnn_weights.hdf5](https://github.com/minimaxir/textgenrnn/blob/master/textgenrnn/textgenrnn_weights.hdf5)

## Source Code
* [Deep Learning for Climiate Modeling Data](https://github.com/azrael417/ClimDeepLearn): specifically, [data_helpers.py](https://github.com/azrael417/ClimDeepLearn/blob/distributed/semanticsegm/utils/data_helpers.py)
* [Kubeflow HOW-TO](https://towardsdatascience.com/kubeflow-how-to-install-and-launch-kubeflow-on-your-local-machine-e0d7b4f7508f)

## HDF5 AI Filter

  Filtering Bigdata with AI is a solution to reduce the burden of managing a large amount of training and testing data.

  HDF5 AI filter can automatically sanitize your data in a scalable manner when you archive data in HDF5.
  It can save a lot of space by storing only models, not real data.
  
  HDF5 AI Filter can 
  * store everything in hierarchy including algorithms to use and learned models.
  * link to the raw data for provenance.
  * set a time to remove raw data and a desired accuracy threshold to prune models.
  * run several ML algorithms in parallel according to the HDF5's group hierarchy.

## Reference
* [Shrink floating point format to accelerate DNN training](https://www.hpcwire.com/2019/04/15/bsc-researchers-shrink-floating-point-formats-to-accelerate-deep-neural-network-training/) 
* [h5cpp](http://h5cpp.org/)
* [bfloat16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_forma)
* [Switch Transformer](https://arxiv.org/abs/2101.03961)
* https://docs.nersc.gov/machinelearning/benchmarks/
* https://semiengineering.com/the-best-ai-edge-inference-benchmark/
* [REMOTE PATHOLOGICAL GAIT CLASSIFICATION SYSTEM](https://arxiv.org/pdf/2105.01634.pdf) (@mfolk)
* [ai.gov](https://www.ai.gov/)
