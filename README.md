# HDF-AI Tools and Information Center

This is a central hub for exchanging AI solutions in HDF.
It also hosts sample AI data / model files in HDF.

## Simplify Your AI Workflow and Data Management

HDF-AI is an ideal solution for managing all your AI data -
from raw training data to model data.

### Query HDF in Natural Langauge / Image / Sound / Video

  * Upload HDF file(s) as source for LLM.
  * Query HDF contents in Natural Langauge.
  * Manipulate data. (e.g., Create a PNG image from `/g1/dset1`)
  * Find a similar dataset given the image, sound, and video.

### Make Bigdata AI-ready using HDF

 * Save metadata with data.
   * algorithms and their versions used, model parameters, authors, etc.  
 * Save training / model / testing data in hiearchy with groups.
 * Save knowledge graph (semantic network) in HDF.

### Reduce Bigdata Storage using HDF AI Filter

Filtering Bigdata with AI is a solution to reduce the burden of managing
a large amount of training/testing & model data.

  HDF AI filter can automatically sanitize your data in a scalable manner
  when you archive data in HDF.
  It can save a lot of space by storing only models, not real data.
  
  HDF AI Filter can 
  * store everything in hierarchy including algorithms to use and
  learned models.
  * link to the raw data for provenance.
  * set a time to remove raw data and a desired accuracy threshold to
  prune models.
  * run several ML algorithms in parallel according to HDF group hierarchy.

## FAQ

  * What is HAI API? This is a high level API that can run I/O-efficient \
AI tasks for HDF data.
  
## User Guide

  * HAI Reference Manual

## Benchmark
* [CyBench](https://cybench.github.io)
* [PHYBench](https://arxiv.org/abs/2504.16074)

## Data
* [ARC-AGI](https://github.com/fchollet/ARC-AGI/)
* [Cat vs. Non-Cat](https://www.floydhub.com/deeplearningai/datasets/cat-vs-noncat/1/train_catvnoncat.h5)
* [Core ML Specification](https://apple.github.io/coremltools/coremlspecification/)
* [Hypersim](https://github.com/apple/ml-hypersim)
* [Megatron-LM](https://github.com/NVIDIA/Megatron-LM/tree/main/tools/retro)
* [Joke Generator](https://info.microsoft.com/ww-Thankyou-ADeepDiveintoServerlessApplications.html) (Jump to 14:40 in the video.)
  * [textgenrnn_weights.hdf5](https://github.com/minimaxir/textgenrnn/blob/master/textgenrnn/textgenrnn_weights.hdf5)
* [Face Emotion Recognition](https://analyticsindiamag.com/face-emotion-recognizer-in-6-lines-of-code/)
* [A Deep Learning-Based Hybrid Model of Global Terrestrial Evaporation](https://zenodo.org/record/5220753#.YeC2bf7MLIU)
* [JFT-3B](https://paperswithcode.com/dataset/jft-3b)

## Code
* [LLM Symbolic Regression](https://github.com/deep-symbolic-mathematics/LLM-SR)
* [Spider 2 SQL](https://spider2-sql.github.io/)
* [TA-STVG](https://github.com/HengLan/TA-STVG)
* [AFlow](https://github.com/FoundationAgents/AFlow)
* [cuDNN](https://docs.nvidia.com/cudnn/index.html)
* [NotaGen](https://github.com/ElectricAlexis/NotaGen)
* [anything-llm](https://github.com/Mintplex-Labs/anything-llm)
* [MILVUS](https://github.com/milvus-io/milvus)
* [vLLM](https://github.com/vllm-project/)
* [Together](https://github.com/togethercomputer)
* [hle](https://github.com/centerforaisafety/hle)
* [DeepSeek](https://github.com/deepseek-ai)
* [llama-gguf-optimize](https://github.com/robbiemu/llama-gguf-optimize)
* [AnnData](https://anndata.readthedocs.io/en/latest/fileformat-prose.html)
* [PIMFlow](https://github.com/yongwonshin/PIMFlow)
* [John Snow Labs](https://github.com/JohnSnowLabs/spark-nlp/blob/47bd96b60cb4790772f0b009ef48c4b44aeb5ae9/python/tensorflow/sddl/arguments.py#L50)
* [GeoWatch](https://gitlab.kitware.com/computer-vision/geowatch)
* [safetensors](https://github.com/huggingface/safetensors)
* [Croissant](https://github.com/mlcommons/croissant)
* [GraphCast](https://github.com/google-deepmind/graphcast)
* [Deep Learning for Climiate Modeling Data](https://github.com/azrael417/ClimDeepLearn): specifically, [data_helpers.py](https://github.com/azrael417/ClimDeepLearn/blob/distributed/semanticsegm/utils/data_helpers.py)
* [Kubeflow HOW-TO](https://towardsdatascience.com/kubeflow-how-to-install-and-launch-kubeflow-on-your-local-machine-e0d7b4f7508f)
* [Keras Spark Rossmann Run Example](https://github.com/horovod/horovod/blob/master/examples/spark/keras/keras_spark_rossmann_run.py)
* [Deep Learning IO Benchmark](https://github.com/hariharan-devarajan/dlio_benchmark)
* [FlexFlow](https://github.com/flexflow/FlexFlow)
* [parallelformers](https://github.com/tunib-ai/parallelformers)
* [Keras/TensorFlow - Save Model in HDF5](https://www.tensorflow.org/guide/keras/save_and_serialize)
* [flowEQ](https://github.com/csteinmetz1/flowEQ) = MATLAB + Python (Keras)


## Reference
* [AI Timeline](https://nhlocal.github.io/AiTimeline/)
* [A Survey of AI Agent Protocols](https://arxiv.org/pdf/2504.16736)
* [PowerShell AIShell](https://github.com/PowerShell/AIShell)
* [Model Leaderboard](https://scale.com/leaderboard)
* [AI Alliance](https://thealliance.ai/)
* [Ai2](https://allenai.org/)
* [AI-driven Science on Supercomputer](https://www.youtube.com/playlist?list=PLcbxjEfgjpO8Dy4bFRtnIknfXkTNNzAuL)
* [Titans](https://arxiv.org/pdf/2501.00663v1)
* [GraphCast](https://deepmind.google/discover/blog/graphcast-ai-model-for-faster-and-more-accurate-global-weather-forecasting/)
* [Shrink floating point format to accelerate DNN training](https://www.hpcwire.com/2019/04/15/bsc-researchers-shrink-floating-point-formats-to-accelerate-deep-neural-network-training/) 
* [h5cpp](http://h5cpp.org/)
* [bfloat16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_forma)
* [Switch Transformer](https://arxiv.org/abs/2101.03961)
* https://docs.nersc.gov/machinelearning/benchmarks/
* https://analyticsindiamag.com/face-emotion-recognizer-in-6-lines-of-code/
* https://semiengineering.com/the-best-ai-edge-inference-benchmark/
* [REMOTE PATHOLOGICAL GAIT CLASSIFICATION SYSTEM](https://arxiv.org/pdf/2105.01634.pdf) (@mfolk)
* [ai.gov](https://www.ai.gov/)
* [SpaceML](https://earthdata.nasa.gov/learn/articles/spaceml)
* [Mathematics for Machine Learning](https://mml-book.github.io/)
* [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html)
* [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/)
* [Applied ML](https://github.com/eugeneyan/applied-ml)
* [A Review of Earth AI](https://www.sciencedirect.com/science/article/pii/S0098300422000036)
* [AI Builder in Power Platform](https://docs.microsoft.com/en-us/ai-builder/)
* [Intel oneAPI AI Analytics Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html)
* [Open Catalyst 2020 (OC20) Dataset](https://github.com/Open-Catalyst-Project/ocp/blob/main/DATASET.md) in LMDB format for Caffe
* [SambaNova AI](https://sambanova.ai/)
* [Horovod](https://github.com/horovod/horovod)
* [DeepHyper](https://deephyper.readthedocs.io/en/latest/)
* [sits](https://e-sensing.github.io/sitsbook/index.html)
* [Drake](https://drake.mit.edu)
* [ZSTD in training mode](http://facebook.github.io/zstd/#small-data)
