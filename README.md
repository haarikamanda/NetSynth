
This is the repository for the project, netSynth: Leveraging Foundation Models for Generating Synthetic Network Traffic.

Motivation:

Collecting data in the area of networks remains a chal-
lenging problem. Existing datasets are often prone to
underspecification and models trained on them fail to gen-
eralize. Additionally, manual data collection requires vast
resources, reinforcing research inequity, and generates
datasets that are unshareable. Given that networking re-
search is inherently tied to the presence of high quality
datasets, the present situation impedes progress for re-
searchers who don’t have access to the infrastructure and
resources for these endeavours. Existing methods, in this
space fail to successfully generate low level features at
the packet level while focusing on high level flow-level
features. Henceforth, we are proposing a methodology
that leverages foundation models to generate high quality
synthetic data that can be used to train machine learning
models for various downstream tasks. In this paper, we
propose netSynth, an auto-regressive model for gener-
ating synthetic data traffic. We utilize recent advances
in representation learning for networks and natural lan-
guage generation to design an encoder-decoder generative
model. We demonstrate the performance of this model
through statistical analysis of generated data. Futher-
more, we compare the fidelity of our generated data to the
ground truth collected data at the packet-level granularity. 

Steps for building the project:

1) Clone the repo and install the required dependencies. The packages are present in requirements.txt and can be installed in either a python virtual environment or an anaconda enivornment.

2) Ensure that you have access to an nvidia GPU and check that it's working with nvidia-smi.

3) Check that you have access to the required passively collected flows for either model training or evaluation. 

The file NetFoundModels.py contains the class netFoundDecoderModel.py which is essentially the implementation of our encoder-decoder model in pytorch. The following command can be used for the evaluation of our model and the hyperparameters can be changed according to the use case. In-case you want to re-train the model, the argument --do_train can be added in order to train the model using the existing weights specified in the --model_name_or_path.

python3 /path_to_you_dir/train/NetfoundPretraining.py --train_file <train_data.csv> --output_dir <output_dir> --eval_steps 10000 --learning_rate 5e-6 --do_eval --evaluation_strategy steps --save_steps 10000 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --overwrite_output_dir  --mlm_probability 0.15 --model_name_or_path /path_to_pre_trained_model --max_eval_samples 100000 --dataloader_num_workers 4 --dataloader_prefetch_factor 
