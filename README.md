
This is the repository for the project, netSynth: Leveraging Foundation Models for Generating Synthetic Network Traffic.

The file NetFoundModels.py contains the class netFoundDecoderModel.py which is essentially the implementation of our encoder-decoder model in pytorch. The following command can be used for the evaluation of our model and the hyperparameters can be changed according to the use case. In-case you want to re-train the model, the argument --do_train can be added in order to train the model using the existing weights specified in the --model_name_or_path

python3 /path_to_you_dir/train/NetfoundPretraining.py --train_file <train_data.csv> --output_dir <output_dir> --eval_steps 10000 --learning_rate 5e-6 --do_eval --evaluation_strategy steps --save_steps 10000 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --overwrite_output_dir  --mlm_probability 0.15 --model_name_or_path /path_to_pre_trained_model --max_eval_samples 100000 --dataloader_num_workers 4 --dataloader_prefetch_factor 
