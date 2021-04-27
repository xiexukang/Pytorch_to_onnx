export PYTHONIOENCODING=UTF-8
export PYTHONPATH=../../:$PYTHONPATH
stage=$1

onnx_model_path=converted_onnx_models/
if [ ${stage} -le -1 ]; then
	# downloaded pretrained model and test data from baiduclude
	# filename: models_and_test_data_for_pytorch_to_onnx.zip
	# md5sum models_and_test_data_for_pytorch_to_onnx.zip
	# ffe4f44b8bdaa7637c144ed9f4c04e1b  models_and_test_data_for_pytorch_to_onnx.zip
	# Link: https://pan.baidu.com/s/1GeRRYw_BunREj1aM6fsEcg 
	# Code: x6hq
	# unzip models_and_test_data_for_pytorch_to_onnx.zip
	# then current directories should look like following:
	.
	├── cmd.sh
	├── conf
	│   ├── fbank.conf
	│   ├── pitch.conf
	│   ├── train_conformer.yaml
	│   └── train_transformer.yaml
	├── exp
	│   ├── dict
	│   │   └── lang_char.txt
	│   ├── final.pt
	│   └── train.yaml
	├── fbank_pitch
	│   ├── test
	│   │   ├── format.data
	│   │   ├── test_feat.ark
	│   │   └── test_feat.scp
	│   └── train_sp
	│       └── global_cmvn
	├── local
	│   ├── aishell_data_prep.sh
	│   └── download_and_untar.sh
	├── path.sh
	├── preconverted_onnx_models
	│   ├── dynamic_dim_decoder_init.onnx
	│   ├── dynamic_dim_decoder_non_init.onnx
	│   ├── dynamic_dim_encoder.onnx
	│   ├── fixed_dim_decoder_init.onnx
	│   ├── fixed_dim_decoder_non_init.onnx
	│   └── fixed_dim_encoder.onnx
	├── pytorch_to_onnx.sh
	├── README.md
	├── run.sh
	├── steps -> ../../../kaldi/egs/wsj/s5/steps
	├── tools -> ../../../tools
	├── utils -> ../../../kaldi/egs/wsj/s5/utils
	└── speech_model -> ../../../speech_model
fi

if [ ${stage} -le 1 ]; then
	if [ -d $onnx_model_path ]; then
		[ -d bk_$onnx_model_path ] && rm -r bk_$onnx_model_path
		mv $onnx_model_path bk_$onnx_model_path
	fi
	mkdir $onnx_model_path
	python speech_model/bin/export_onnx.py \
		--gpu 7 \
		--config exp/train.yaml \
		--test_data fbank_pitch/test/format.data \
		--checkpoint ./exp/final.pt \
		--beam_size 5 \
		--batch_size 1 \
		--penalty 0.0 \
		--dict exp/dict/lang_char.txt \
		--cmvn fbank_pitch/train_sp/global_cmvn \
		--onnx_model_path ${onnx_model_path}

	python speech_model/bin/change_dynamic_input_length.py \
		--onnx_model_path ${onnx_model_path}
fi
if [ ${stage} -le 2 ]; then
	if [ ! -d $onnx_model_path ]; then
		echo "run with stage=1 OR set onnx_model_path=preconverted_onnx_models" && exit 0
	fi
	python speech_model/bin/onnx_recognize.py \
		--gpu 7 \
		--config exp/train.yaml \
		--test_data fbank_pitch/test/format.data  \
		--beam_size 5 \
		--batch_size 1 \
		--dict exp/dict/lang_char.txt \
		--cmvn fbank_pitch/train_sp/global_cmvn \
		--onnx_encoder_path=${onnx_model_path}/dynamic_dim_encoder.onnx \
		--onnx_decoder_init_path=${onnx_model_path}/dynamic_dim_decoder_init.onnx \
		--onnx_decoder_non_init_path=${onnx_model_path}/dynamic_dim_decoder_non_init.onnx
	# expected output:
	# 2021-01-04 17:35:25,226 INFO 甚至出现交易几乎停止的情况
	# 2021-01-04 17:35:26,434 INFO 一二线城市虽然也处于调整中
	# 2021-01-04 17:35:27,559 INFO 但因为聚集了过多公共资源
	# 2021-01-04 17:35:29,156 INFO 为了规避三四线城市明显过剩的市场风险
	# 2021-01-04 17:35:30,278 INFO 标杆房企必然调整市场战略
	# 2021-01-04 17:35:31,247 INFO 因此土地储备至关重要
	# 2021-01-04 17:35:32,449 INFO 中原地产首席分析师张大伟说
	# 2021-01-04 17:35:33,493 INFO 一线城市土地供应量减少
	# 2021-01-04 17:35:34,550 INFO 也助推了土地市场的火爆
	# 2021-01-04 17:35:35,762 INFO 北京仅新增住宅土地供应十宗
fi

