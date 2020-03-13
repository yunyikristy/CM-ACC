#python main_pretrain.py --root_path ./data_kinetics --video_path video/train --annotation_path video/train.csv --result_path ./results --dataset kinetics_va --n_classes 400 --model va --model_depth 18 --resnet_shortcut A --batch_size 128 --n_threads 32 --checkpoint 5 --no_val --learning_rate=1e-2
python main_pretrain_as.py --root_path ./data --video_path UCF-101-frame --annotation_path ucfTrainTestlist/ucf101_01.json --result_path ./results --dataset ucf101_va --n_classes 101 --model va --model_depth 18 --resnet_shortcut A --batch_size 16 --n_threads 32 --checkpoint 5 --no_val --learning_rate=1e-2
