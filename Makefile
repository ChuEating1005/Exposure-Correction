train_fuse:
	python train/train_fuse.py --display_iter=50 --snapshot_iter=50 --device=1 --load_pretrain=True

train_backword:
	python train/train_backword.py --display_iter=50 --snapshot_iter=50 --device=0 --load_pretrain=True

train_gan:
	python train/train_gan.py --display_iter=50 --snapshot_iter=50 --device=2 --load_pretrain=True

test_backword:
	python test/test_backword.py
	
test_fuse:
	python test/test_fuse.py

test_gan:
	python test/test_gan.py --test_dir data/test_data/Over --generator_path snapshots/weights_gan/gan_generator.pth