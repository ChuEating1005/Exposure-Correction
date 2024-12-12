train_fuse:
	python train/lowlight_train_fuse.py --display_iter=50 --snapshot_iter=50 --device=1 --load_pretrain=True

train_backword:
	python train/lowlight_train_backword.py --display_iter=50 --snapshot_iter=50 --device=0 --load_pretrain=True

train_gan:
	python train/lowlight_train_gan.py --display_iter=50 --snapshot_iter=50 --device=2 --load_pretrain=True

test_backword:
	python test/lowlight_test_backword.py
	
test_fuse:
	python test/lowlight_test_fuse.py

test_gan:
	python test/lowlight_test_gan.py --test_dir data/test_data/Over --generator_path snapshots/gan_discriminator_Epoch199.pth