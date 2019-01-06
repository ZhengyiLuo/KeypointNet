KeypointNet
This is an implementation of the keypoint network proposed in "Discovery of Latent 3D Keypoints via End-to-end Geometric Reasoning [pdf]". Given a single 2D image of a known class, this network can predict a set of 3D keypoints that are consistent across viewing angles of the same object and across object instances. These keypoints and their detectors are discovered and learned automatically without keypoint location supervision [demo].

Datasets:
ShapeNet's rendering for Cars, Planes, Chairs.

Each set contains:

tfrecords
train.txt, a list of tfrecords used for training.
dev.txt, a list of tfrecords used for validation.
test.txt, a list of tfrecords used for testing.
projection.txt, storing the global 4x4 camera projection matrix.
job.txt, storing ShapeNet's object IDs in each tfrecord.
Training:
Run main.py --model_dir=MODEL_DIR --dset=DSET

Testing:
Run main.py --model_dir=MODEL_DIR --dset=DSET --test

where MODEL_DIR is a folder for storing model checkpoints: (see tf.estimator), and DSET should point to the folder containing tfrecords (download above).

Inference:
Run main.py --model_dir=MODEL_DIR --input=INPUT --predict


where MODEL_DIR is the model checkpoint folder, and INPUT is a folder containing png or jpeg test images. We trained the network using the total batch size of 256 (8 x 32 replicas). You may have to tune the learning rate if your batch size is different.

Code credit:
Supasorn Suwajanakorn

Contact:
supasorn@gmail.com, [snavely,tompson,mnorouzi]@google.com

(This is not an officially supported Google product)