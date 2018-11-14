from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from scipy import misc
import tensorflow as tf
import argparse
import sys

# output_dir = "/home/paperspace/zen/6dof/6dof_data/zen_plane/"
# input_dir = "/home/paperspace/zen/6dof/models/research/keypointnet/tools/output_plane/02691156/"

def get_matrix(lines):
    return np.array([[float(y) for y in x.strip().split(" ")] for x in lines])


def read_model_view_matrices(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    return get_matrix(lines[:4]), get_matrix(lines[4:])


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def generate(files, output_dir, input_dir, gen_record, chunk_size = 40):
    file_chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]
    f_progress = open(gen_record, "r")
    done_records = []
    for i in f_progress:
        done_records.append(i.strip())
        
    print("done records: " + str(len(done_records)))
    f_progress.close()

    for i in range(len(file_chunks)):
        if not '{0:04}'.format(i) + ".tfrecord" in done_records:
            f_progress = open(gen_record, "a")
            files = file_chunks[i]
            record_name = output_dir + '{0:04}'.format(i) + ".tfrecord"

            with tf.python_io.TFRecordWriter(record_name) as tfrecord_writer:
                with tf.Graph().as_default():
                    im0 = tf.placeholder(dtype=tf.uint8)
                    im1 = tf.placeholder(dtype=tf.uint8)
                    encoded0 = tf.image.encode_png(im0)
                    encoded1 = tf.image.encode_png(im1)
                    with tf.Session() as sess:
                        for file_name in files:
                            count = 0
                            indir = input_dir + file_name  + "/"
                            while tf.gfile.Exists(indir + "%06d.txt" % count):
                                image0 = misc.imread(indir + "%06d.png" % (count * 2))
                                image1 = misc.imread(indir + "%06d.png" % (count * 2 + 1))

                                mat0, mat1 = read_model_view_matrices(
                                    indir + "%06d.txt" % count)

                                mati0 = np.linalg.inv(mat0).flatten()
                                mati1 = np.linalg.inv(mat1).flatten()
                                mat0 = mat0.flatten()
                                mat1 = mat1.flatten()

                                st0, st1 = sess.run([encoded0, encoded1],
                                                    feed_dict={im0: image0, im1: image1})

                                example = tf.train.Example(features=tf.train.Features(feature={
                                    'img0': bytes_feature(st0),
                                    'img1': bytes_feature(st1),
                                    'mv0': tf.train.Feature(
                                        float_list=tf.train.FloatList(value=mat0)),
                                    'mvi0': tf.train.Feature(
                                        float_list=tf.train.FloatList(value=mati0)),
                                    'mv1': tf.train.Feature(
                                        float_list=tf.train.FloatList(value=mat1)),
                                    'mvi1': tf.train.Feature(
                                        float_list=tf.train.FloatList(value=mati1)),
                                }))

                                tfrecord_writer.write(example.SerializeToString())
                                count += 1
                    f_progress.write('{0:04}'.format(i) + ".tfrecord\n")
                    print("finished: " + '{0:04}'.format(i) + ".tfrecord")
            f_progress.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', dest='target',
                        required=True,
                        help='target set: car, plane, chair')


    if '--' not in sys.argv:
        parser.print_help()
        exit(1)

    argv = sys.argv[sys.argv.index('--') + 1:]
    args, _ = parser.parse_known_args(argv)
    if args.target == 'car':
        output_dir = "/NAS/home/shapenet_rendering/shapenet_car/cars_with_keypoints/"
        input_dir = "/NAS/home/shapenet_rendering/shapenet_car/02958343/"
        gen_record = output_dir + "car_progress.txt"
    elif args.target == "plane":
        output_dir = "/NAS/home/shapenet_rendering/shapenet_plane/planes_with_keypoints/"
        input_dir = "/NAS/home/shapenet_rendering/shapenet_plane/02691156/"
        gen_record = output_dir + "plane_progress.txt"
    elif args.target == "chair":
        output_dir = "/NAS/home/shapenet_rendering/shapenet_chair/chairs_with_keypoints/"
        input_dir = "/NAS/home/shapenet_rendering/shapenet_chair/03001627/"
        gen_record = output_dir + "chair_progress.txt"
    else:
        parser.print_help()
        exit(1)
    
    files = sorted(os.listdir(input_dir))

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    generate(files, output_dir, input_dir, gen_record, 40)


    
