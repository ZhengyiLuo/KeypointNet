import os
import subprocess
from subprocess import Popen, PIPE, STDOUT
from multiprocessing import Process
import multiprocessing as mp
import argparse
import sys


batch_size = 8


def setup_files(output_folder, script_path, job_filename, progress_filename, folder_id):
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
        if not os.path.isdir(os.path.join(output_folder,folder_id)):
            os.mkdir(os.path.join(output_folder,folder_id))

    f = open( script_path + job_filename, "r") 
    f_save = open( output_folder + progress_filename,"a+") 
    ids = []
    done_ids = []
    for s in f:
        ids.append(s.split(",")[1].strip())

    for s in f_save:
        done_ids.append(s.strip())
    f_save.close()

    
    return ids, done_ids

def generate_jobs(script_path, ids, done_ids, folder_id, dof):

    jobs = []
    for name in ids:
        input_model = os.path.join(shapnet_path, folder_id, name, "model.obj")
        if name not in done_ids and os.path.isfile(input_model):
            
            output_folder_path = os.path.join(output_folder, folder_id, name)
            if dof == "2":
                jobs.append(["blender", "-b", "--python", script_path + "render.py", "--", "-m", input_model, "-o", output_folder_path, "-s", "128", "-n", "100", "-fov", "5", "-roll", "-scale", "0.8"])
            elif dof == "3":
                jobs.append(["blender", "-b", "--python", script_path + "render_free.py", "--", "-m", input_model, "-o", output_folder_path, "-s", "128", "-n", "100", "-fov", "5", "-roll", "-scale", "0.8", "-light"])
            else:
                jobs = []
                parser.print_help()
                exit(1)
        elif not os.path.isfile(input_model):
            print("file do not exist: " + input_model) 
        elif name in done_ids:
            print("lmao")

    return jobs

def generate_pics(command):
    f_save = open( output_folder + progress_filename,"a") 
    output_folder_path = command[8]
    gen_id = command[8].split("/")[-1]
    if not os.path.isdir(os.path.join(output_folder_path)):
        os.mkdir(output_folder_path)
    # else:
        # os.system('rm -rf ' + output_folder_path)
    subprocess.call(command, stdin=PIPE, stdout=open(os.devnull, 'wb'), stderr=open(os.devnull, 'wb'))
    f_save.write(gen_id + "\n")
    print("Done, output: " + gen_id)
    f_save.close()



if __name__ == '__main__':
    shapnet_path = "/NAS/home/6dof/6dof_data/ShapeNetCore.v1"
    script_path = "/NAS/home/6dof/keypointnet/tools/"

    # shapnet_path = "/home/paperspace/zen/6dof/6dof_data/ShapeNetCore.v1"
    # render_path = "/home/paperspace/zen/6dof/models/research/keypointnet/tools/"



    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', dest='target',
                        required=True,
                        help='target set: car, plane, chair')

    parser.add_argument('-d', '--dof', dest='dof',
                        required=True,
                        help='degree of freedom')


    if '--' not in sys.argv:
        parser.print_help()
        exit(1)

    argv = sys.argv[sys.argv.index('--') + 1:]
    args, _ = parser.parse_known_args(argv)
    if args.target == 'car':
        output_folder = "/NAS/home/6dof/6dof_data/3dof/shapenet_car_light/"
        progress_filename = "progress_car_light.txt"
        job_filename = "job_car.txt"
        folder_id = "02958343"
    elif args.target == "plane":
        output_folder = "/NAS/home/6dof/6dof_data/3dof/shapenet_plane_light/"
        job_filename = "job_plane_light.txt"
        progress_filename = "progress_plane.txt"
        folder_id = "02691156" # plane
    elif args.target == "chair":
        output_folder = "/NAS/home/6dof/6dof_data/3dof/shapenet_chair_light/"
        progress_filename = "progress_chair_light.txt"
        job_filename = "job_chair.txt"
        folder_id = "03001627" # chair
    elif args.target == "sofa":
        output_folder = "/NAS/home/6dof/6dof_data/2dof/sofas_with_keypoints/"
        progress_filename = "progress_sofa.txt"
        job_filename = "job_sofa.txt"
        folder_id = "04256520" # sofa
    else:
        parser.print_help()
        exit(1)


    ids, done_ids = setup_files(output_folder, script_path, job_filename, progress_filename, folder_id)
    # print(len(ids))
    # print(len(done_ids))
    jobs = generate_jobs(script_path, ids, done_ids, folder_id, args.dof)
    print("Number of jobs: " + str(len(jobs)))

    pool = mp.Pool(processes=batch_size)
    pool.map(generate_pics, jobs)
