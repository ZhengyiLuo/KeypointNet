import os

render_path = "/NAS/home/6dof/models/research/keypointnet/tools/"
output_path = "/NAS/home/shapenet_rendering/shapenet_chair/03001627/"

f_save = open( render_path + "progress_chair.txt","w+") 
progress_car = []

for i in f_save:
    progress_car.append(i.strip())

files = os.listdir(output_path)
keep = []

for i in files:
    in_files = os.listdir(output_path + i)
    if len(in_files) == 300:
        f_save.write(i + "\n")

f_save.close()