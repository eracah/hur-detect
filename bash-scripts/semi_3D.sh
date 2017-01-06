

script="gpu_run_yolo_lowmem.sh $1"
shift
echo $@
./$script --3D True  $@ 
