

script="gpu_run_yolo.sh $1"
shift
echo $@
./$script $@ 
