

script="gpu_run_yolo_lowmem.sh $1"
shift
echo $@
./$script --lambda_ae 0 --3D True  $@
