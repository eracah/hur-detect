

script=gpu"$1"_run_yolo_verbose.sh
shift
echo $@
./$script --lambda_ae 0  $@
