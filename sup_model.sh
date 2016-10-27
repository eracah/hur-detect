

script=gpu"$1"_run_yolo.sh
shift
echo $@
./$script --lambda_ae 0  $@
