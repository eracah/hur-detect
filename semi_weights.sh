

script=gpu"$1"_run_yolo.sh
shift
echo $@
./$script --ae_load_path /storeSSD/cbeckham/nersc/models/output/full_image_1/12.model $@
