# link your kitti data path
YOUR_KITTI_DATA_PATH=~/data/kitti_object
ln -s $YOUR_KITTI_DATA_PATH/training/ ./data/kitti/
ln -s $YOUR_KITTI_DATA_PATH/testing/ ./data/kitti/

# create kitti infos database
python -m liga.datasets.kitti.lidar_kitti_dataset create_kitti_infos
python -m liga.datasets.kitti.lidar_kitti_dataset create_gt_database_only
