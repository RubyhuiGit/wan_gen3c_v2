# tools文件夹下面更新了4个文件

1、step0_unzip_waymo.py  主要是用于批量解压waymo数据集
2、step1_waymo_parse.py  是从waymo数据集里面，提取出前、左前、右前摄像头数据 + 标定的外参
3、step2_vipe_parse      读取waymo数据集，分别调用vipe获取前、左前、右前摄像头的数据进行重建，然后将标定的外参copy到当前目录下
4、metadata_tools        读取文件夹，生成想要的metadata.json文件


# 训练 （参考run.sh）
  多增加一个--use_waymo_datasets 表示切换到waymo数据集vipe的结果模式，参考run.sh
          --read_num_thread  视频读取线程数

# 测试
  运行test_gen3c.py，增加一个use_waymo_datasets切换的标志位，读取waymo vipe的数据开始test