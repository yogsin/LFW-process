# 使用说明
## 文件列表

### lfw_face_verification.py
用于在lfw上进行10折交叉验证

### lfw_pair_dir_extract.py
用于抽取lfw数据的pair列表，生成人脸验证所需的人脸对信息

Usage: python lfw_pair_dir_extract.py lfw_pair_list lfw_l_file lfw_r_file lfw_aligned_dir

参数说明：
lfw_pair_list：lfw官方提供的pair.txt人脸比对列表

lfw_l_file：pair.txt人脸比对列表每行第一个人脸路径+0

lfw_r_file：pair.txt人脸比对列表每行第二个人脸路径+0（1)(0表示与第一个人来自同一个人，1相反)

lfw_aligned_dir：lfw图像所在路径
