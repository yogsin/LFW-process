#!/usr/bin/env python
# -*- coding:utf8 -*-
import os
import sys
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA   
pca=PCA(n_components=256)
# 设置当前的工作环境在caffe下  
caffe_root = '/home/work/Jimmy/caffe-face/'   
# 我们也把caffe/python也添加到当前环境  
sys.path.insert(0, caffe_root + 'python')  
import caffe  
os.chdir(caffe_root)#更换工作目录  
 
deploy='/home/work/Jimmy/caffe-face/face_example/face_deploy.prototxt'
caffe_model='/home/work/Jimmy/caffe-face/face_example/face_snapshot/face_cleaned_centerloss_iter_28000.caffemodel'  #自己训练的模型
#mean_file=root + 'face_example/centerloss_mean.npy'  #均值文件
lfw_l_file = '/home/work/Jimmy/caffe-face/face_example/data/lfw_l_info'    #待匹配子图1路径与标签文件
lfw_r_file = '/home/work/Jimmy/caffe-face/face_example/data/lfw_r_info'    #待匹配子图2路径与标签文件
lfw_src_file = '/home/work/DataSource/lfw/'    #LFW图像源文件
save_path = '/home/work/Jimmy/caffe-face/face_example/data/wrong_recogniced/'    #错误识别图像对保存路径

def calculate_threshold(distance,labels,num):    
    '''
    #计算识别率,
    选取阈值，计算识别率
    '''    
    accuracy = []
    best_acc = 0.0
    best_threshold = 0.05
    predict = np.empty((num,))
    threshold = 0.05
    while threshold <= 0.95 :
        for i in range(num):
            if distance[i] >= threshold:
                predict[i] =0
            else:
                predict[i] =1
        predict_right =0.0
        for i in range(num):
            if predict[i]==labels[i]:
                predict_right = 1.0+predict_right
        current_accuracy = (predict_right/num)
        if current_accuracy > best_acc:
            best_acc = current_accuracy
            best_threshold = threshold
        accuracy.append(current_accuracy)
        threshold=threshold+0.001
    print accuracy
    print 'train accuracy is: %f' %best_acc
    return best_threshold

def calculate_accuracy(distance,labels,num,threshold):    
    '''
    #计算识别率,
    根据阈值，计算识别率
    '''    
    predict = np.empty((num,))
    wrong_index = []
    for i in range(num):
        if distance[i] >= threshold:
            predict[i] = 0
        else:
            predict[i] = 1
    predict_right = 0.0
    for i in range(num):
        if predict[i]==labels[i]:
            predict_right = 1.0+predict_right
        else:
            wrong_index.append(i)
            print 'incorect index: %d' %i#打印误识别样本标签
            print 'simalarity: %f' %distance[i]#打印误识别样本标签
    accuracy = (predict_right/num)
    print 'test threshold is: %f' %threshold
    print 'test accuracy is: %f' %accuracy
    return wrong_index

def read_csv_file(csv_file):
    path_and_labels = []
    f = open(csv_file, 'rb')
    for line in f:
        line = line.strip('\r\n')
        path, label = line.split(' ')
        label = int(label)
        #path->path_and_labels[xxx][0]
        #label->path_and_labels[xxx][1]
        path_and_labels.append((path, label))
    f.close()
#    random.shuffle(path_and_labels)
    return path_and_labels

#左子图特征抽取
def batch_feature_l_generate(net, transformer, path_and_labels_l):
    i = 0
    for path,label in path_and_labels_l:
        iml=caffe.io.load_image(path)
        net.blobs['data'].data[i,...] = transformer.preprocess('data',iml)      #执行上面设置的图片预处理操作，并将图片载入到blob中
        i += 1
    #执行测试  
    #import datetime
    #starttime = datetime.datetime.now()
    #print starttime
    out = net.forward()  
    feature_l = np.float64(net.blobs['fc5'].data[0:300])
    feature_l=np.reshape(feature_l,(300,512))
    #np.savetxt("/home/chenningji/caffe-master/examples/deepid/deepidfeature_l.txt",feature_l)
    
    #获取镜像图像特征
    i = 0
    for path,label in path_and_labels_l:
        iml=np.fliplr(caffe.io.load_image(path))
        net.blobs['data'].data[i,...] = transformer.preprocess('data',iml)      #执行上面设置的图片预处理操作，并将图片载入到blob中
        i += 1
    out = net.forward()
    feature_l_mirror = np.float64(net.blobs['fc5'].data[0:300])
    #feature_l = np.float64(out['deepid'])
    feature_l_mirror = np.reshape(feature_l_mirror,(300,512))
    feature_l = np.column_stack((feature_l, feature_l_mirror))
    return feature_l

#右子图特征及标签抽取
def batch_feature_r_generate(net, transformer, path_and_labels_r):
    i = 0
    labels=np.empty((300,))
    for path,label in path_and_labels_r:
        imr=caffe.io.load_image(path)
        net.blobs['data'].data[i,...] = transformer.preprocess('data',imr)      #执行上面设置的图片预处理操作，并将图片载入到blob中
        labels[i]=int(label)    #获取类别标签 0->同一个人 1->不同的人
        # print 'label:'
        # print int(label)
        i += 1
    #print np.min(net.blobs['data'].data[0,...])
    #print np.max(net.blobs['data'].data[0,...])
    #print net.blobs['data'].data[0,...]

    #执行测试  
    #import datetime
    #starttime = datetime.datetime.now()
    #print starttime
    out = net.forward()  
    #starttime = datetime.datetime.now()
    #print starttime
    #labels = np.loadtxt(labels_filename, str, delimiter='\t')   #读取类别名称文件  
    #prob= net.blobs['prob'].data[0].flatten() #取出最后一层（prob）属于某个类别的概率值，并打印  
    #print prob  
    #order=prob.argsort()[9]  #将概率值排序，取出最大值所在的序号   
    #argsort()函数是从小到大排列  
    #print 'the class is:',labels[order]   #将该序号转换成对应的类别名称，并打印 
    feature_r = np.float64(net.blobs['fc5'].data[0:300])
    #feature_r = np.float64(out['deepid'])
    feature_r=np.reshape(feature_r,(300,512))
    
    #获取镜像图像特征
    i = 0
    for path,label in path_and_labels_r:
        imr=np.fliplr(caffe.io.load_image(path))
        net.blobs['data'].data[i,...] = transformer.preprocess('data',imr)      #执行上面设置的图片预处理操作，并将图片载入到blob中
        i += 1
    out = net.forward()
    feature_r_mirror = np.float64(net.blobs['fc5'].data[0:300])
    #feature_l = np.float64(out['deepid'])
    feature_r_mirror = np.reshape(feature_r_mirror,(300,512))
    feature_r = np.column_stack((feature_r, feature_r_mirror))
    #np.savetxt("/home/chenningji/caffe-master/examples/deepid/deepidfeature_r.txt",feature_r)
    return feature_r, labels

def deepid_generate():
    caffe.set_mode_gpu()
    caffe.set_device(2)
    net = caffe.Net(deploy,caffe_model,caffe.TEST)   #加载model和network  
    #图片预处理设置  
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})  #设定图片的shape格式(1,3,200,200)  
    transformer.set_transpose('data', (2,0,1))    #改变维度的顺序，由原始图片(200,200,3)变为(3,200,200)  
    #transformer.set_raw_scale('data', 255)    # 缩放到[-1, 1]之间
    transformer.set_mean('data', np.array((127.5, 127.5, 127.5))*0.0078125)#0.0039062)    #减去均值  
    #transformer.set_raw_scale('data', 0.0078125)    # 缩放到[-1, 1]之间  
    transformer.set_channel_swap('data', (2,1,0))   #交换通道，将图片由RGB变为BGR  
    #iml=caffe.io.load_image(imgl)                   #加载图片  

    #计算子图1特征
    #-------------------------
    #读取待匹配子图1路径及标签
    path_and_labels_l = read_csv_file(lfw_l_file)
    #将数据集分成28份批量计算特征再合并，其中每份样本数200
    for i in range(20):
        print 'processing batch %d ...' %i
        batch_feature_l = batch_feature_l_generate(net, transformer, path_and_labels_l[i*300:(i+1)*300])
        if i == 0:
            feature_l = batch_feature_l
        else:
            feature_l = np.row_stack((feature_l, batch_feature_l))
        print 'batch %d done' %i

    #计算子图2特征
    #-------------------------
    #读取待匹配子图2路径及标签
    path_and_labels_r = read_csv_file(lfw_r_file)
    #将数据集分成28份批量计算特征再合并，其中每份样本数200
    for i in range(20):
        print 'processing batch %d ...' %i
        batch_feature_r, batch_labels = batch_feature_r_generate(net, transformer, path_and_labels_r[i*300:(i+1)*300])
        if i == 0:
            feature_r = batch_feature_r
            labels = batch_labels
        else:
            feature_r = np.row_stack((feature_r, batch_feature_r))
            labels  = np.row_stack((labels, batch_labels))
        print 'batch %d done' %i
    labels=np.zeros((len(path_and_labels_r), ))
    i = 0
    for path,label in path_and_labels_r:
        labels[i] = int(label)
        i += 1
    total_wrong = 0
    for fold_n in range(10):
        test_index = range(fold_n * 600, (fold_n + 1) * 600)
        train_index = list(set(range(6000)).difference(set(test_index)))
        all_pre_feature  = np.row_stack((feature_l[train_index], feature_r[train_index]))
        all_pca_train_feature=pca.fit_transform(all_pre_feature)
        pca_train_l=all_pca_train_feature[0:5400]
        pca_train_r=all_pca_train_feature[5400:10800]
        #print labels[0:200]
        #print labels[200:400]
        #print labels[400:600]
        #np.cast['int32'](labels)
        #label_r = labels.astype(int)
        #计算相似度（余弦相似度）
        #-----------------------------------------------------------------------------------------------------------------------------------------
        feature_inner_product = np.sum(pca_train_r * pca_train_l, axis=1)    #子图1与子图2特征内积
        feature_cos_sim = feature_inner_product / (np.sqrt(np.sum(np.square(pca_train_r), axis=1)) * np.sqrt(np.sum(np.square(pca_train_l), axis=1)))    #计算余弦相似度
        feature_cos_sim_norm = (feature_cos_sim - np.min(feature_cos_sim))/(np.max(feature_cos_sim) -np.min(feature_cos_sim))    #归一化相似度
        print feature_cos_sim_norm.shape
        print feature_cos_sim_norm
        print np.max(feature_cos_sim)
        print np.min(feature_cos_sim)
        threshold = calculate_threshold(feature_cos_sim,labels[train_index],5400)    #获取最大准确率对应阈值

        all_test_feature  = np.row_stack((feature_l[test_index], feature_r[test_index]))
        all_pca_test_feature=pca.transform(all_test_feature)
        pca_test_l=all_pca_test_feature[0:600]
        pca_test_r=all_pca_test_feature[600:1200]
        #计算相似度（余弦相似度）
        #-----------------------------------------------------------------------------------------------------------------------------------------
        feature_inner_product = np.sum(pca_test_r * pca_test_l, axis=1)    #子图1与子图2特征内积
        feature_cos_sim = feature_inner_product / (np.sqrt(np.sum(np.square(pca_test_r), axis=1)) * np.sqrt(np.sum(np.square(pca_test_l), axis=1)))    #计算余弦相似度
        feature_cos_sim_norm = (feature_cos_sim - np.min(feature_cos_sim))/(np.max(feature_cos_sim) -np.min(feature_cos_sim))    #归一化相似度
        wrong_index = calculate_accuracy(feature_cos_sim,labels[test_index],600,threshold)
        total_wrong += len(wrong_index)
        #保存识别错误的图片对及对应的原图片
        for j in range(len(wrong_index)):
            j_img_l_path = path_and_labels_l[wrong_index[j]][0]#左图路径
            buf_path = j_img_l_path.split('/')
            j_img_l_src_path = lfw_src_file + buf_path[-2] +'/' + buf_path[-1]#左图原图片路径
            j_img_r_path = path_and_labels_r[wrong_index[j]][0]#右图路径
            buf_path = j_img_r_path.split('/')
            j_img_r_src_path = lfw_src_file + buf_path[-2] +'/' + buf_path[-1]#右图原图片路径
            im = Image.open(j_img_l_path)
            im.save(save_path + 'fold' + str(fold_n) + str(j) + '_L.jpg')
            im = Image.open(j_img_r_path)
            im.save(save_path + 'fold' + str(fold_n) + str(j) + '_R.jpg')
            im = Image.open(j_img_l_src_path)
            im.save(save_path + 'fold' + str(fold_n) + str(j) + '_L_src_' + j_img_l_path.split('/')[-1])
            im = Image.open(j_img_r_src_path)
            im.save(save_path + 'fold' + str(fold_n) + str(j) + '_R_src_' + j_img_r_path.split('/')[-1])
    #print np.load(mean_file).mean(1).mean(1)
    #计算相似度（欧式距离）
    #-----------------------------------------------------------------------------------------------------------------------------------------
    # feature_dif = feature_r - feature_l    #子图1与子图2特征做差
    # feature_L2 = np.sqrt(np.sum(feature_dif * feature_dif, axis=1))    #子图1与子图2特征欧式距离
    # feature_L2_norm = (feature_L2 - np.min(feature_L2))/(np.max(feature_L2) -np.min(feature_L2))    #归一化相似度
    # print feature_L2_norm
    # print np.max(feature_L2)
    # print np.min(feature_L2)
    # max_accuracy = calculate_accuracy(feature_L2_norm,labels,5600)    #获取最大准确率
    # print 'max accuracy is: %f' %max_accuracy
    print "{} pairs are wrong recognized".format(total_wrong)
    print 'mean accuracy is: {}'.format(1 - total_wrong/6000.0) 

if __name__ == '__main__':
    deepid_generate()
