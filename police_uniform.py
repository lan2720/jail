#coding:utf-8
import os
import h5py
import glob
import numpy as np

def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a 
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    return cos


def generate_feature():
    test_dir = '/data1/Project/Jail/警服test'
    images = glob.glob(os.path.join(test_dir, '*.png'))
    for image_path in images:
        os.system('python /data1/Project/TF_FeatureExtraction/example_feat_extract.py --network resnet_v2_101 --checkpoint /data1/Project/TF_FeatureExtraction/checkpoints/resnet_v2_101.ckpt --image_path %s --out_file %s --layer_names resnet_v2_101/logits --preproc_func inception --batch_size=1' % (image_path, image_path.split('/')[-1].split('.')[0]+'.h5'))


def main():
    with h5py.File('/data1/Project/TF_FeatureExtraction/features.h5','r') as f:
        criterion = f['resnet_v2_101']['logits'].value.squeeze(axis=1).squeeze(axis=1)
    with h5py.File('./10.h5','r') as f:
        features = f['resnet_v2_101']['logits'].value.squeeze(axis=1).squeeze(axis=1)
    
    #for i in range(features.shape[0]):
    track = []
    for j in range(criterion.shape[0]):
        val = cos_sim(features, criterion[j])
        track.append(val)
    print('max = %f' % max(track))


if __name__ == '__main__':
    main()
    #generate_feature()
