# coding:utf-8
# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import numpy as np
import h5py
import shutil
import time

sys.path.append('/data1/Softwares/openpose/build/python/openpose');
sys.path.append('/data1/Project/TF_FeatureExtraction')

import openpose as op
from feature_extractor.feature_extractor import FeatureExtractor
from example_feat_extract import feature_extraction_queue

params = dict()
params["logging_level"] = 3
params["output_resolution"] = "-1x-1"
params["net_resolution"] = "-1x368"
params["model_pose"] = "BODY_25"
params["alpha_pose"] = 0.6
params["scale_gap"] = 0.3
params["scale_number"] = 1
params["render_threshold"] = 0.5
# If GPU version is built, and multiple GPUs are available, set the ID here
params["num_gpu_start"] = 0
params["disable_blending"] = False
# Ensure you point to the correct path where models are located
params["default_model_folder"] = "/data1/Softwares/openpose/models/"
# Construct OpenPose object allocates GPU memory
openpose = op.OpenPose(params)

def body_parts_box(keypoints, box_size=20):
    boxes = []
    for k in keypoints[[2,5]]:
        if k[2] == 0.0:
            boxes.append([])
        else:
            leftup = (int(k[0]-box_size/2.0), int(k[1]-box_size/2.0))
            bottomdown = (int(k[0]+box_size/2.0), int(k[1]+box_size/2.0))
            boxes.append([leftup, bottomdown])
    return boxes


def upper_body_box(keypoints, indices=[0,1,2,3,4,5,6,7,8,9,12]):
    upper_keypoints = keypoints[indices][:,:2].astype(int)
    if np.any(upper_keypoints == 0.0):
        return None
    x,y,w,h = cv2.boundingRect(upper_keypoints)
    return [(x,y), (x+w,y+h)]


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


def cos_sim_batch(vector_a, matrix_b):
    """
    计算向量和矩阵之间的余弦相似度，最后取其中的最大值
    :param vector_a: 向量 a [d,]
    :param matrix_b: 矩阵 b [3,d]
    :return: sim
    """
    vector_a = np.mat(vector_a) # [1,d]
    matrix_b = np.mat(matrix_b) # [N,d]
    num = np.dot(vector_a, matrix_b.T) # [1,N]
    denom = np.linalg.norm(vector_a) * np.linalg.norm(matrix_b, axis=1) # [N,]
    cos = np.array(num).squeeze(0) / denom
    return np.max(cos)


def check_police(feature, criterion):
    #track = []
    #for i in range(criterion.shape[0]):
    val = cos_sim_batch(feature, criterion)
    #    track.append(val)
    #score = max(track)
    return val

def run_video(file_path=None):
    shutil.rmtree('tmp/')
    os.mkdir('tmp/')
    batch_size = 1
    with h5py.File('/data1/Project/TF_FeatureExtraction/features.h5','r') as f:
        criterion = f['resnet_v2_101']['logits'].value.squeeze(axis=1).squeeze(axis=1) #[N,d]
    feature_extractor = FeatureExtractor(
        network_name='resnet_v2_101',
        checkpoint_path='/data1/Project/TF_FeatureExtraction/checkpoints/resnet_v2_101.ckpt',
        batch_size=batch_size,
        num_classes=1001,
        preproc_func_name='inception',
        preproc_threads=2
    )
    FLAG = False
    keypoints_track = []
    if not file_path:
        file_path = 0
    cap = cv2.VideoCapture(file_path)
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 15, (int(cap.get(3)), int(cap.get(4))))
    while 1:
        t = cv2.getTickCount()
        # Read new image
        ret, frame = cap.read()
        #cv2.imwrite('screenshot.jpg', frame)
        if not ret:
            break
        # Output keypoints and the image with the human skeleton blended on it
        keypoints, output_image = openpose.forward(frame, True)
        #keypoints_track.append(keypoints)
        for i in range(keypoints.shape[0]):
            #boxes = body_parts_box(keypoints[i,:,:])
            box = upper_body_box(keypoints[i,:,:])
            if box is None:
                continue
            if box[0][0] <=0 or box[0][1] <= 0 or box[1][0] <= 0 or box[1][1] <=0:
                print(keypoints[i,:,:])
                FLAG = True
            cv2.imwrite('tmp/p%d.jpg'%i, frame[box[0][1]:box[1][1]+1, box[0][0]:box[1][0]+1, :])
            feature_data = feature_extraction_queue(feature_extractor, 'tmp/p%d.jpg'%i, 
                            ['resnet_v2_101/logits'], batch_size, num_classes=1001)
            feature = feature_data['resnet_v2_101/logits'].squeeze(axis=1).squeeze(axis=1)
            score = check_police(feature, criterion)
            if score >= 0.5: # is police
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)
            shutil.rmtree('tmp/')
            os.mkdir('tmp/')
            cv2.rectangle(output_image, box[0], box[1], color, 2)
            #if score >= 0.5:
            #    cv2.imwrite('police_uniform/%s.jpg' % str(time.time()).replace('.', ''), frame[box[0][1]:box[1][1]+1, box[0][0]:box[1][0]+1, :])
            cv2.putText(output_image, "%.3f"%score, box[0], cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255))
            #cv2.drawContours(output_image, [box], 0, (0,0,255), 2)
            #for bb in boxes:
            #    if not bb:
            #        continue
            #    cv2.rectangle(output_image, bb[0], bb[1], (0, 0, 255), 2)
        #for i in range(keypoints.shape[0]):
        #    detect_action(keypoins[i,:,:])
        out.write(output_image)
        # Compute FPS
        t = (cv2.getTickCount() - t)/cv2.getTickFrequency()
                
        fps = 1.0 / t
        cv2.putText(output_image, "%.1f FPS"%fps, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
        # Display the image
        cv2.imshow("output", output_image)
        if FLAG:
            key = cv2.waitKey(0)
            if key == ord(' '):
                FLAG = False
                continue
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    feature_extractor.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def main():
    file_path = '/data1/Project/Jail/给杨博士/法制行为录像/output3.mp4'
    run_video(file_path)


if __name__ == '__main__':
    main()
