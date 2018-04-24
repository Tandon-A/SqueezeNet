import tensorflow as tf 
import numpy as np 
from glob import glob
from PIL import Image
import os 
"""
Import squeezenet model definition.
"""
from squeezenet_model import SqueezeNet



def get_image_new(image_path,width,height):
    image = Image.open(image_path)
    image = image.resize([width,height],Image.BILINEAR)
    image = np.array(image,dtype=np.float32)
    image = np.subtract(image,118)
    image = np.divide(image,255)
    return np.array(image)

def get_labels(data_path,file_path):
    files = os.listdir(data_path)
    with open(file_path,"r") as f:
        data = f.readlines()
    labels = []
    for i in files:
        for j in range(len(data)):
            if str(data[j].split(' ')[0]) == str(i):
                labels.append(data[j].split(' ')[1])
                break
    return labels        
    

def test(sq_net,test_files,labels):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_vars = tf.trainable_variables()
        v0_vars = [var for var in train_vars if var.name.startswith('squeezenet_v0')]
        v0_res_vars = [var for var in train_vars if var.name.startswith('squeezenet_v0_res')]
        v1_vars = [var for var in train_vars if var.name.startswith('squeezenet_v1')]
        saver_v0  = tf.train.Saver(var_list=v0_vars,max_to_keep=None)
        saver_v0_res  = tf.train.Saver(var_list=v0_res_vars,max_to_keep=None)
        saver_v1  = tf.train.Saver(var_list=v1_vars,max_to_keep=None)
        #change second argument of restore function to path where weights are saved (model wise)
        saver_v0.restore(sess,model_dir+"v0_8000\\")
        saver_v0_res.restore(sess,model_dir+"v0_res_8000\\")
        saver_v1.restore(sess,model_dir + "v1_8000\\")
        
        for i in range(len(test_files)):
            img = np.reshape(get_image_new(test_files[i],227,227),(1,227,227,3))
            pred_v0,pred_v0_res,pred_v1 = sess.run(sq_net.model_prediction(sq_net.inputs,False),feed_dict={sq_net.inputs:img})
            pred_v0 = np.argmax(pred_v0,1)
            pred_v0_res = np.argmax(pred_v0_res,1)
            pred_v1 = np.argmax(pred_v1,1)
            
            print ("img = %r pred_v0 = %r, %r pred_v0_res = %r, %r pred_v1 = %r, %r" %(i,pred_v0,labels[pred_v0],pred_v0_res,labels[pred_v0_res],pred_v1,labels[pred_v1]))
                    

#change model_dir to parent directory where model weights are saved
model_dir = "squeeze\\model\\"
#path of training data stored classes wise 
data_path = "squeeze\\dataset\\tiny-imagenet-200\\train\\"
#path of labels for tiny imagenet 
file_path = "squeeze\\dataset\\tiny-imagenet-200\\words.txt"
#path of testing data 
test_files = glob("squeeze\\dataset\\tiny-imagenet-200\\test\images\\*.JPEG")
input_shape = 227,227,3
batch_size = 256
lr_rate = 0.04
out_classes = 200
is_train = False
tf.reset_default_graph()

sq_net = SqueezeNet(input_shape,out_classes,lr_rate,is_train)
labels = get_labels(data_path)
test(sq_net,test_files,labels)
