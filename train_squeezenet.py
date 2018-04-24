import tensorflow as tf 
import numpy as np 
from PIL import Image 
"""
Import SqueezeNet model definition
"""
from squeezenet_model import SqueezeNet


"""
Function to convert dense labels to one hot labels
"""
def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

"""
Function to load image from path.
"""
def get_image_new(image_path,width,height):
    image = Image.open(image_path)
    image = image.resize([width,height],Image.BILINEAR)
    image = np.array(image,dtype=np.float32)
    image = np.subtract(image,118)
    image = np.divide(image,255)
    
    return image


"""
Function to calculate accuracy
"""	
def acc(pred,lab):
    predic = tf.equal(tf.argmax(pred,1),tf.argmax(lab,1))
    accu = tf.reduce_sum(tf.cast(predic,tf.float32))
    return accu


"""
Function to get training data path files & labels and validation data path files & labels. (Training and Cross Validation text files)
"""    
def get_data(tr_file,val_file):
    data_files = []
    with open(tr_file,"r") as f:
        data_files = f.readlines()
    tr_data = []
    tr_labels = []
    for i in data_files:
        file = i.split(' ')
        tr_data.append(file[0])
        tr_labels.append(int(file[1]))
    tr_data = np.array(tr_data)
    tr_labels = np.array(tr_labels)
    perm = np.random.permutation(tr_data.shape[0])
    tr_data = tr_data[perm]
    tr_labels = tr_labels[perm]
    data_files = []
    with open(val_file,"r") as f:
        data_files = f.readlines()
    cv_data = []
    cv_labels = []
    for i in data_files:
        file = i.split(' ')
        cv_data.append(file[0])
        cv_labels.append(int(file[1]))
    
    return tr_data,np.array(tr_labels),cv_data,np.array(cv_labels)
    

"""
Training Function 
"""
    
def train(sq_net,lr_rate,max_iter,out_classes,batch_size,tr_data_files,tr_labels,cv_data_files,cv_labels,log_file):
    train_vars = tf.trainable_variables()
    v0_vars = [var for var in train_vars if var.name.startswith('squeezenet_v0')]
    v0_res_vars = [var for var in train_vars if var.name.startswith('squeezenet_v0_res')]
    v1_vars = [var for var in train_vars if var.name.startswith('squeezenet_v1')]
    saver_v0  = tf.train.Saver(var_list=v0_vars,max_to_keep=None)
    saver_v0_res  = tf.train.Saver(var_list=v0_res_vars,max_to_keep=None)
    saver_v1  = tf.train.Saver(var_list=v1_vars,max_to_keep=None)
    print ("strating training")
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
    
        step = 0
        bs = 0      
        max_cv_len = int(len(cv_data_files)/batch_size)*batch_size
        max_bs_len = int(len(tr_data_files)/batch_size)*batch_size
        while step < max_iter: 
            
            lr_rate = 0.0001 * (1 - step/max_iter)
            if bs >= max_bs_len:
                bs = 0 
            
            batch_files = tr_data_files[bs:(bs+batch_size)]
            batch_images = np.array([get_image_new(sample_file,227,227) for sample_file in batch_files]).astype(np.float32)
            batch_labels = np.array(dense_to_one_hot(tr_labels[bs:(bs+batch_size)],out_classes))
            
            sess.run([sq_net.v0_opt,sq_net.v0_res_opt,sq_net.v1_opt],feed_dict={sq_net.inputs:batch_images,sq_net.labels:batch_labels,sq_net.lr_rate:lr_rate})
                   
                        
            if step % 5 == 0:
                loss_v0,loss_v0_res,loss_v1 = sess.run([sq_net.loss_v0,sq_net.loss_v0_res,sq_net.loss_v1],feed_dict={sq_net.inputs:batch_images,sq_net.labels:batch_labels,sq_net.lr_rate:lr_rate})
                print ("step = %r, loss_v0 = %r loss v0_res = %r loss _v1 = %r" % (step,loss_v0,loss_v0_res,loss_v1))
            
            if step % 250 == 0: 
                cv_bs = 0
                acc_v0 = 0
                acc_v0_res = 0
                acc_v1 = 0
                                    
                while cv_bs < max_cv_len:
                        
                    cv_files = cv_data_files[cv_bs:(cv_bs+batch_size)]
                    cv_images = np.array([get_image_new(sample_file,227,227) for sample_file in cv_files]).astype(np.float32)
                    cv_img_labels = np.array(dense_to_one_hot(cv_labels[cv_bs:(cv_bs+batch_size)],out_classes))
                       
                    pred_v0,pred_v0_res,pred_v1 = sess.run(sq_net.model_prediction(sq_net.inputs,False),feed_dict={sq_net.inputs:cv_images,sq_net.labels:cv_img_labels,sq_net.lr_rate:lr_rate})
                    acc_cv_bs = sess.run(acc(pred_v0,cv_img_labels),feed_dict={sq_net.inputs:cv_images,sq_net.labels:cv_img_labels,sq_net.lr_rate:lr_rate})
                    acc_v0 = acc_v0 + acc_cv_bs
                    acc_cv_bs = sess.run(acc(pred_v0,cv_img_labels),feed_dict={sq_net.inputs:cv_images,sq_net.labels:cv_img_labels,sq_net.lr_rate:lr_rate})
                    acc_v0_res = acc_v0_res + acc_cv_bs
                    acc_cv_bs = sess.run(acc(pred_v0,cv_img_labels),feed_dict={sq_net.inputs:cv_images,sq_net.labels:cv_img_labels,sq_net.lr_rate:lr_rate})
                    acc_v1 = acc_v1 + acc_cv_bs
                    cv_bs = cv_bs + batch_size
                    print ("calc cv =%r" %(cv_bs))

                acc_v0 = (float(acc_v0)/max_cv_len) * 100.0
                acc_v0_res = (float(acc_v0_res)/max_cv_len) * 100.0
                acc_v1 = (float(acc_v1)/max_cv_len) * 100.0
                print ("Step = %r  acc_v0 = %r acc_v0_res = %r acc_v1 = %r" %(step,acc_v0,acc_v0_res,acc_v1))  
                with open(log_file,"a") as f:
                    f.write(str(str(step) + " " +  str(acc_v0) +" "+str(acc_v0_res) + " " +str(acc_v1) +"\n"))
                    
            if step % 500 == 0:
                dir_path_v0  =  model_dir + "v0_" + str(step)+"\\"
                # change second argument of restore function to the path where model weights have to be saved.
                saver_v0.save(sess,dir_path_v0,write_meta_graph=True)
                dir_path_v0_res  =  model_dir + "v0_res_" + str(step)+"\\"
                saver_v0_res.save(sess,dir_path_v0_res,write_meta_graph=True)
                dir_path_v1  =  model_dir + "v1_" + str(step)+"\\"
                saver_v1.save(sess,dir_path_v1,write_meta_graph=True)
                print ("### Model weights Saved step = %r ###" %(step))
            
            else:
                print ("step = %r" %(step))
            
            
            bs = bs + batch_size
            step = step + 1


#change model_dir to parent directory where model weights are to be saved. 
model_dir = "squeeze\\model\\"
#change tr_file path as produced by generate_files.py file
tr_file = "squeeze\\dataset\\train_file1.txt"
#change val_file path as produced by generate_files.py file
val_file = "squeeze\\dataset\\val_file1.txt"
log_file = model_dir +"log_file.txt"
input_shape = 227,227,3
batch_size = 256
lr_rate = 0.0001
out_classes = 200
is_train = True
max_iter = 25000
tr_data_files,tr_labels,cv_data_files,cv_labels = get_data(tr_file,val_file)
tf.reset_default_graph()

sq_net = SqueezeNet(input_shape,out_classes,lr_rate,is_train)

train(sq_net,lr_rate,max_iter,out_classes,batch_size,tr_data_files,tr_labels,cv_data_files,cv_labels,log_file)
