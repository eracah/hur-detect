


import sys
import tensorflow as tf
import numpy as np
from os.path import join
if __name__ == "__main__":
    sys.path.append("../../")
from dotpy_src.metrics.mAP import calc_batch_metrics, EpochMetrics, calc_ap_one_class
from dotpy_src.configs import configs
import time



def fit(model, generator, val_generator,num_epochs, loss_func, opt):
    with tf.Session() as sess:

        tr_steps_per_epoch= generator.num_ims / generator.batch_size
        val_steps_per_epoch = val_generator.num_ims / val_generator.batch_size
        
        
        y_true, y_preds = get_y_true_y_preds_tensors(model, generator.batch_size,generator.data.labels.shape[1:])
        
        with tf.name_scope("loss"):
            loss_tensor = loss_func(y_true, y_preds)
            tf.summary.scalar("loss", loss_tensor)
        
        with tf.name_scope("accuracy"):
            accuracy_tensor = tf.placeholder(dtype=tf.float32, shape=())
            tf.summary.scalar("accuracy", accuracy_tensor)
        
        summaries_dir = get_summaries_dir()
        train_writer = tf.summary.FileWriter(summaries_dir + '/train',
                                      sess.graph)
        val_writer = tf.summary.FileWriter(summaries_dir + '/val')
        
        
        
        input_ = model.input

        train_step = opt.minimize(loss_tensor)
        
        merged = tf.summary.merge_all()
        
        sess.run(tf.global_variables_initializer())
        tr_global_step_counter = 0
        val_global_step_counter = 0
        for ep in range(num_epochs):
            
            tr_mAP, tr_APs = get_epoch_accuracy(generator, model, sess, input_)
            val_mAP, val_APs = get_epoch_accuracy(val_generator, model, sess, input_)
 
            for step in range(tr_steps_per_epoch):
                im, boxes = generator.next()
                _,summary = sess.run([train_step, merged],feed_dict={input_:im, y_true:boxes, accuracy_tensor:tr_mAP})
                train_writer.add_summary(summary,tr_global_step_counter)
                tr_global_step_counter += 1
                
            
            for step in range(val_steps_per_epoch):
                im, boxes = generator.next()
                summary, val_loss = sess.run([merged, loss_tensor], feed_dict={input_:im, y_true:boxes, accuracy_tensor:val_mAP})
                val_writer.add_summary(summary, val_global_step_counter)
                val_global_step_counter += 1
  
            

            

            
            
                
                



def get_summaries_dir():
    if configs["exp_name"] == "None":
        exp_name = "_".join([configs["base_model"], configs["detection_model"]]) + "_" + str(int(time.time()))
    else:
        exp_name = configs["exp_name"]
    return join(configs["logs_dir"],exp_name )



def get_epoch_accuracy(generator, model, sess,input_):
    epm = EpochMetrics()
    batch_accuracy_tensors, y_true = get_batch_accuracy_tensors(calc_batch_metrics, model, generator)
    steps_per_epoch = generator.num_ims / generator.batch_size
    for step in range(steps_per_epoch):
        im, boxes = generator.next()
        updated_metrics = sess.run(batch_accuracy_tensors, feed_dict={y_true:boxes, input_:im})
        epm.update_metrics(*updated_metrics)

    final_metrics = epm.get_final_metrics()
    aps_voc12, placeholders= calc_ap_one_class()
    all_aps12 = {}


    for c in final_metrics[0].keys():
        placefillers = [d[c] for d in final_metrics]
        all_aps12[c] = sess.run(aps_voc12, feed_dict = dict(zip(placeholders, placefillers)) )


    mAP = np.mean(all_aps12.values())
    return mAP, all_aps12



def get_y_true_y_preds_tensors(model, batch_size, label_shape):
    output_tensors = model.outputs
        
    label_batch_shape = tuple([batch_size] + list(label_shape))
        
        
    label_tensor = tf.placeholder(tf.float32,shape=label_batch_shape, name="label")
    return label_tensor, output_tensors



def get_batch_accuracy_tensors(acc_func, model,generator):
        batch_size = generator.batch_size
        y_true, y_preds = get_y_true_y_preds_tensors(model, batch_size,generator.data.labels.shape[1:])
        batch_metrics = calc_batch_metrics(y_true, y_preds)
        return batch_metrics, y_true
        
    





