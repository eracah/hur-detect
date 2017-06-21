


import sys
import tensorflow as tf
import numpy as np
from os.path import join
if __name__ == "__main__":
    sys.path.append("../../")
#from dotpy_src.metrics.mAP import calc_batch_metrics, EpochMetrics, calc_ap_one_class
from dotpy_src.configs import configs
import time



def fit(model, generator, val_generator,num_epochs, loss_func, opt):
    with tf.Session() as sess:

        tr_steps_per_epoch= generator.num_ims / generator.batch_size
        val_steps_per_epoch = val_generator.num_ims / val_generator.batch_size
        

        y_true, y_preds = get_y_true_y_preds_tensors(model, generator.batch_size,generator.data.labels.shape[1:])
        
        
        
        #running_average_loss = tf.placeholder(dtype=tf.float32,shape=())
        loss_tensor = loss_func(y_true, y_preds)
        with tf.name_scope("loss"):
            tf.summary.scalar("loss", loss_tensor)
            #tf.summary.scalar("running_average_loss", running_average_loss)
            
            
        
#         with tf.name_scope("accuracy"):
#             accuracy_tensor = tf.placeholder(dtype=tf.float32, shape=())
#             tf.summary.scalar("accuracy", accuracy_tensor)
    
        summaries_dir = get_summaries_dir()
        train_writer = tf.summary.FileWriter(summaries_dir + '/train',
                                      sess.graph)
        train_epoch_writer = tf.summary.FileWriter(summaries_dir + '/train_epoch',
                                      sess.graph)
        val_writer = tf.summary.FileWriter(summaries_dir + '/val')
        val_epoch_writer = tf.summary.FileWriter(summaries_dir + '/val_epoch',
                                      sess.graph)
        
        
        input_ = model.input

        train_step = opt.minimize(loss_tensor)
        
        merged = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())
        tr_global_step_counter = 0
        val_global_step_counter = 0
        print "beginning training"
        
        for epoch in range(num_epochs):
            t0 = time.time()
            
            
            tr_global_step_counter =  run_loss_loop(type_="tr", epoch=epoch, steps_per_epoch=tr_steps_per_epoch, 
                                                    step_writer=train_writer, 
                                                    epoch_writer=train_epoch_writer, generator=generator, 
                                                    train_step=train_step, loss_tensor=loss_tensor, summary_op=merged, 
                                                    input_=input_, y_true=y_true, global_step=tr_global_step_counter, sess=sess)
            
            
            
            t1 = time.time()
            epoch_time = t1-t0
            print "epoch time: ", epoch_time
            write_summary(epoch_time, "epoch_time", train_epoch_writer, epoch)
            val_global_step_counter = run_loss_loop(type_="val", epoch=epoch, steps_per_epoch=val_steps_per_epoch, 
                                                    step_writer=val_writer, epoch_writer=val_epoch_writer, generator=val_generator, 
                                                    train_step=None, loss_tensor=loss_tensor, summary_op=merged, 
                                                    input_=input_, y_true=y_true, global_step=val_global_step_counter,sess=sess)
            
#             if epoch % 10 == 0:
#                 get_epoch_accuracy(generator, model, sess, input_,train_epoch_writer, epoch)
#                 get_epoch_accuracy(val_generator, model, sess, input_, val_epoch_writer, epoch)
            
  
            

            

            



def write_summary(value, name, writer, index):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        writer.add_summary(summary, index)
    



def run_loss_loop(type_, epoch, steps_per_epoch, step_writer, epoch_writer, generator, 
                  train_step, loss_tensor, summary_op, input_, y_true, global_step, sess):
    
    if type_ == "tr":
        sess_list = [train_step,loss_tensor, summary_op]
    else:
        sess_list = [loss_tensor,loss_tensor, summary_op]
    
    loss_sum = 0.0
    for step in range(steps_per_epoch):
        im, boxes = generator.next()

        _, cur_loss, summary = sess.run(sess_list,feed_dict={input_:im, 
                                                            y_true:boxes})
        print cur_loss
        loss_sum += cur_loss


        step_writer.add_summary(summary,global_step)
        global_step += 1
    average_loss = loss_sum / float(steps_per_epoch)
    print "at epoch %i, the loss for %s is %8.4f" %(epoch,type_, average_loss)
    write_summary(average_loss,"running_average_loss", epoch_writer, epoch)
    return global_step




    

def get_summaries_dir():
    if configs["exp_name"] == "None":
        exp_name = "_".join([configs["base_model"], configs["detection_model"]]) + "_" + str(int(time.time()))
    else:
        exp_name = configs["exp_name"]
    return join(configs["logs_dir"],exp_name )

def get_epoch_accuracy(generator, model, sess,input_,writer,epoch):
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
    write_summary(mAP,"accuracy", writer, epoch)
    for cls, ap in all_aps12.iteritems():
        write_summary(ap,"class " + str(cls) + " accuracy", writer, epoch)

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
        
    



# a =tf.Variable(dtype=tf.float32, initial_value=1.0)

# b = tf.placeholder(dtype=tf.int32)

# eq_zero = tf.cast(tf.equal(b,0), dtype=tf.float32)
# d=a.assign_add(1.0)
# c=a.assign_add(-d*eq_zero)



# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in range(5):
#         for j in range(10):
#             print sess.run(c,feed_dict={b:j})
        

