


import tensorflow as tf
import numpy as np



def fit(model, generator, val_generator,num_epochs, loss_func, opt):
    with tf.Session() as sess:
        steps_per_epoch= generator.num_ims / generator.batch_size
        input_ = model.input
        loss_tensor, label  = get_loss_tensor(loss_func, model, generator)

        train_step = opt.minimize(loss_tensor)
        # accuracy = get_accuracy
        sess.run(tf.global_variables_initializer())
  
        for ep in range(num_epochs):
            for step in range(steps_per_epoch):
                im, box = generator.next()
                _,cur_loss = sess.run([train_step, loss_tensor],feed_dict={input_:im,label:box})
                print "loss: %8.4f"% cur_loss
#TODO: add streaming mAP calc for train and val



def get_loss_tensor(loss_func, model, generator):
        batch_size = generator.batch_size
        output_tensors = model.outputs
        
        label_batch_shape = tuple([batch_size] + list(generator.data.labels.shape[1:]))
        
        
        label_tensor = tf.placeholder(tf.float32,shape=label_batch_shape, name="label")
        

        loss_tensor = loss_func(label_tensor, output_tensors)

        return loss_tensor, label_tensor





