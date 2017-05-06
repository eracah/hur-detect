


import sys
import tensorflow as tf
slim=tf.contrib.slim
if __name__ == "__main__":
    sys.path.append("../../../")



from dotpy_src.box_encode_decode.ssd.make_anchors_orig import make_anchors_for_one_fmap
from dotpy_src.box_encode_decode.ssd.encode import encode
from dotpy_src.configs import configs
from dotpy_src.postprocessing.utils import get_int_tensor_shape, sort_some_lists_of_tensors
from dotpy_src.postprocessing.unpack import unpack_net_output
from utils import ssd_losses



loss_weights = None



def compute_loss(y_true, y_preds):
    '''y_true: the boxes Nx15x5 tensor
       y_preds: a list of 7?  tensors of Nxfy x fx x k where k = 4*number of anchors + number_of_anchors*num_classes,
       N is number of examples'''

    gclasses, glocalizations, gscores = encode(y_true)
    
    localizations, logits, predictions = unpack_net_output(y_preds)
    
    #sort so in same order
    logits, localizations, gclasses, glocalizations, gscores = sort_some_lists_of_tensors(logits,
                                                                                           localizations, 
                                                                                           gclasses, 
                                                                                           glocalizations, 
                                                                                           gscores)

    final_loss = ssd_losses(logits, localizations,
                      gclasses, glocalizations, gscores,
                      match_threshold=configs["matching_threshold"],
                      negative_ratio=configs["negative_ratio"],
                      alpha=configs["alpha"],
                      label_smoothing=configs["label_smoothing"],
                      scope="ssd_losses")
    return final_loss



if __name__ == "__main__":
    with tf.Session() as sess:
        from dotpy_src.load_data.get_generator import get_generator

        gen=get_generator("tr", batch_size=2)
        bboxes = tf.placeholder(tf.float32,shape=(2,15,5),name="bboxes")
        shapes = [(2, 6, 9, 48),
                 (2, 3, 5, 32),
                 (2, 96, 144, 32),
                 (2, 24, 36, 48),
                 (2, 12, 18, 48),
                 (2, 48, 72, 48),
                 (2, 1, 1, 32)]

        y_preds = [tf.ones((shape)) for shape in shapes]

        final_loss = compute_loss(bboxes, y_preds)
        for im, box in gen:
            print sess.run(final_loss, feed_dict={bboxes:box})





