


import sys
import tensorflow as tf
import numpy as np
if __name__ == "__main__":
    sys.path.append("../../")
from dotpy_src.postprocessing.utils import sort_some_lists_of_tensors
from dotpy_src.box_encode_decode.ssd.encode import encode
from dotpy_src.box_encode_decode.ssd.utils import get_boxes_labels_zero_out_holes
from dotpy_src.box_encode_decode.ssd.decode import decode
from dotpy_src.postprocessing.ssd import detected_bboxes
from dotpy_src.postprocessing.unpack import unpack_net_output
from dotpy_src.postprocessing.bboxes import bboxes_matching_batch
from dotpy_src.postprocessing.utils import reshape_list
from dotpy_src.metrics.utils import precision_recall, average_precision_voc07, average_precision_voc12, EpochMetrics
from dotpy_src.configs import configs



def calc_batch_metrics(y_true, y_preds):
    gclasses, glocalizations, gscores = encode(y_true)

    
    # changes so labels go from 1 to 4
    glabels, gbboxes = get_boxes_labels_zero_out_holes(y_true)
    localizations, logits, predictions = unpack_net_output(y_preds)
    
    logits, localizations, gclasses, glocalizations, gscores, predictions = sort_some_lists_of_tensors(logits,
                                                                                           localizations, 
                                                                                           gclasses, 
                                                                                           glocalizations, 
                                                                                          gscores, predictions)

    #boolean mask of whether gtruth object is difficult or not (only used for pascal voc?)
    gdifficults = tf.zeros_like(glabels)

    
    # Performing post-processing on CPU: loop-intensive, usually more efficient.
    #with tf.device('/device:CPU:0'):
    localizations = decode(localizations)
    
    # get top k  predicted boxes after nms
    rscores, rbboxes = detected_bboxes(predictions, localizations,
                            select_threshold=configs["select_threshold"],
                            nms_threshold=configs["nms_threshold"],
                            clipping_bbox=None,
                            top_k=configs["select_top_k"],
                            keep_top_k=configs["keep_top_k"])
    
    #match the predicted boxes to ground truth boxes and compute the TP and FP statistics.
    num_gbboxes, tp, fp, rscores =     bboxes_matching_batch(rscores.keys(), rscores, rbboxes,
                              glabels, gbboxes, gdifficults,
                              matching_threshold=configs["matching_threshold"])
    
    return num_gbboxes, tp, fp, rscores    



def calc_ap_one_class():
    num_classes = configs["num_classes"]
    
    
    num_gbboxes = tf.placeholder(dtype=tf.int32)
    tp = tf.placeholder(dtype=tf.bool)
    fp = tf.placeholder(dtype=tf.bool)
    rscores = tf.placeholder(dtype=tf.float32)
    num_detections = tf.placeholder(dtype=tf.int32)
    
    
    # Add to summaries precision/recall values.
    aps_voc07 = {}
    aps_voc12 = {}

    # Precison and recall values.
    prec, rec = precision_recall(num_gbboxes, num_detections,
                                tp, fp, rscores )

    # Average precision VOC07.
    v = average_precision_voc07(prec, rec)
    aps_voc07 = v

    # Average precision VOC12.
    v = average_precision_voc12(prec, rec)

    aps_voc12 = v
        
    return aps_voc12,[num_gbboxes, num_detections, tp, fp, rscores]



if __name__ == "__main__":
    with tf.Session() as sess:
        from dotpy_src.load_data.get_generator import get_generator
        gen=get_generator("tr", batch_size=2)
        y_true = tf.placeholder(tf.float32,shape=(2,15,5),name="y_true")
        shapes = [(2, 6, 9, 48),
                 (2, 3, 5, 32),
                 (2, 96, 144, 32),
                 (2, 24, 36, 48),
                 (2, 12, 18, 48),
                 (2, 48, 72, 48),
                 (2, 1, 1, 32)]

        y_preds = [tf.ones((shape)) for shape in shapes]
        #num_gbboxes, tp, fp, rscores 
        batch_metrics = calc_batch_metrics(y_true, y_preds)
        epm = EpochMetrics()
        for ind, (im, box) in enumerate(gen):
            updated_batch_metrics  = sess.run(batch_metrics, feed_dict={y_true:box})
            epm.update_metrics(*updated_batch_metrics)
            break
        
        final_metrics = epm.get_final_metrics()
        aps_voc12, placeholders = calc_ap_one_class()
        all_aps12 = {}


        print final_metrics
        for c in final_metrics[0].keys():
            placefillers = [d[c] for d in final_metrics]
            all_aps12[c] = sess.run(aps_voc12, feed_dict = dict(zip(placeholders, placefillers)) )
        print all_aps12
        
        mAP12 = np.mean(all_aps12.values())
        print mAP12





