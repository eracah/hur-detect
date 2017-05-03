



def detected_bboxes(predictions, localisations,
                    select_threshold=None, nms_threshold=0.5,
                    clipping_bbox=None, top_k=400, keep_top_k=200):
    """Get the detected bounding boxes from the SSD network output.
    """
    # Select top_k bboxes from predictions, and clip
    rscores, rbboxes =         ssd_common.tf_ssd_bboxes_select(predictions, localisations,
                                        select_threshold=select_threshold,
                                        num_classes=self.params.num_classes)
    rscores, rbboxes =         tfe.bboxes_sort(rscores, rbboxes, top_k=top_k)
    # Apply NMS algorithm.
    rscores, rbboxes =         tfe.bboxes_nms_batch(rscores, rbboxes,
                             nms_threshold=nms_threshold,
                             keep_top_k=keep_top_k)
    # if clipping_bbox is not None:
    #     rbboxes = tfe.bboxes_clip(clipping_bbox, rbboxes)
    return rscores, rbboxes

