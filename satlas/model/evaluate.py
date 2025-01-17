import json
import numpy as np
import os, os.path
import rasterio.features
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
import shapely
import skimage.draw
import skimage.io
import skimage.morphology
import tqdm
import torch
import torchvision
import cv2
from skimage.morphology import skeletonize
import satlas.model.dataset
import satlas.util
from shapely.geometry import Polygon
from shapely.ops import unary_union
from torchvision.ops import nms
from skimage.measure import find_contours

# Match predicted points with ground truth points to compute precision and recall.
def compute_loc_performance(gt_array, pred_array, eval_type, distance_tolerance, iou_threshold):
    # distance_matrix below doesn't work when preds is empty, so handle that first
    if len(pred_array) == 0:
        return [], [], gt_array.tolist()

    # Different matrix comparison depending on evaluating using center or IoU
    if eval_type == 'center':
        # Building distance matrix using Euclidean distance pixel space
        # multiplied by the UTM resolution (10 m per pixel)
        dist_mat = distance_matrix(pred_array, gt_array, p=2)
        dist_mat[dist_mat > distance_tolerance] = 99999
    elif eval_type == 'iou':
        # Compute pair-wise IoU between the current class targets and all predictions
        dist_mat = torchvision.ops.box_iou(gt_array, pred_array)
        dist_mat[dist_mat < iou_threshold] = 99999
        dist_mat = torch.transpose(dist_mat, 1, 0).cpu().detach().numpy()

    # Using Hungarian matching algorithm to assign lowest-cost gt-pred pairs
    rows, cols = linear_sum_assignment(dist_mat)

    if eval_type == 'center':
        tp_inds = [
            {"pred_idx": rows[ii], "gt_idx": cols[ii]}
            for ii in range(len(rows))
            if dist_mat[rows[ii], cols[ii]] < distance_tolerance
        ]
    elif eval_type == 'iou':
        tp_inds = [
            {"pred_idx": rows[ii], "gt_idx": cols[ii]}
            for ii in range(len(rows))
            if dist_mat[rows[ii], cols[ii]] > iou_threshold
        ]

    tp = [
        {'pred': pred_array[a['pred_idx']].tolist(), 'gt': gt_array[a['gt_idx']].tolist()}
        for a in tp_inds
    ]
    tp_inds_pred = set([a['pred_idx'] for a in tp_inds])
    tp_inds_gt = set([a['gt_idx'] for a in tp_inds])
    fp = [pred_array[i].tolist() for i in range(len(pred_array)) if i not in tp_inds_pred]
    fn = [gt_array[i].tolist() for i in range(len(gt_array)) if i not in tp_inds_gt]

    return tp, fp, fn

def point_score(gt, pred, eval_type, distance_tolerance=20, iou_threshold=0.5):
    tp, fp, fn = [], [], []

    for scene_id in gt.keys():
        cur_tp, cur_fp, cur_fn = compute_loc_performance(gt[scene_id], pred[scene_id], eval_type, distance_tolerance, iou_threshold)
        tp += [{'scene_id': scene_id, 'pred': a['pred'], 'gt': a['gt']} for a in cur_tp]
        fp += [{'scene_id': scene_id, 'point': a} for a in cur_fp]
        fn += [{'scene_id': scene_id, 'point': a} for a in cur_fn]

    return len(tp), len(fp), len(fn)

class SegmentAccuracyEvaluator(object):
    def __init__(self, task, spec, detail_func=None, params=None):
        self.task = task
        self.detail_func = detail_func
        self.scores = []

    def evaluate(self, valid, gt, pred):
        pred_argmax = pred.argmax(dim=1)
        score = (((pred_argmax == gt) & valid).float().sum() / torch.count_nonzero(valid))
        self.scores.append(score.item())

        if self.detail_func:
            # Get per-class scores.
            for cls_id, cls_name in enumerate(self.task['categories']):
                cls_mask = ((gt == cls_id) & valid).long()
                cls_count = torch.count_nonzero(cls_mask).item()
                if cls_count == 0:
                    continue

                scores = ((pred_argmax == gt) & valid).long()
                correct = (scores*cls_mask).sum()
                score = correct.item() / cls_count

                self.detail_func('{}_{}'.format(self.task['name'], cls_name), score)

    def score(self):
        if len(self.scores) == 0:
            score = 0
        else:
            score = np.mean(self.scores)

        return score, None

class SegmentLogProbEvaluator(object):
    def __init__(self, task, spec, detail_func=None, params=None):
        self.scores = []

    def evaluate(self, valid, gt, pred):
        # Select output probabilities corresponding to the ground truth class indices.
        index = gt[:, None, :, :]
        probs = pred.gather(dim=1, index=index)[:, 0, :, :]
        score = ((torch.log(probs) * valid.float()).sum() / torch.count_nonzero(valid))
        self.scores.append(score.item())

    def score(self):
        if len(self.scores) == 0:
            score = 0
        else:
            score = np.mean(self.scores)

        return score, None

class SegmentF1Evaluator(object):
    def __init__(self, task, spec, detail_func=None, params=None):
        self.evaluator = BinSegmentF1Evaluator(task, spec, detail_func=detail_func, params=params)

    def evaluate(self, valid, gt, pred):
        gt = torch.nn.functional.one_hot(gt.long(), num_classes=pred.shape[1]).permute(0, 3, 1, 2)
        return self.evaluator.evaluate(valid, gt, pred)

    def score(self):
        return self.evaluator.score()

class SegmentMIOUEvaluator(object):
    def __init__(self, task, spec, detail_func=None, params=None):
        self.n_classes = len(task['categories'])
        self.scores = []

    def evaluate(self, valid, gt, pred):
        gt = torch.flatten(gt)
        pred = torch.flatten(pred.argmax(dim=1))

        iou = 0
        n_observed = self.n_classes
        for i in range(self.n_classes):
            y_t = (gt == i).to(torch.int32)
            y_p = (pred == i).to(torch.int32)

            inter = torch.sum(y_t * y_p)
            union = torch.sum((y_t + y_p > 0).to(torch.int32))

            if union == 0:
                n_observed -= 1
            else:
                iou += inter / union

        self.scores.append((iou / n_observed).item())

    def score(self):
        if len(self.scores) == 0:
            score = 0
        else:
            score = np.mean(self.scores)

        return score, None

class BinSegmentAccuracyEvaluator(object):
    def __init__(self, task, spec, detail_func=None, params=None):
        self.scores = []

    def evaluate(self, valid, gt, pred):
        score = ((((pred > 0.5) == (gt > 0.5)) & valid).float().sum() / torch.count_nonzero(valid)).item()
        self.scores.append(score.item())

    def score(self):
        if len(self.scores) == 0:
            score = 0
        else:
            score = np.mean(self.scores)

        return score, None

class BinSegmentF1Evaluator(object):
    def __init__(self, task, spec, detail_func=None, params=None):
        '''
        params: list of thresholds, one for each class.
        '''
        self.task = task
        self.detail_func = detail_func

        # Set thresholds: each class can have multiple threshold options.
        num_classes = len(self.task['categories'])
        if params:
            self.thresholds = [[threshold] for threshold in params]
        else:
            self.thresholds = [[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95] for _ in range(num_classes)]

        self.gt_counts = [0 for _ in range(len(self.thresholds))]
        self.true_positives = [[0]*len(self.thresholds[i]) for i in range(len(self.thresholds))]
        self.false_positives = [[0]*len(self.thresholds[i]) for i in range(len(self.thresholds))]
        self.false_negatives = [[0]*len(self.thresholds[i]) for i in range(len(self.thresholds))]

    def evaluate(self, valid, gt, pred):
        for cls_idx, cls_thresholds in enumerate(self.thresholds):
            for threshold_idx, threshold in enumerate(cls_thresholds):
                pred_bin = (pred[:, cls_idx, :, :] > threshold) & valid
                gt_bin = (gt[:, cls_idx, :, :] > threshold) & valid
                tp = torch.count_nonzero(pred_bin & gt_bin).item()
                fp = torch.count_nonzero(pred_bin & torch.logical_not(gt_bin)).item()
                fn = torch.count_nonzero(torch.logical_not(pred_bin) & gt_bin).item()
                self.gt_counts[cls_idx] += torch.count_nonzero(gt_bin).item()
                self.true_positives[cls_idx][threshold_idx] += tp
                self.false_positives[cls_idx][threshold_idx] += fp
                self.false_negatives[cls_idx][threshold_idx] += fn

    def score(self):
        best_scores = []
        best_thresholds = []

        for cls_idx, cls_thresholds in enumerate(self.thresholds):
            if self.gt_counts[cls_idx] == 0:
                best_thresholds.append(0.5)
                continue

            best_score = None
            best_threshold = None

            for threshold_idx, threshold in enumerate(cls_thresholds):
                tp = self.true_positives[cls_idx][threshold_idx]
                fp = self.false_positives[cls_idx][threshold_idx]
                fn = self.false_negatives[cls_idx][threshold_idx]

                if tp + fp == 0:
                    precision = 0
                else:
                    precision = tp / (tp + fp)

                if tp + fn == 0:
                    recall = 0
                else:
                    recall = tp / (tp + fn)

                if precision + recall < 0.01:
                    f1 = 0
                else:
                    f1 = 2 * precision * recall / (precision + recall)

                if self.detail_func:
                    self.detail_func('{}_{}@{}'.format(self.task['name'], self.task['categories'][cls_idx], threshold), f1)

                if best_score is None or f1 > best_score:
                    best_score = f1
                    best_threshold = threshold

            best_scores.append(best_score)
            best_thresholds.append(best_threshold)

            if self.detail_func:
                self.detail_func('{}_{}'.format(self.task['name'], self.task['categories'][cls_idx]), best_score)

        if len(best_scores) == 0:
            return 0.0, best_thresholds

        return np.mean(best_scores), best_thresholds

class RegressSimpleEvaluator(object):
    def __init__(self, task, spec, cmp_func, detail_func=None, params=None):
        self.scores = []
        self.cmp_func = cmp_func

    def evaluate(self, valid, gt, pred):
        pred = torch.clip(pred, 0, 255)
        score = (self.cmp_func(pred/255, gt/255) * valid.float()).sum() / torch.count_nonzero(valid)
        self.scores.append(score.item())

    def score(self):
        if len(self.scores) == 0:
            score = 0
        else:
            score = np.mean(self.scores)

        return -score, None

class RegressMseEvaluator(object):
    def __init__(self, task, spec, detail_func=None, params=None):
        self.evaluator = RegressSimpleEvaluator(
            task=task,
            spec=spec,
            cmp_func=lambda pred, gt: torch.square(pred - gt),
            detail_func=detail_func,
            params=params
        )

    def evaluate(self, valid, gt, pred): return self.evaluator.evaluate(valid, gt, pred)
    def score(self): return self.evaluator.score()

class RegressL1Evaluator(object):
    def __init__(self, task, spec, detail_func=None, params=None):
        self.evaluator = RegressSimpleEvaluator(
            task=task,
            spec=spec,
            cmp_func=lambda pred, gt: torch.abs(pred - gt),
            detail_func=detail_func,
            params=params
        )

    def evaluate(self, valid, gt, pred): return self.evaluator.evaluate(valid, gt, pred)
    def score(self): return self.evaluator.score()

class DetectF1Evaluator(object):
    def __init__(self, task, spec, detail_func=None, params=None):
        '''
        params: list of thresholds, one for each class.
        '''
        self.eval_type = spec.get('EvalType', 'center')
        self.task = task
        self.detail_func = detail_func

        # Set thresholds: each class can have multiple threshold options.
        num_classes = len(self.task['categories'])
        if params:
            self.thresholds = [[threshold] for threshold in params]
        else:
            self.thresholds = [[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95] for _ in range(num_classes)]
            # self.thresholds = [list(np.arange(0.05, 1.0, 0.05)) for _ in range(num_classes)]

        self.true_positives = [[0]*len(self.thresholds[i]) for i in range(len(self.thresholds))]
        self.false_positives = [[0]*len(self.thresholds[i]) for i in range(len(self.thresholds))]
        self.false_negatives = [[0]*len(self.thresholds[i]) for i in range(len(self.thresholds))]

    def evaluate(self, gt_raw, pred_raw, iou_threshold=0.5):

        # Compute F1 score for each class individually, at each of that class' score thresholds
        for cls_idx, cls_thresholds in enumerate(self.thresholds):

            if cls_idx == 0:
                continue

            # Filter out ground truth objects with the current class label
            # If the evaluation type is center-based, replace boxes with centers
            gt = {}
            for image_idx, target in enumerate(gt_raw):
                # Get the relevant boxes (i.e., matching cls).
                # Have to handle no-box case separately since labels length is >= 1.
                if len(target['boxes']) == 0:
                    boxes = target['boxes'].numpy()
                else:
                    boxes = target['boxes'][target['labels'] == cls_idx, :].numpy()

                if self.eval_type == 'center':
                    gt[image_idx] = np.stack([
                        (boxes[:, 0] + boxes[:, 2])/2,
                        (boxes[:, 1] + boxes[:, 3])/2,
                    ], axis=1)
                elif self.eval_type == 'iou':
                    gt[image_idx] = boxes

            for threshold_idx, threshold in enumerate(cls_thresholds):
                pred = {}
                for image_idx, output in enumerate(pred_raw):
                    # Get the relevant boxes (i.e., matching cls and sufficient score).
                    if len(output['boxes']) == 0:
                        boxes = output['boxes'].numpy()
                    else:
                        selector = (output['scores'] >= threshold) & (output['labels'] == cls_idx)
                        boxes = output['boxes'][selector, :].numpy()

                    # If the evaluation type is center-based, replace the predicted boxes with centers
                    # Else if it is iou-based, keep the [x1,y1,x2,y2] box format
                    if self.eval_type == 'center':
                        pred[image_idx] = np.stack([
                            (boxes[:, 0] + boxes[:, 2])/2,
                            (boxes[:, 1] + boxes[:, 3])/2,
                        ], axis=1)
                    elif self.eval_type == 'iou':
                        pred[image_idx] = boxes

                tp, fp, fn = point_score(gt, pred, self.eval_type)
                self.true_positives[cls_idx][threshold_idx] += float(tp)
                self.false_positives[cls_idx][threshold_idx] += float(fp)
                self.false_negatives[cls_idx][threshold_idx] += float(fn)

    def score(self):
        best_scores = []
        best_thresholds = []
        best_precisions = []
        best_recalls = []

        for cls_idx, cls_thresholds in enumerate(self.thresholds):
            best_score = None
            best_threshold = None
            best_precision = None
            best_recall = None

            if cls_idx == 0:
                best_thresholds.append(0.5)
                continue

            for threshold_idx, threshold in enumerate(cls_thresholds):
                tp = self.true_positives[cls_idx][threshold_idx]
                fp = self.false_positives[cls_idx][threshold_idx]
                fn = self.false_negatives[cls_idx][threshold_idx]

                if tp + fp == 0:
                    precision = 0
                else:
                    precision = tp / (tp + fp)

                if tp + fn == 0:
                    recall = 0
                else:
                    recall = tp / (tp + fn)

                if precision + recall < 0.001:
                    f1 = 0
                else:
                    f1 = 2 * precision * recall / (precision + recall)

                if self.detail_func:
                    self.detail_func('{}_{}@{}_{}_{}'.format(self.task['name'], self.task['categories'][cls_idx], threshold, precision, recall), f1)

                if best_score is None or f1 > best_score:
                    best_score = f1
                    best_threshold = threshold
                    best_precision = precision
                    best_recall = recall

            best_precisions.append(best_precision)
            best_recalls.append(best_recall)
            best_scores.append(best_score)
            best_thresholds.append(best_threshold)

        # In all-background-class cases, avoid divide-by-zero errors
        if len(best_scores) == 0:
            return 0.0, best_thresholds, best_precisions, best_recalls

        return sum(best_scores) / len(best_scores), best_thresholds, best_precisions, best_recalls

class ClassificationAccuracyEvaluator(object):
    def __init__(self, task, spec, detail_func=None, params=None):
        self.task = task
        self.detail_func = detail_func

        num_classes = len(self.task['categories'])

        self.tot_samples = 0
        self.accuracies = []

    def evaluate(self, gt, pred):
        score = (pred.argmax(dim=1) == gt).float().sum()
        self.tot_samples += len(gt)
        self.accuracies.append(score.item())

    def score(self):
        if self.tot_samples == 0:
            return 0.0, None
        return sum(self.accuracies) / self.tot_samples, None


class ClassificationF1Evaluator(object):
    def __init__(self, task, spec, detail_func=None, params=None):
        '''
        params: list of thresholds, one for each class.
        '''
        self.task = task
        self.detail_func = detail_func

        # Set thresholds: each class can have multiple threshold options.
        num_classes = len(self.task['categories'])
        if params:
            self.thresholds = [[threshold] for threshold in params]
        else:
            self.thresholds = [list([x/100 for x in range(1, 100)]) for _ in range(num_classes)]

        self.samples = [[] for _ in range(len(self.thresholds))]

    def evaluate(self, gt, pred):
        for cls_idx in range(len(self.thresholds)):
            for example_idx in range(len(gt)):
                self.samples[cls_idx].append((pred[example_idx, cls_idx].item(), gt[example_idx].item() == cls_idx))

    def score(self):
        best_scores = []
        best_thresholds = []

        for cls_idx, cls_thresholds in enumerate(self.thresholds):
            # To compute scores at many thresholds, we iterate over samples from least probability to highest probability.
            # We start by computing tp/fp/fn as if the threshold is 0 (all samples predicted true).
            # Then during every iteration, we update tp/fp/fn incrementally as if the threshold were equal to the current sample's probability.
            cls_thresholds.sort()
            cls_samples = self.samples[cls_idx]
            cls_samples.sort(key=lambda sample: sample[0])
            threshold_idx = 0
            tp = sum([1 for sample in cls_samples if sample[1]])
            fp = sum([1 for sample in cls_samples if not sample[1]])
            fn = 0

            if tp == 0:
                # Skip this class if there are no positive examples.
                best_thresholds.append(0.5)
                continue

            best_score = 0
            best_threshold = 0.5

            def get_f1():
                if tp == 0:
                    return 0
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                if precision + recall < 0.01:
                    return 0
                return 2 * precision * recall / (precision + recall)

            for sample_prob, sample_match in cls_samples:
                if threshold_idx >= len(cls_thresholds):
                    break
                while threshold_idx < len(cls_thresholds) and sample_prob > cls_thresholds[threshold_idx]:
                    cur_score = get_f1()
                    if cur_score > best_score:
                        best_score = cur_score
                        best_threshold = cls_thresholds[threshold_idx]
                    threshold_idx += 1

                # Update tp/fp/fn to account for the threshold passing above this sample's probability.
                if sample_match:
                    tp -= 1
                    fn += 1
                else:
                    fp -= 1

            best_scores.append(best_score)
            best_thresholds.append(best_threshold)

            if self.detail_func:
                self.detail_func('{}_{}'.format(self.task['name'], self.task['categories'][cls_idx]), best_score)

        if len(best_scores) == 0:
            return 0.0, best_thresholds

        return np.mean(best_scores), best_thresholds

evaluator_registry = {
    'segment': {
        'accuracy': SegmentAccuracyEvaluator,
        'logprob': SegmentLogProbEvaluator,
        'f1': SegmentF1Evaluator,
        'mIou': SegmentMIOUEvaluator,
    },
    'bin_segment': {
        'accuracy': BinSegmentAccuracyEvaluator,
        'f1': BinSegmentF1Evaluator,
    },
    'regress': {
        'mse': RegressMseEvaluator,
        'l1': RegressL1Evaluator,
    },
    'detect': {
        'f1': DetectF1Evaluator,
    },
    'instance': {
        'f1': DetectF1Evaluator,
    },
    'classification': {
        'accuracy': ClassificationAccuracyEvaluator,
        'f1': ClassificationF1Evaluator,
    },
    'multi-label-classification': {
        'accuracy': ClassificationAccuracyEvaluator,
    },
}

# Convert masks (in format produced by Mask R-CNN) to polygons.
# Returns list, where each element is list of polygons corresponding to that mask.
# Most masks will correspond to one polygon, but sometimes there may be multiple if the mask has several disjoint components.
# The masks and boxes arguments should be numpy arrays.
def polygonize_masks(masks, boxes):
    polygons = [[] for _ in masks]

    for box_idx, (mask, box) in enumerate(zip(masks, boxes)):
        mask = (mask > 0.5).astype(np.uint8)

        # We use rasterio for initial polygonization and rastachimp for polygon simplification.
        shapes = list(rasterio.features.shapes(mask))

        # Should have at least one shape for foreground and one for background.
        # If not, then it means there's no foreground (or background).
        if len(shapes) < 2:
            continue

        # Helper to convert mask coordinates (which is within the predicted box) to image coordinates.
        def mask_to_image(x, y):
            # Rescale.
            x = x * (box[2] - box[0]) / mask.shape[1]
            y = y * (box[3] - box[1]) / mask.shape[0]
            # Include offset.
            x += box[0]
            y += box[1]
            return int(x), int(y)

        for (shp, value) in shapes:
            # Discard shape corresponding to the background.
            if value != 1:
                continue

            # Convert to shapely and simplify.
            shp = shapely.geometry.shape(shp)
            shp = shp.simplify(tolerance=2)

            # Convert to our format, i.e., list of rings.
            # First ring is exterior ring and rest (if any) are interior holes.
            exterior = [mask_to_image(x, y) for x, y in shp.exterior.coords]
            interiors = [[mask_to_image(x, y) for x, y in interior.coords] for interior in shp.interiors]
            coords = [exterior] + interiors
            polygons[box_idx].append(coords)

    return polygons

# Convert segmentation class map to colors.
def segmentation_mask_to_color(task, label_im, gt=None):
    if task['type'] == 'segment':
        vis = np.zeros((label_im.shape[0], label_im.shape[1], 3), dtype='uint8')
        for cls, color in enumerate(task['colors']):
            vis[label_im == cls] = color

        if task.get('BackgroundInvalid', False) and gt:
            # Mask out pixels in label_im that are background in the ground truth labels.
            vis[gt == 0] = task['colors'][0]

    elif task['type'] == 'bin_segment':
        if label_im.dtype != bool:
            label_im = label_im > 0.5

        vis = np.zeros((label_im.shape[1], label_im.shape[2], 3), dtype='uint8')
        for cls, color in enumerate(task['colors']):
            vis[label_im[cls, :, :]] = color

    return vis

def segmentation_centerline(task, label_im, image, gt=None):
    if task['type'] == 'bin_segment':
        if label_im.dtype != bool:
            # convert to boolean array
            label_im = label_im > 0.5 # shape (6, 512, 512) with 6 classes
            
        vis = np.zeros((label_im.shape[1], label_im.shape[2], 3), dtype='uint8')

        for cls in range(len(label_im)):  # 6 classes for sentinel-2 and 47 classes for aerial
            if cls == 26:
                # Tạo một mask nhị phân từ `label_im`
                # print(label_im[26])
                
                mask = label_im[cls] > 0  # Chuyển lớp hiện tại thành nhị phân
                
                # Chuyển đổi từ đường viền thành đường trung tâm (skeleton)
                centerline = skeletonize(mask)
                
                kernel = np.ones((3, 3), np.uint8)  # Kích thước kernel xác định độ dày (3x3 là thêm độ dày 1 pixel)
                dilated_centerline = cv2.dilate(centerline.astype(np.uint8), kernel, iterations=1)

                # Vẽ đường giãn nở lên ảnh
                coords = np.column_stack(np.where(dilated_centerline))
                color = [min(c + 50, 255) for c in task['colors'][cls]]
                for coord in coords:
                    image[coord[0], coord[1]] = task['colors'][cls]

        for cls, color in enumerate(task['colors']):
            vis[label_im[cls, :, :]] = color

    return image

def visualize_outputs(task, image, outputs, targets=None, return_vis=False, return_out=False, evaluator_params=None):
    '''
    Returns output visualization and/or raw outputs given the task dict, input image, and outputs.
    Also returns gt visualization if targets is provided.
    '''
    task_type = task['type']

    # These will be overwritten with label visualizations if the task supports it and return_vis=True.
    vis_output = None
    vis_gt = None

    # These will be overwritten with the actual outputs if return_out=True.
    raw_outputs = None
    raw_output_format = None

    if task_type == 'segment':
        if evaluator_params:
            # Use user-provided score thresholds for each class.
            # If multiple classes have probability exceeding threshold at a pixel,
            # the pixel will end up colored based on the last class.
            pred_probs = outputs.cpu().numpy()
            pred_cls = np.zeros(pred_probs.shape[1:3], dtype=np.uint8)
            for cls_id, threshold in enumerate(evaluator_params):
                pred_cls[pred_probs[cls_id, :, :] > threshold] = cls_id
        else:
            pred_cls = outputs.argmax(dim=0).cpu().numpy()

        if return_vis:
            vis_output = segmentation_mask_to_color(task, pred_cls)
            if targets:
                vis_gt = segmentation_mask_to_color(task, targets['im'].numpy())

        if return_out:
            raw_outputs = pred_cls.astype(np.uint8)
            raw_output_format = 'png'

    elif task_type == 'bin_segment':
        cur_outputs = outputs.cpu().numpy()
        if evaluator_params:
            cur_outputs_bin = np.stack([
                cur_outputs[channel_idx, :, :] > evaluator_params[channel_idx]
                for channel_idx in range(cur_outputs.shape[0])
            ], axis=0)
        else:
            cur_outputs_bin = cur_outputs > 0.5

        if return_vis:
            vis_output = segmentation_centerline(task, cur_outputs_bin, image)
            if targets:
                vis_gt = segmentation_centerline(task, targets['im'].numpy() > 0.5)

        if return_out:
            enc = satlas.util.encode_multiclass_binary(cur_outputs_bin.transpose(1, 2, 0))
            raw_outputs = enc
            raw_output_format = 'npy'

    elif task_type == 'regress':
        output = np.clip(outputs.cpu().numpy(), 0, 255).astype(np.uint8)

        if return_vis:
            vis_output = output
            if targets:
                vis_gt = targets['im'].numpy()

        if return_out:
            raw_outputs = output
            raw_output_format = 'png'

    elif task_type == 'detect':
        output_boxes = outputs['boxes'].long().cpu().numpy()
        output_scores = outputs['scores'].cpu().numpy()
        output_categories = outputs['labels'].cpu().numpy()

        if evaluator_params:
            output_score_thresholds = np.array([evaluator_params[category_id] for category_id in output_categories], dtype=np.float32)
            wanted = output_scores >= output_score_thresholds

            output_boxes = output_boxes[wanted, :]
            output_scores = output_scores[wanted]
            output_categories = output_categories[wanted]

        if return_vis:
            def get_color(category_id):
                if 'colors' in task:
                    return task['colors'][category_id]
                return [255, 255, 0]

            vis_output = image.copy()

            for box_idx, box in enumerate(output_boxes):
                color = get_color(outputs['labels'][box_idx].item())
                left = satlas.util.clip(box[0], 0, vis_output.shape[1])
                top = satlas.util.clip(box[1], 0, vis_output.shape[0])
                right = satlas.util.clip(box[2], 0, vis_output.shape[1])
                bottom = satlas.util.clip(box[3], 0, vis_output.shape[0])
                vis_output[top:bottom, left:left+2, :] = color
                vis_output[top:bottom, right-2:right, :] = color
                vis_output[top:top+2, left:right, :] = color
                vis_output[bottom-2:bottom, left:right, :] = color

            if targets:
                gt_boxes = targets['boxes'].long().numpy()
                vis_gt = image.copy()
                for box_idx, box in enumerate(gt_boxes):
                    color = get_color(targets['labels'][box_idx].item())
                    left = satlas.util.clip(box[0], 0, vis_gt.shape[1])
                    top = satlas.util.clip(box[1], 0, vis_gt.shape[0])
                    right = satlas.util.clip(box[2], 0, vis_gt.shape[1])
                    bottom = satlas.util.clip(box[3], 0, vis_gt.shape[0])
                    vis_gt[top:bottom, left:left+2, :] = color
                    vis_gt[top:bottom, right-2:right, :] = color
                    vis_gt[top:top+2, left:right, :] = color
                    vis_gt[bottom-2:bottom, left:right, :] = color

        if return_out:
            raw_outputs = [(
                int(box[0]),
                int(box[1]),
                int(box[2]),
                int(box[3]),
                task['categories'][output_categories[i]],
                float(output_scores[i]),
            ) for i, box in enumerate(output_boxes)]
            raw_output_format = 'json'

    elif task_type == 'instance':
        outputs = apply_nms(outputs, iou_threshold=0.3)
        masks = []
        for mask in outputs['masks']:
            mask = mask.cpu().numpy().squeeze() > 0.5
            simplified = simplify_polygon(mask, tolerance=0.5)
            if simplified:
                regularized = regularize_polygon(simplified, grid_size=5.0, angle_tolerance=0.5)
                masks.append(regularized)
        output_boxes = outputs['boxes'].long().cpu().numpy()
        output_scores = outputs['scores'].cpu().numpy()
        output_categories = outputs['labels'].cpu().numpy()
        output_masks = outputs['masks'].cpu().numpy()[:, 0, :, :]

        if evaluator_params:
            output_score_thresholds = np.array([evaluator_params[category_id] for category_id in output_categories], dtype=np.float32)
            wanted = output_scores >= output_score_thresholds

            output_boxes = output_boxes[wanted, :]
            output_scores = output_scores[wanted]
            output_categories = output_categories[wanted]
            output_masks = output_masks[wanted, :, :]

        # Polygonize the masks.
        output_polygons = polygonize_masks(output_masks, output_boxes)

        if return_vis:
            def get_color(category_id):
                if 'colors' in task:
                    return task['colors'][category_id]
                return [255, 255, 0]

            vis_output = image.copy()

            # Draw the polygons on vis_output.
            # We first draw it on line_im, then dilate line_im, then apply it on vis_output.
            for box_idx, polygons in enumerate(output_polygons):
                color = get_color(outputs['labels'][box_idx].item())
                for coords in polygons:
                    exterior = np.array(coords[0], dtype=np.int32)
                    rows, cols = skimage.draw.polygon_perimeter(exterior[:, 1], exterior[:, 0], shape=(vis_output.shape[0], vis_output.shape[1]))
                    vis_output[rows, cols, :] = color

            if targets:
                gt_boxes = targets['boxes'].long().numpy()
                gt_masks = targets['masks'].numpy()
                vis_gt = image.copy()
                for box_idx, mask in enumerate(gt_masks):
                    color = get_color(targets['labels'][box_idx].item())
                    vis_gt[mask > 0, :] = color

        if return_out:
            raw_outputs = []
            for box_idx, polygons in enumerate(output_polygons):
                category_name = task['categories'][output_categories[box_idx]]
                score = float(output_scores[box_idx])
                for coords in polygons:
                    raw_outputs.append([
                        'fake_polygon_id',
                        coords,
                        category_name,
                        score,
                        {}, # no properties
                    ])
            raw_output_format = 'json'

    elif task_type == 'classification':
        if return_out:
            category_id = torch.argmax(outputs).item()
            raw_outputs = "{}\n".format(category_id)
            raw_output_format = "txt"

    return vis_output, vis_gt, raw_outputs, raw_output_format

def simplify_polygon(mask, tolerance=2.0):
    """Simplify polygon using Douglas-Peucker Algorithm."""
    contours = find_contours(mask, 0.5)
    polygons = []
    for contour in contours:
        # Bỏ qua contour không đủ điểm
        if len(contour) < 4:
            continue
        poly = Polygon(contour[:, ::-1])  # Đảo ngược trục (y, x) -> (x, y)
        if poly.is_valid and poly.area > 100:  # Lọc polygon nhỏ
            simplified_poly = poly.simplify(tolerance, preserve_topology=True)
            if simplified_poly.is_valid and simplified_poly.area > 100:
                polygons.append(simplified_poly)
    return unary_union(polygons) if polygons else None

def apply_nms(out, iou_threshold=0.5):
    """
    Áp dụng Non-Maximum Suppression (NMS) cho các đa giác trong output.
    """
    boxes = out['boxes']  # (N, 4) - bounding boxes
    scores = out['scores']  # (N,) - confidence scores
    masks = out['masks']
    # Áp dụng NMS
    keep_indices = nms(boxes, scores, iou_threshold)

    # Lọc các đối tượng được giữ lại
    out['boxes'] = boxes[keep_indices]
    out['labels'] = out['labels'][keep_indices]
    out['scores'] = scores[keep_indices]
    out['masks'] = masks[keep_indices]

    return out

def regularize_polygon(polygon, grid_size=1.0, angle_tolerance=5.0):
    """Regularize the edges of a polygon to align with straight lines or right angles."""
    if polygon.is_empty or not isinstance(polygon, Polygon):
        return polygon  # Skip empty or non-polygon geometries

    def snap_to_grid(coord, grid_size):
        return round(coord / grid_size) * grid_size

    def snap_coords_to_grid(coords, grid_size):
        return [(snap_to_grid(x, grid_size), snap_to_grid(y, grid_size)) for x, y in coords]

    snapped_coords = snap_coords_to_grid(list(polygon.exterior.coords), grid_size)

    def regularize_angle(p1, p2, p3, angle_tolerance):
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)
        angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
        angle = np.degrees(angle) % 180

        if abs(angle - 90) <= angle_tolerance or abs(angle - 180) <= angle_tolerance:
            return p2  # Already aligned
        else:
            avg_x = (p1[0] + p3[0]) / 2
            avg_y = (p1[1] + p3[1]) / 2
            return (snap_to_grid(avg_x, grid_size), snap_to_grid(avg_y, grid_size))

    regularized_coords = []
    coords = snapped_coords
    for i in range(len(coords) - 2):
        p1, p2, p3 = coords[i], coords[i + 1], coords[i + 2]
        p2 = regularize_angle(p1, p2, p3, angle_tolerance)
        regularized_coords.append(p2)

    regularized_coords.append(regularized_coords[0])  # Close the polygon
    return Polygon(regularized_coords)

def evaluate(config, model, device, loader, half_enabled=False, vis_dir=None, probs_dir=None, out_dir=None, print_details=True, evaluator_params_list=None):
    # Is a specific task chosen for evaluation?
    selected_task = config.get('EvaluateTask', None)

    with torch.no_grad():
        all_losses = [[] for _ in config['Tasks']]

        # For computing detailed per-class scores and such, if print_details is set.
        details = {}
        def add_detail(k, score):
            if k not in details:
                details[k] = []
            details[k].append(score)
        detail_func = None
        if print_details:
            detail_func = add_detail

        # Initialize evaluators.
        evaluators = []
        for task_idx, spec in enumerate(config['Tasks']):
            task_name = spec['Name']
            task = spec['Task']
            task['name'] = task_name # Since we only pass the dict, not the name, to score functions.
            task_type = task['type']
            metric = spec.get('Metric', 'accuracy')

            if evaluator_params_list:
                evaluator_params = evaluator_params_list[task_idx]
            else:
                evaluator_params = None

            evaluator_cls = evaluator_registry[task_type][metric]
            evaluators.append(evaluator_cls(task=task, spec=spec, detail_func=detail_func, params=evaluator_params))

        if print_details:
            # Enable tqdm progress when details is on.
            loader = tqdm.tqdm(loader)

        for images, targets, info in loader:
            gpu_images = [image.to(device).float()/255 for image in images]
            gpu_targets = [
                [{k: v.to(device) for k, v in target_dict.items()} for target_dict in cur_targets]
                for cur_targets in targets
            ]

            with torch.cuda.amp.autocast(enabled=half_enabled):
                outputs, losses = model(gpu_images, gpu_targets, selected_task=selected_task)

            for task_idx, spec in enumerate(config['Tasks']):
                all_losses[task_idx].append(losses[task_idx].item())

                task_name = spec['Name']
                task = spec['Task']
                task_type = task['type']

                if selected_task and selected_task != task_name:
                    continue

                # Save output probabilities.
                # Do this before evaluator because for some task types,
                # we may skip the image based on gt validity.
                if probs_dir:
                    for image_idx, image in enumerate(images):
                        example_id = info[image_idx]['example_id']

                        if task_type in ['segment', 'bin_segment']:
                            cur_outputs = outputs[task_idx][image_idx, :, :, :].cpu().numpy()
                            np.save(os.path.join(probs_dir, '{}_{}.npy'.format(example_id, task_name)), cur_outputs)

                if task_type in ['segment', 'bin_segment', 'regress']:
                    valid = torch.stack([target[task_idx]['valid'] != 0 for target in targets], dim=0)
                    if torch.count_nonzero(valid) == 0:
                        continue

                    valid_im = torch.stack([target[task_idx]['valid_im'] for i, target in enumerate(gpu_targets) if valid[i]], dim=0)
                    gt = torch.stack([target[task_idx]['im'] for i, target in enumerate(gpu_targets) if valid[i]], dim=0)
                    pred = torch.stack([output for i, output in enumerate(outputs[task_idx]) if valid[i]], dim=0)

                    evaluators[task_idx].evaluate(valid_im, gt, pred)

                elif task_type == 'detect' or task_type == 'instance':
                    gt = []
                    pred = []
                    valid = torch.stack([target[task_idx]['valid'] != 0 for target in targets], dim=0)
                    for target, output in zip(gpu_targets, outputs[task_idx]):
                        if not target[task_idx]['valid']:
                            continue
                        gt.append({
                            'boxes': target[task_idx]['boxes'].cpu(),
                            'labels': target[task_idx]['labels'].cpu(),
                            'masks': target[task_idx]['masks'].cpu(),
                        })
                        pred.append({
                            'boxes': output['boxes'].cpu(),
                            'labels': output['labels'].cpu(),
                            'scores': output['scores'].cpu(),
                            'masks': output['masks'].cpu()
                        })
                    evaluators[task_idx].evaluate(gt, pred)

                elif task_type == 'classification':
                    valid = torch.stack([target[task_idx]['valid'] for target in gpu_targets], dim=0) > 0
                    if torch.count_nonzero(valid) == 0:
                        continue

                    gt = torch.cat([target[task_idx]['label'] for i, target in enumerate(gpu_targets) if valid[i]], dim=0)
                    pred = torch.stack([output for i, output in enumerate(outputs[task_idx]) if valid[i]], dim=0)
                    evaluators[task_idx].evaluate(gt, pred)

                elif task_type == 'multi-label-classification':
                    valid = torch.tensor([target[task_idx]['valid'] for target in gpu_targets]) > 0
                    if torch.count_nonzero(valid) == 0:
                        continue

                    gt = torch.cat([target[task_idx]['labels'] for i, target in enumerate(gpu_targets) if valid[i]], dim=0).to(torch.float32)
                    pred = torch.stack([output > 0.5 for i, output in enumerate(outputs[task_idx]) if valid[i]], dim=0)
                    evaluators[task_idx].evaluate(gt, pred)

                evaluator_params = None
                if evaluator_params_list:
                    evaluator_params = evaluator_params_list[task_idx]

                # Visualize outputs.
                if vis_dir or out_dir:
                    for image_idx, image in enumerate(images):
                        example_id = info[image_idx]['example_id']
                        save_prefix = '{}_{}'.format(example_id, task_name)
                        image = image.numpy().transpose(1, 2, 0)[:, :, 0:3]

                        vis_output, vis_gt, raw_outputs, raw_output_format = visualize_outputs(
                            task=task,
                            image=image,
                            outputs=outputs[task_idx][image_idx],
                            targets=targets[image_idx][task_idx],
                            return_vis=vis_dir is not None,
                            return_out=out_dir is not None,
                            evaluator_params=evaluator_params,
                        )

                        if vis_output is not None:
                            # skimage.io.imsave(os.path.join(vis_dir, '{}_im.png'.format(save_prefix)), image)
                            skimage.io.imsave(os.path.join(vis_dir, '{}_out.png'.format(save_prefix)), vis_output)
                        if vis_gt is not None:
                            skimage.io.imsave(os.path.join(vis_dir, '{}_gt.png'.format(save_prefix)), vis_gt)

                        if raw_outputs is not None:
                            if raw_output_format == 'png':
                                skimage.io.imsave(os.path.join(out_dir, '{}.png'.format(save_prefix)), raw_outputs)
                            elif raw_output_format == 'npy':
                                np.save(os.path.join(out_dir, '{}.npy'.format(save_prefix)), raw_outputs)
                            elif raw_output_format == 'json':
                                with open(os.path.join(out_dir, '{}.json'.format(save_prefix)), 'w') as f:
                                    json.dump(raw_outputs, f)
                            elif raw_output_format == 'txt':
                                with open(os.path.join(out_dir, '{}.txt'.format(save_prefix)), 'w') as f:
                                    f.write(raw_outputs)

        avg_loss = np.mean([loss for losses in all_losses for loss in losses])
        avg_losses = [np.mean(losses) for losses in all_losses]
        precisions = []
        recalls = []
        avg_scores = []
        final_evaluator_params = []
        for evaluator in evaluators:
            score, params, precision, recall = evaluator.score()
            precisions.append(precision)
            recalls.append(recall)
            avg_scores.append(score)
            final_evaluator_params.append(params)

    if print_details:
        for k, v in details.items():
            print(k, np.mean(v))

    return avg_loss, avg_losses, avg_scores, final_evaluator_params, precisions, recalls
