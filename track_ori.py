import sys
sys.path.insert(0, './yolov5')

from yolov5.utils.google_utils import attempt_download
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, \
    check_imshow
from yolov5.utils.torch_utils import select_device, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from collections import deque
import numpy as np


palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
point1, point2, point3, point4 = 0, 0, 0, 0

up_status, down_staus = False, False
def draw_rectangle(event, x, y, flags, im0):
    global ix, iy
    global up_status, down_staus
    global point1, point2, point3, point4
    if event == cv2.EVENT_LBUTTONDOWN:
        print('buttom down')
        ix, iy = x, y
        point1, point2 = x, y
        up_status = True
    if event == cv2.EVENT_LBUTTONUP:
        print('bottom up')
        # cv2.rectangle(im0, (ix, iy), (x, y), (0, 0, 255), 4)
        point3, point4 = x, y
        down_staus = True

def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, track_status, pre_alone_num, identities=None, offset=(0, 0)):
    global up_status, down_staus
    global point1, point2, point3, point4
    pre_frame_id = track_status['pre_frame_id']
    current_frame_id = []
    center_points = []
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        center_points.append([cx, cy])
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        current_frame_id.append(id)
        if id not in track_status['center_points'].keys():
            track_status['center_points'][id] = deque(maxlen=20)
            track_status['center_points'][id].append([cx, cy])
        else:
            track_status['center_points'][id].append([cx, cy])
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
        #根据预测的中心点画轨迹
        if len(track_status['center_points'][id]) > 2:
            start_point = track_status['center_points'][id][0]
            for i in range(1, len(track_status['center_points'][id])):
                points = track_status['center_points'][id][i]
                cv2.line(img, tuple(start_point), tuple(points), color, 4)
                start_point = points

    center_points = np.array(center_points)
    n, _ = center_points.shape
    center_points_1 = np.repeat(center_points, n, axis=0)
    center_points_2 = np.tile(center_points, [n, 1])

    distance = np.sqrt(np.sum(np.power(center_points_1 - center_points_2, 2), axis=1))
    distance = distance.reshape(n, n)
    distance[range(n), range(n)] = 1000000  #将对角线元素设置为很大

    # print('distance: ', distance)
    # print('*'*100)
    judge_array = np.full((n, n), 100) # 判断像素距离大于100的，为单独行走的人
    alone_num = np.sum(np.sum(distance > judge_array, 1) == n)


    text5 = 'Number of people walking alone: {}; ' \
            'Number of people walking in groups: {}'.format(alone_num, n - alone_num)

    if pre_frame_id:
        exist_id = set(pre_frame_id) & set(current_frame_id)
        leaving_num = len(pre_frame_id) - len(exist_id)
        coming_num = len(current_frame_id) - len(exist_id)
        text3 = 'Number of people coming in: {}'.format(coming_num)
        text4 = 'Number of people leaving: {}'.format(leaving_num)

    else:
        text3 = 'Number of people coming in: {}'.format(len(current_frame_id))
        text4 = 'Number of people leaving: {}'.format(0)
        pre_alone_num = alone_num


    if abs(pre_alone_num - alone_num) > 3:
        text7 = 'Crowd Generation / Crowd destruction Occurs!'
    else:
        text7 = 'No Crowd Generation / Crowd destruction Occurs!'
    color7 = compute_color_for_labels(7)
    t_size = cv2.getTextSize(text7, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
    cv2.rectangle(img, (100, 700 + t_size[1] + 4), (100 + t_size[0] + 3, 700 + t_size[1] + 4), color7, -1)
    cv2.putText(img, text7, (100, 700 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

    track_status['pre_frame_id'] = current_frame_id

    text1 = 'Current Frame People Num: {}'.format(len(identities))
    color1 = compute_color_for_labels(1)
    t_size = cv2.getTextSize(text1, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
    cv2.rectangle(img, (100, 100 + t_size[1] + 4), (100 + t_size[0] + 3, 100 + t_size[1] + 4), color1, -1)
    cv2.putText(img, text1, (100, 100 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

    text2 = 'Tracking Total People Num: {}'.format(len(track_status['center_points'].keys()))
    color2 = compute_color_for_labels(2)
    t_size = cv2.getTextSize(text2, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
    cv2.rectangle(img, (100, 200 + t_size[1] + 4), (100 + t_size[0] + 3, 200 + t_size[1] + 4), color2, -1)
    cv2.putText(img, text2, (100, 200 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

    color3 = compute_color_for_labels(3)
    t_size = cv2.getTextSize(text3, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
    cv2.rectangle(img, (100, 300 + t_size[1] + 4), (100 + t_size[0] + 3, 300 + t_size[1] + 4), color3, -1)
    cv2.putText(img, text3, (100, 300 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

    color4 = compute_color_for_labels(4)
    t_size = cv2.getTextSize(text4, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
    cv2.rectangle(img, (100, 400 + t_size[1] + 4), (100 + t_size[0] + 3, 400 + t_size[1] + 4), color4, -1)
    cv2.putText(img, text4, (100, 400 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

    color5 = compute_color_for_labels(5)
    t_size = cv2.getTextSize(text5, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
    cv2.rectangle(img, (100, 500 + t_size[1] + 4), (100 + t_size[0] + 3, 500 + t_size[1] + 4), color5, -1)
    cv2.putText(img, text5, (100, 500 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    print(up_status, down_staus, point1, point2, point3, point4)
    print('*'*100)
    if up_status and down_staus:
        cv2.rectangle(img, (point1, point2), (point3, point4), color5, 4)
        # up_status, down_staus = False, False
        m1 = center_points >= [point1, point2]
        m2 = center_points <= [point3, point4]
        m = np.hstack((m1, m2))
        num = np.sum(np.sum(m, axis=1) == 4)
        text6 = 'There are {} people within the rectangle'.format(num)
        color6 = compute_color_for_labels(6)
        t_size = cv2.getTextSize(text6, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (100, 600 + t_size[1] + 4), (100 + t_size[0] + 3, 600 + t_size[1] + 4), color6, -1)
        cv2.putText(img, text6, (100, 600 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    pre_alone_num = alone_num
    return img, track_status, pre_alone_num


def detect(opt):
    out, source, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, imgsz = \
        opt.output, opt.source, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.save_vid, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(yolo_weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    save_path = str(Path(out))
    txt_path = str(Path(out)) + '/results.txt'

    track_status = {'center_points': {}, 'pre_frame_id': []}
    cv2.namedWindow('image')
    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        pre_alone_num = 0
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            global im0
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                xywh_bboxs = []
                confs = []

                # Adapt detections to deep sort input format
                for *xyxy, conf, cls in det:
                    # to deep sort format
                    x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                    xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                    xywh_bboxs.append(xywh_obj)
                    confs.append([conf.item()])

                xywhs = torch.Tensor(xywh_bboxs)
                confss = torch.Tensor(confs)

                # pass detections to deepsort
                outputs = deepsort.update(xywhs, confss, im0)

                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    _, track_status, pre_alone_num = draw_boxes(im0, bbox_xyxy, track_status, pre_alone_num, identities)
                    # to MOT format
                    tlwh_bboxs = xyxy_to_tlwh(bbox_xyxy)

                    # Write MOT compliant results to file
                    if save_txt:
                        for j, (tlwh_bbox, output) in enumerate(zip(tlwh_bboxs, outputs)):
                            bbox_top = tlwh_bbox[0]
                            bbox_left = tlwh_bbox[1]
                            bbox_w = tlwh_bbox[2]
                            bbox_h = tlwh_bbox[3]
                            identity = output[-1]
                            with open(txt_path, 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_top,
                                                            bbox_left, bbox_w, bbox_h, -1, -1, -1, -1))  # label format

            else:
                deepsort.increment_ages()

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if show_vid:
                # cv2.imshow(p, im0)
                cv2.setMouseCallback('image', draw_rectangle, im0)
                cv2.imshow('image', im0)

                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

    if save_txt or save_vid:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', type=str, default='yolov5/weights/yolov5s.pt', help='model.pt path')
    parser.add_argument('--deep_sort_weights', type=str, default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7', help='ckpt.t7 path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='./people.mp4', help='source')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.2, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', default=[0], type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)

    with torch.no_grad():
        detect(args)
