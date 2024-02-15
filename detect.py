# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path
from PIL import Image
import pytesseract
import torch
import time

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

#file = open('xys.txt', "w")
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/Tesseract'

def detect_n_digit_number(img , part):
    # Open the image file

    # Perform any necessary preprocessing (resizing, filtering, etc.)
    # For example, resize the image to improve OCR accuracy
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img =  img.crop(part)
    img = img.resize((400, 200))

    # Use Tesseract to extract text from the image
    extracted_text = pytesseract.image_to_string(img, config='--psm 6')

    # Search for 4-digit numbers in the extracted text
    import re
    four_digit_numbers = re.findall(r'\b\d+\b', extracted_text)

    return four_digit_numbers

@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/data.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    
    # Read the template images
    template = cv2.imread("temp_pics/partbar.png")
    template_sneaker = cv2.imread('/Users/mehravehahmadi/yolov5/temp_pics/sneaker_temp.jpg')  # Replace with the path to your template image
    template_mag = cv2.imread('/Users/mehravehahmadi/yolov5/temp_pics/mag_temp.jpg')  # Replace with the path to your template image
    template_2x = cv2.imread('/Users/mehravehahmadi/yolov5/temp_pics/2x_temp.png')  # Replace with the path to your template image
    template_sk = cv2.imread('/Users/mehravehahmadi/yolov5/temp_pics/sk_temp.png')  # Replace with the path to your template image
    template_jet = cv2.imread('/Users/mehravehahmadi/yolov5/temp_pics/jet_temp.jpg')  # Replace with the path to your template image

    # Read the video
    #cap = cv2.VideoCapture(source)

    # Get the height and width of the templates
    h, w = template.shape[:2]
    template_height_sneaker, template_width_sneaker, _ = template_sneaker.shape
    template_height_mag, template_width_mag, _ = template_mag.shape
    template_height_2x, template_width_2x , _ = template_2x.shape
    template_height_sk, template_width_sk , _ = template_sk.shape
    template_height_jet, template_width_jet, _ = template_jet.shape

    
    is_match = False
    is_match2 = False
    
    matchs = 0
    sneakers = 0
    magnets = 0
    twoxs = 0
    sks = 0
    jets = 0

    coin = 0
    score = 0
    max1 = 0
    min1 = 100000
    detected_count = 0
    last_line = 'CENTER'
    line_changes_count = 0
    jumped = False
    is_power_jumper = False
    power_jumper_count = 0
    is_box = False
    box_count = 0
    jump_count = 0
    frame_counter = 0
    start_power_time = False

    device = torch.device("mps")
    print("Running on:", device)

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        
        st = False
        detect_num = 0

         # Read a frame from the video
        ret, frame = vid_cap.read()

        if not ret:
            break
        
        if start_power_time :
            frame_counter +=1
        if frame_counter  == 100:
            start_power_time = False
            frame_counter = 0

        #score = detect_n_digit_number(frame , (1160,30,1420,125))
        #if len(detect_n_digit_number(frame , (1220,150,1350,250)))>0:
        #    coin = detect_n_digit_number(frame , (1220,150,1350,250))
        #print(coin)
        frame = frame[1750: 1950 , 200:1200]
        frame1 = frame[30: 100 , 380:420]
        frame2 = frame[30: 100 , 866:906]
        frame3 = frame[10:130 , 10:130]
        frame4 = frame[10:130 , 496:616]

        result1 = cv2.matchTemplate(frame1, template, cv2.TM_CCOEFF_NORMED)
        min_val1, max_val1, min_loc1, max_loc1 = cv2.minMaxLoc(result1)

        result2 = cv2.matchTemplate(frame2, template, cv2.TM_CCOEFF_NORMED)
        min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(result2)

        result_sneaker = cv2.matchTemplate(frame3, template_sneaker, cv2.TM_CCOEFF_NORMED)
        min_val_sneaker, max_val_sneaker, min_loc_sneaker, max_loc_sneaker = cv2.minMaxLoc(result_sneaker)
        
        result_sneaker2 = cv2.matchTemplate(frame4, template_sneaker, cv2.TM_CCOEFF_NORMED)
        min_val_sneaker2, max_val_sneaker2, min_loc_sneaker2, max_loc_sneaker2 = cv2.minMaxLoc(result_sneaker2)
       
        result_meg = cv2.matchTemplate(frame3, template_mag, cv2.TM_CCOEFF_NORMED)
        min_val_mag, max_val_mag, min_loc_mag, max_loc_mag = cv2.minMaxLoc(result_meg)
        
        result_meg2 = cv2.matchTemplate(frame4, template_mag, cv2.TM_CCOEFF_NORMED)
        min_val_mag2, max_val_mag2, min_loc_mag2, max_loc_mag2 = cv2.minMaxLoc(result_meg2)

        result_2x = cv2.matchTemplate(frame3, template_2x, cv2.TM_CCOEFF_NORMED)
        min_val_2x, max_val_2x, min_loc_2x, max_loc_2x = cv2.minMaxLoc(result_2x)

        result_2x2 = cv2.matchTemplate(frame4, template_2x, cv2.TM_CCOEFF_NORMED)
        min_val_2x2, max_val_2x2, min_loc_2x2, max_loc_2x2 = cv2.minMaxLoc(result_2x2)
        
        result_sk = cv2.matchTemplate(frame3, template_sk, cv2.TM_CCOEFF_NORMED)
        min_val_sk, max_val_sk, min_loc_sk, max_loc_sk = cv2.minMaxLoc(result_sk)

        result_sk2 = cv2.matchTemplate(frame4, template_sk, cv2.TM_CCOEFF_NORMED)
        min_val_sk2, max_val_sk2, min_loc_sk2, max_loc_sk2 = cv2.minMaxLoc(result_sk2)
       
        result_jet = cv2.matchTemplate(frame3, template_jet, cv2.TM_CCOEFF_NORMED)
        min_val_jet, max_val_jet, min_loc_jet, max_loc_jet = cv2.minMaxLoc(result_jet)

        result_jet2 = cv2.matchTemplate(frame4, template_jet, cv2.TM_CCOEFF_NORMED)
        min_val_jet2, max_val_jet2, min_loc_jet2, max_loc_jet2 = cv2.minMaxLoc(result_jet2)

        if max_val1 > 0.79 and not is_match:
            
            #Sneaker##############################
            # Draw a rectangle around the matched region
            top_left = max_loc1
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(frame1, top_left, bottom_right, (0, 255, 0), 2)

            if max_val_sneaker > 0.5 :
            # Draw a bounding box around the detected region
                top_left = max_loc_sneaker
                bottom_right = (top_left[0] + template_width_sneaker, top_left[1] + template_height_sneaker)
                cv2.rectangle(frame3, top_left, bottom_right, (255, 0, 225), 10)
                sneakers+=1
                #print('match_snk1')
            
            #magnet##############################
            if max_val_mag > 0.5 :
            # Draw a bounding box around the detected region
                top_left = max_loc_mag
                bottom_right = (top_left[0] + template_width_mag, top_left[1] + template_height_mag)
                cv2.rectangle(frame3, top_left, bottom_right, (255, 0, 225), 10)
                magnets+=1
                #print('match_mag1')

           
            #2x###################################
            if max_val_2x > 0.5:
                # Draw a bounding box around the detected region
                top_left = max_loc_2x
                bottom_right = (top_left[0] + template_width_2x, top_left[1] + template_height_2x)
                cv2.rectangle(frame3, top_left, bottom_right, (0, 255, 225), 2)
                twoxs +=1
                #print('2x1')
        
            #skate################################
            if max_val_sk > 0.75:
                # Draw a bounding box around the detected region
                top_left = max_loc_sk
                bottom_right = (top_left[0] + template_width_sk, top_left[1] + template_height_sk)
                cv2.rectangle(frame3, top_left, bottom_right, (0, 255, 0), 4)
                sks+=1
                #print('sk1')
            
            #jet###############################
            if max_val_jet > 0.5:
                # Draw a bounding box around the detected region
                top_left = max_loc_jet
                bottom_right = (top_left[0] + template_width_jet, top_left[1] + template_height_jet)
                cv2.rectangle(frame3, top_left, bottom_right, (0, 0, 225), 2)
                jets+=1
                #print('jet1')
            
            is_match = True
            matchs+=1
        
        if max_val1 <= 0.79:
            is_match = False


        if max_val2 > 0.79 and not is_match2:
            
            #Sneaker##############################
            # Draw a rectangle around the matched region
            top_left = max_loc2
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(frame2, top_left, bottom_right, (0, 255, 0), 2)
            
            if max_val_sneaker2 > 0.5 :
            # Draw a bounding box around the detected region
                top_left = max_loc_sneaker2
                bottom_right = (top_left[0] + template_width_sneaker, top_left[1] + template_height_sneaker)
                cv2.rectangle(frame4, top_left, bottom_right, (255, 0, 225), 10)
                sneakers+=1
                #print('match_snk2')


            #magnet##############################
            if max_val_mag2 > 0.5 :
            # Draw a bounding box around the detected region
                top_left = max_loc_mag2
                bottom_right = (top_left[0] + template_width_mag, top_left[1] + template_height_mag)
                cv2.rectangle(frame4, top_left, bottom_right, (255, 0, 225), 10)
                magnets+=1
                #print('match_mag2')
           
            #2x###################################
            if max_val_2x2 > 0.5:
                # Draw a bounding box around the detected region
                top_left = max_loc_2x2
                bottom_right = (top_left[0] + template_width_2x, top_left[1] + template_height_2x)
                cv2.rectangle(frame4, top_left, bottom_right, (0, 255, 225), 2)
                twoxs +=1
                #print('2x2')
            
            #skate################################
            if max_val_sk2 > 0.75:
                # Draw a bounding box around the detected region
                top_left = max_loc_sk2
                bottom_right = (top_left[0] + template_width_sk, top_left[1] + template_height_sk)
                cv2.rectangle(frame4, top_left, bottom_right, (0, 255, 0), 4)
                sks+=1
                #print('sk2')
            
            #jet###############################
            if max_val_jet2 > 0.5:
                # Draw a bounding box around the detected region
                top_left = max_loc_jet2
                bottom_right = (top_left[0] + template_width_jet, top_left[1] + template_height_jet)
                cv2.rectangle(frame4, top_left, bottom_right, (0, 0, 225), 2)
                jets+=1
                #print('jet2')

            is_match2 = True
            matchs+=1
        
        
        if max_val2 <= 0.79:
            is_match2 = False
        
        
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            #print('0')

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=True, visualize=visualize)
           # print('1')


        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
           # print('2')


        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                #frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape, device=device)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                confP = 0
                confDn = 0
                confB = 0
                confs = []
                confDns = []
                confBs = []
                confPs = []
                xys = []
                xyPs = []
                xyDns = []
                xyBs = []
                xyxy = None
                xyxyP = None
                xyxyDn = None
                xyxyB = None

                for *xyxy, conf, cls in reversed(det):

                    if cls == 0 :
                        confBs.append(conf.item())
                        xyBs.append(xyxy)

                    if cls == 1 :
                        confs.append(conf.item())
                        xys.append(xyxy)
                    
                    if cls == 2 :
                        confDns.append(conf.item())
                        xyDns.append(xyxy)

                    if cls == 3 :                   
                        confPs.append(conf.item())
                        xyPs.append(xyxy)
                
                if confBs:
                    confB = max(confBs)
                    idx = confBs.index(max(confBs))
                    xyxyB = xyBs[idx]
                
                if confs:
                    conf = max(confs)
                    idx = confs.index(max(confs))
                    xyxy = xys[idx]

                if confDns:
                    confDn = max(confDns)
                    idx = confDns.index(max(confDns))
                    xyxyDn = xyDns[idx]

                if confPs:
                    confP = max(confPs)
                    idx = confPs.index(max(confPs))
                    xyxyP = xyPs[idx]
                
                hB = None
                hD = None
                hDn = None
                hP = None
                wB = None
                wD = None
                wDn = None
                wP = None
                for cls in [0, 1 ,2 ,3] :

                    if save_img or save_crop or view_img  :  
                        c = int(cls)  # integer class
                        if c == 0 and confB < 0.6 :
                            continue 
                        if c == 1 and conf < 0.8 :
                            continue 
                        if c == 2 and confDn < 0.6 :
                            continue 
                        if c == 3 and confP < 0.6 :
                            continue 

                        if c == 1 :
                            if (xyxy[1].item()+xyxy[3].item())/2 < 800 and not jumped:
                                #print('JUMP')
                                jumped = True
                                if not start_power_time:
                                    jump_count +=1
                            elif (xyxy[1].item()+xyxy[3].item())/2 >= 800:
                                jumped = False

                            if (xyxy[0].item()+xyxy[2].item())/2 < min1:
                                min1 = (xyxy[0].item()+xyxy[2].item())/2
                            elif (xyxy[0].item()+xyxy[2].item())/2 > max1:
                                max1 = (xyxy[0].item()+xyxy[2].item())/2
                            if (xyxy[0].item()+xyxy[2].item())/2 < 570 :
                                #print('LEFT')
                                if last_line != 'LEFT' and last_line != 'RIGHT':
                                    last_line = 'LEFT'
                                    line_changes_count +=1
                            elif (xyxy[0].item()+xyxy[2].item())/2 >= 660 and (xyxy[0].item()+xyxy[2].item())/2 <= 800 :
                                #print('CENTER')
                                if last_line != 'CENTER':
                                    last_line = 'CENTER'
                                    line_changes_count +=1 
                            elif (xyxy[0].item()+xyxy[2].item())/2 > 820 :
                                #print('RIGHT')
                                if last_line != 'RIGHT' and last_line != 'LEFT':
                                    last_line = 'RIGHT'
                                    line_changes_count +=1
                        if c == 0 :
                            print('Box =  ' + str(confB))
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {confB:.2f}')
                            annotator.box_label(xyxyB ,label  + str(conf) + ' ' + str(jump_count) + ' ' + str(line_changes_count) + '  w =  ' +  str((xyxyB[0].item()+xyxyB[2].item())/2) + ' ' +  last_line, color=colors(c, True))
                            wB = (xyxyB[0].item()+xyxyB[2].item())/2
                            hB = (xyxyB[1].item()+xyxyB[3].item())/2
                        if c == 1 :
                            print('Dino =  ' + str(conf))
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy ,label  + str(conf) + ' ' + str(jump_count) + ' ' + str(line_changes_count) + '  w =  ' +  str((xyxy[0].item()+xyxy[2].item())/2) + ' ' +  last_line, color=colors(c, True))
                            wD = (xyxy[0].item()+xyxy[2].item())/2
                            hD = (xyxy[1].item()+xyxy[3].item())/2
                        if c == 2 :
                            print('Dinon =  ' + str(confDn))
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {confDn:.2f}')
                            annotator.box_label(xyxyDn ,label  + str(confDn) + ' ' + str(jump_count) + ' ' + str(line_changes_count) + '  w =  ' +  str((xyxyDn[0].item()+xyxyDn[2].item())/2) + ' ' +  last_line, color=colors(c, True))
                            wDn = (xyxyDn[0].item()+xyxyDn[2].item())/2
                            hDn = (xyxyDn[1].item()+xyxyDn[3].item())/2
                        if c == 3 :
                            print('Power Jumper =  ' + str(confP))
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {confP:.2f}')
                            annotator.box_label(xyxyP , label + str(confP)  + '  w =  ' +  str((xyxyP[0].item()+xyxyP[2].item())/2) +  ' ' +  last_line, color=colors(c, True))
                            wP = (xyxyP[0].item()+xyxyP[2].item())/2
                            hP = (xyxyP[1].item()+xyxyP[3].item())/2
                        detected_count += 1
                        detect_num +=1
                if (hD and hP) :
                    print(' h Diff = ' + str(hD - hP))
                    print(' w Diff = ' + str(wD - wP))
                    if abs(hD - hP) < 400 and abs(wD - wP) < 100 and not is_power_jumper:
                        print('POWER JUMPER')
                        is_power_jumper = True
                        start_power_time = True
                        power_jumper_count += 1
                elif (hDn and hP) :
                    print(' h Diff = ' + str(hDn - hP))
                    print(' w Diff = ' + str(wDn - wP))
                    if abs(hDn - hP) < 400 and abs(wDn - wP) < 400 and not is_power_jumper:
                        print('POWER JUMPER')
                        is_power_jumper = True
                        start_power_time = True
                        power_jumper_count += 1
                else:
                        is_power_jumper = False

                if (hD and hB) :
                    print(' h Diff = ' + str(hD - hB))
                    print(' w Diff = ' + str(wD - wB))
                    if abs(hD - hB) < 400 and abs(wD - wB) < 100 and not is_box:
                        print('BOX')
                        is_box = True
                        box_count += 1
                elif (hDn and hB) :
                    print(' h Diff = ' + str(hDn - hB))
                    print(' w Diff = ' + str(wDn - wB))
                    if abs(hDn - hB) < 400 and abs(wDn - wB) < 400 and not is_box:
                        print('BOX')
                        is_box = True
                        box_count += 1
                else:
                    is_box = False
                

              #  print('-------------------------')
            
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)
                

        #LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    print('line changes = ' , line_changes_count, 'jump count = ', jump_count, 
          ' power jumpers = ' ,power_jumper_count,'sneakers = ',sneakers,'magnets = ', magnets,'2xs = ', twoxs,
          'skates = ', sks, 'jets = ' , jets , 'boxes = ' , box_count  , 
          'score = ', score, 'coin =', coin)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        print(detected_count)
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/data.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    start_time = time.time()
    
    run(**vars(opt))

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time/60} mins")

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
