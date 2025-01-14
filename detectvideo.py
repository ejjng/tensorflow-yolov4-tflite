import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import csv

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
#flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/road.mp4', 'path to input video')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')
#flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'mp4v', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_boolean('dis_cv2_window', False, 'disable cv2 window during the process') # this is good for the .ipynb

def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    #input_size = FLAGS.size
    video_name = FLAGS.video
    video_path = './data/videos/'+video_name+'.mp4'
    output_path = './detection_results/'+video_name+'.mp4'
    input_W = 1920
    input_H = 1088

    print("Video from: ", video_path )
    vid = cv2.VideoCapture(video_path)
    f = open('logs/'+video_name+'.csv','w')

    # by default VideoCapture returns float instead of int
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    out = cv2.VideoWriter(output_path, codec, fps, (width, height))

    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])    
    infer = saved_model_loaded.signatures['serving_default']

    frame_id = 0
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #image = Image.fromarray(frame)
        else:
            if frame_id == vid.get(cv2.CAP_PROP_FRAME_COUNT):
                print("Video processing complete")
                break
            raise ValueError("No image! Try with another video format")

        prev_time = time.time()
        frame_size = frame.shape[:2]
        #image_data = cv2.resize(frame, (input_W, input_H))
        image_data = frame / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
    
        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        row = str(frame_id)
        for i in range(valid_detections.numpy()[0]):
            for j in range(4):
                row += ','
                row += str(boxes.numpy()[0][i][j])
            row += ','
            row += str(scores.numpy()[0][i])
            row += ','
            row += str(int(classes.numpy()[0][i]))
        f.write(row)
        f.write('\n')

        image = utils.draw_bbox(frame, pred_bbox)
        #cv2.imwrite('./detection_results/images/'+str(frame_id)+'.jpg', image)
        curr_time = time.time()
        exec_time = curr_time - prev_time
        result = np.asarray(image)
        info = "frame id: "+ str(frame_id) +" time: %.2f ms" %(1000*exec_time)
        print(info)

        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        '''if not FLAGS.dis_cv2_window:
            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("result", result)
            if cv2.waitKey(1) & 0xFF == ord('q'): break'''

        out.write(result)

        frame_id += 1
    f.close()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
