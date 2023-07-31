# Standard modules
import os
import sys
import math
import time

# Logging
import logging
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
handler_format = logging.Formatter('%(asctime)s : [%(name)s - %(lineno)d] %(levelname)-8s - %(message)s')
stream_handler.setFormatter(handler_format)
logger.addHandler(stream_handler)

# Advanced modules
import cv2
import numpy as np

class GetTelops(object):
    def __init__(self, input_video: str, output_json: str):
        # Input video and its attributes
        self.input_video_name = input_video
        self.input_vudeo = cv2.VideoCapture(input_video)

        self.length = int(self.input_vudeo.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.input_vudeo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.input_vudeo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.size = (self.width, self.height)

        self.fps = self.input_video.get(cv2.CAP_PROP_FPS)

        # Paddle OCR
        self.OCR = PaddleOCR(land = 'japan', max_text_length = 30)

        # Output video attributes
        self.json_file = output_json

    def _print_information(self):
        logger.info('-' * 50)
        logger.info('Input video:')
        logger.info('  Name: %s' % (self.input_video_name))
        logger.info('  Length: %d' % (self.length))
        logger.info('  Width: %d in pixel' % (self.width))
        logger.info('  Height: %d in pixel' % (self.height))
        logger.info('  FPS: %f' % (self.fps))
        logger.info('Output json:')
        logger.info('  Name: %s' % (self.json_file))
        logger.info('-' * 50)

    def optical_character_recognition(self, frame):
        """ Paddle OCR
        ocr with paddleocr
        args：
            img: img for ocr, support ndarray, img_path and list or ndarray
            det: use text detection or not. If false, only rec will be exec. Default is True
            rec: use text recognition or not. If false, only det will be exec. Default is True
            cls: use angle classifier or not. Default is True. 
                 If true, the text with rotation of 180 degrees can be recognized.
                 If no text is rotated by 180 degrees, use cls=False to get better performance.
                 Text with rotation of 90 or 270 degrees can be recognized even if cls=False.
        """
        result = self.OCR.ocr(img = frame, det = True, rec = True, cls = True)
        print(result)

    def check_one_frame(self, iframe: int) -> np.ndarray:
        self.input_vudeo.set(cv2.CAP_PROP_POS_FRAMES, iframe)
        ret, frame = self.input_vudeo.read()
        logger.info('Frame: %4d / %d, read frame: %s' % (iframe, self.length, ret))
        if ret:
            self.optical_character_recognition(frame = frame)

    def check_all_frames(self) -> None:
        for i in range(self.length):
            self.check_one_frame(iframe = i)

if __name__ == '__main__':
    start_time = time.time()

    input_video = '../Inputs/Videos/ANN-2023.06.23.mp4'
    output_json = '../Outputs/ANN-2023.06.23_telop.json'
    generator = GetTelops(input_video = input_video, output_json = output_json)
    generator.check_all_frames()

    end_time = time.time()
    logger.info('Duration: %.4f sec' % (end_time - start_time))