# [file name]: gate_detector.py
# [file content begin]
from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import logging
from skimage.morphology import skeletonize
from skimage import img_as_ubyte
import re
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LogicGateDetector:
    def __init__(self, model_path=None):
        try:
            # Use default model path if not provided
            if model_path is None:
                # Try to find the model in common locations
                possible_paths = [
                    r"D:\pross image && text 2\pretrained model\train7\weights\best.pt",
                    "./pretrained_model/train7/weights/best.pt",
                    "./model_weights/best.pt"
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        model_path = path
                        break
                
                if model_path is None:
                    raise FileNotFoundError("Could not find model weights file")
            
            logger.info(f"Loading YOLO model from {model_path}")
            self.model = YOLO(model_path)
            self.class_names = ["AND", "OR", "NAND", "NOR", "EXOR", "EXNOR", "NOT", "NOR INVERTED", "NAND INVERTED"]
            self.wire_color_lower = np.array([0, 0, 200])
            self.wire_color_upper = np.array([100, 100, 255])
            self.min_wire_length = 20
            self.canny_threshold1 = 50
            self.canny_threshold2 = 150
            self.pin_proximity_threshold = 15
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to initialize detector: {str(e)}")
            # Create a mock detector for text-only operation
            self.model = None
            self.class_names = ["AND", "OR", "NAND", "NOR", "XOR", "XNOR", "NOT"]

    def detect_circuit(self, image_path):
        try:
            if self.model is None:
                logger.warning("Using mock detector - returning empty results")
                return [], []
                
            logger.info(f"Processing image: {image_path}")
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image at {image_path}")
            
            processed_img = self._preprocess_image(img)
            gates = []
            results = self.model(img)
            
            for result in results:
                for i, box in enumerate(result.boxes):
                    class_id = int(box.cls)
                    confidence = float(box.conf)
                    bbox = box.xyxy[0].tolist()
                    
                    gate_info = {
                        "id": f"gate_{i}",
                        "type": self.class_names[class_id],
                        "confidence": confidence,
                        "bbox": bbox,
                        "center": (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)),
                        "inputs": self._estimate_io_positions(bbox, 'input'),
                        "output": self._estimate_io_positions(bbox, 'output'),
                        "connections": []
                    }
                    gates.append(gate_info)

            connections = self._find_connections_enhanced(processed_img, gates)
            logger.info(f"Found {len(gates)} gates and {len(connections)} connections")
            return gates, connections
            
        except Exception as e:
            logger.error(f"Detection failed: {str(e)}")
            return [], []

    def _preprocess_image(self, img):
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, self.canny_threshold1, self.canny_threshold2)
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=1)
            skeleton = skeletonize(dilated > 0)
            return img_as_ubyte(skeleton)
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            return img

    def _estimate_io_positions(self, bbox, io_type):
        try:
            x1, y1, x2, y2 = bbox
            if io_type == 'input':
                return [
                    (x1 - 10, int((y1 + y2) / 2) - 15),
                    (x1 - 10, int((y1 + y2) / 2) + 15)
                ]
            else:
                return [(x2 + 10, int((y1 + y2) / 2))]
        except Exception as e:
            logger.error(f"IO position estimation failed: {str(e)}")
            return []

    def _find_connections_enhanced(self, processed_img, gates):
        connections = []
        try:
            contours, _ = cv2.findContours(processed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            all_pins = []
            pin_info = []
            
            for gate in gates:
                for output_pos in gate["output"]:
                    all_pins.append(output_pos)
                    pin_info.append((output_pos, gate["id"], False, -1))
                for input_num, input_pos in enumerate(gate["inputs"]):
                    all_pins.append(input_pos)
                    pin_info.append((input_pos, gate["id"], True, input_num))
            
            for cnt in contours:
                if cv2.contourArea(cnt) > self.min_wire_length:
                    cnt_points = cnt.squeeze()
                    if len(cnt_points.shape) == 1:
                        cnt_points = np.array([cnt_points])
                    
                    for i, pin_pos in enumerate(all_pins):
                        distances = np.linalg.norm(cnt_points - pin_pos, axis=1)
                        if np.min(distances) < self.pin_proximity_threshold:
                            _, gate_id, is_input, input_num = pin_info[i]
                            for j, other_pin_pos in enumerate(all_pins):
                                if i != j and np.min(np.linalg.norm(cnt_points - other_pin_pos, axis=1)) < self.pin_proximity_threshold:
                                    _, other_gate_id, other_is_input, other_input_num = pin_info[j]
                                    if is_input and not other_is_input:
                                        connections.append({
                                            "source": other_gate_id,
                                            "target": gate_id,
                                            "input_num": input_num + 1
                                        })
                                    elif not is_input and other_is_input:
                                        connections.append({
                                            "source": gate_id,
                                            "target": other_gate_id,
                                            "input_num": other_input_num + 1
                                        })
            return connections
        except Exception as e:
            logger.error(f"Connection finding failed: {str(e)}")
            return []
# [file content end]