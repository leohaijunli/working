"""HailoDetector - Detector supporting both single images and video frames"""

import time
from pathlib import Path
import cv2
import numpy as np
import argparse

from common import (
    HailoPythonInferenceEngine, 
    DetectionPostProcessor,
    scale_detections_to_original, 
    format_detection_results,
    print_detection_summary
)


class HailoDetector:
    """Hailo-8L Detector - Supports both image file paths and numpy arrays (video frames)"""

    def __init__(self, 
                 hef_path: str = "../models/yolo26n.hef",
                 conf_threshold: float = 0.25,
                 normalize: bool = False,
                 verbose: bool = False):
        
        self.hef_path = hef_path
        self.conf_threshold = conf_threshold
        self.normalize = normalize
        self.verbose = verbose
        
        print(f"[Init] Loading Hailo model: {hef_path}")
        self.engine = HailoPythonInferenceEngine(hef_path)
        
        print("✅ HailoDetector initialized (Supports image paths and numpy frames)")

    def detect(self, image_input, save_output: bool = False):
        """
        Unified detection interface.
        image_input can be:
            - str: Path to the image file
            - np.ndarray: A single image frame (BGR or RGB)
        """
        if isinstance(image_input, str):
            return self._detect_from_path(image_input, save_output)
        elif isinstance(image_input, np.ndarray):
            return self._detect_from_array(image_input, save_output)
        else:
            raise TypeError(f"Unsupported input type: {type(image_input)}")

    def _detect_from_path(self, image_path: str, save_output: bool = False):
        """Perform detection from an image path"""
        print(f"[Detect] Processing image: {image_path}")
        
        orig_image = HailoPythonInferenceEngine.load_image(image_path)
        return self._run_inference(orig_image, save_output)

    def _detect_from_array(self, frame: np.ndarray, save_output: bool = False):
        """Perform detection from a numpy array (video frame)"""
        # Ensure BGR format (common OpenCV format)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # Convert to uint8 if necessary (adjust based on your pipeline)
            if frame.dtype == np.uint8:
                orig_image = frame
            else:
                orig_image = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
        else:
            orig_image = frame
        return self._run_inference(orig_image, save_output)

    def _run_inference(self, orig_image: np.ndarray, save_output: bool = False):
        """Core inference logic (Shared)"""
        orig_h, orig_w = orig_image.shape[:2]

        # Preprocessing
        if self.verbose:
            print("[Preprocessing...]", end="")
            
        input_data, orig_size, scale, pad_w, pad_h = HailoPythonInferenceEngine.preprocess_array(
            orig_image, normalize=self.normalize
        )
        
        # Execute inference
        t_start = time.perf_counter()
        results, stats = self.engine.infer(
            input_data, 
            verbose=self.verbose, 
            save_output=save_output, 
            conf_threshold=self.conf_threshold
        )
        total_time = time.perf_counter() - t_start

        if self.verbose:
            print(f"✓ Inference complete: {total_time*1000:.2f}ms | Detected {len(results)} objects")

        # Scale results back to original dimensions
        results = scale_detections_to_original(results, orig_h, orig_w, scale, pad_w, pad_h)



        # Draw bounding boxes
        output_image = DetectionPostProcessor.draw_bboxes(self,orig_image, results, thickness=2)

        return {
            "results": results,
            "output_image": output_image,
            "original_image": orig_image,
            "total_time_ms": total_time * 1000,
            "num_detections": len(results),
            "stats": stats
        }

    def save_result(self, result_dict: dict, output_path: str = "output_detected.jpg"):
        """Save the detection result image"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), result_dict["output_image"])
        print(f"✓ Result saved: {output_path}")
        return output_path

    def run_and_save(self, image_path: str, output_path: str = "output_detected.jpg"):
        """One-click detection + save (convenient for CLI calls)"""
        result = self.detect(image_path)
        
        # Save image
        self.save_result(result, output_path)
        
        # Print summary information
        print_detection_summary(
            title="DETECTION SUMMARY (Hailo-8L + Python Head)",
            image_path=image_path,
            model_info={"HEF": self.hef_path},
            total_time_ms=result.get("total_time_ms", 0),
            conf_threshold=self.conf_threshold,
            num_detections=result.get("num_detections", 0),
            output_path=output_path
        )
        
        return result


# ====================== Command Line Entry Point ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hailo Single Image Detector")
    
    parser.add_argument("image", type=str, 
                        help="Input image path, e.g., images/bus.jpg")
    
    parser.add_argument("--hef", type=str, default="yolo26n.hef",
                        help="Path to the HEF model (Default: yolo26n.hef)")
    
    parser.add_argument("--output", type=str, default="output_detected.jpg",
                        help="Output path for the saved image")
    
    parser.add_argument("--conf-threshold", type=float, default=0.3,
                        help="Confidence threshold, default 0.3")
    
    parser.add_argument("--verbose", action="store_true",
                        help="Display detailed inference logs")
    
    parser.add_argument("--normalize", action="store_true",
                        help="Whether to normalize input to [0,1]")

    args = parser.parse_args()

    # Check if the input image exists
    if not Path(args.image).exists():
        print(f"❌ Error: Image does not exist → {args.image}")
        exit(1)

    print(f"Using model: {args.hef}")
    print(f"Input image: {args.image}")
    print(f"Output path: {args.output}\n")

    # Create detector instance
    detector = HailoDetector(
        hef_path=args.hef,
        conf_threshold=args.conf_threshold,
        normalize=args.normalize,
        verbose=args.verbose
    )

    # Run detection and save results
    detector.run_and_save(
        image_path=args.image,
        output_path=args.output
    )