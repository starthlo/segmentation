import cv2
import numpy as np
import os
from typing import Tuple, Dict, Any


class AdaptiveSlabDetector:
    """Intelligent slab detection that adapts parameters based on image characteristics."""

    def __init__(self, rollers_folder="rollers"):
        self.rollers_folder = rollers_folder
        self.ruler_templates = self._load_ruler_templates()

    def _load_ruler_templates(self):
        """Load ruler templates for background matching."""
        templates = {}
        if os.path.exists(self.rollers_folder):
            for file in os.listdir(self.rollers_folder):
                if file.endswith((".JPG", ".jpg")):
                    template_path = os.path.join(self.rollers_folder, file)
                    template = cv2.imread(template_path)
                    if template is not None:
                        templates[file] = template
        return templates

    def analyze_image_characteristics(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze image to determine optimal processing parameters."""
        h, w = image.shape[:2]

        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        analysis = {"image_size": (w, h), "total_pixels": w * h}

        # 1. Analyze brightness distribution
        brightness = lab[:, :, 0]
        analysis["avg_brightness"] = np.mean(brightness)
        analysis["brightness_std"] = np.std(brightness)
        analysis["very_bright_ratio"] = np.sum(brightness > 200) / (w * h)
        analysis["very_dark_ratio"] = np.sum(brightness < 50) / (w * h)

        # 2. Analyze color distribution
        analysis["red_ratio"] = self._get_color_ratio(hsv, [0, 80, 80], [15, 255, 255])
        analysis["yellow_ratio"] = self._get_color_ratio(
            hsv, [15, 100, 100], [35, 255, 255]
        )
        analysis["blue_ratio"] = self._get_color_ratio(
            hsv, [100, 50, 50], [130, 255, 255]
        )

        # 3. Texture analysis
        analysis["texture_variance"] = np.var(cv2.Laplacian(gray, cv2.CV_64F))

        # 4. Edge density
        edges = cv2.Canny(gray, 50, 150)
        analysis["edge_density"] = np.sum(edges > 0) / (w * h)

        # 5. Determine stone type
        analysis["stone_type"] = self._classify_stone_type(analysis)

        # 6. Determine background type
        analysis["background_type"] = self._classify_background_type(analysis)

        return analysis

    def _get_color_ratio(self, hsv: np.ndarray, lower: list, upper: list) -> float:
        """Get ratio of pixels in specified color range."""
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        return np.sum(mask > 0) / (hsv.shape[0] * hsv.shape[1])

    def _classify_stone_type(self, analysis: Dict[str, Any]) -> str:
        """Classify the type of stone based on characteristics."""
        brightness = analysis["avg_brightness"]
        brightness_std = analysis["brightness_std"]
        texture_var = analysis["texture_variance"]
        very_bright_ratio = analysis.get("very_bright_ratio", 0)

        # Check for light marble with high brightness
        if brightness > 140 and very_bright_ratio > 0.3:
            if brightness_std < 25:
                return "light_uniform"  # Very light, uniform marble
            else:
                return "light_veined"  # Light marble with strong veining
        elif brightness > 150 and brightness_std < 30:
            return "light_uniform"  # Light marble, uniform color
        elif brightness > 150 and brightness_std > 50:
            return "light_veined"  # Light marble with veining
        elif brightness < 100:
            return "dark_stone"  # Dark stone/granite
        elif texture_var > 1000:
            return "high_texture"  # High texture stone
        else:
            return "medium_stone"  # Medium tone stone

    def _classify_background_type(self, analysis: Dict[str, Any]) -> str:
        """Classify the background/ruler type."""
        red_ratio = analysis["red_ratio"]
        yellow_ratio = analysis["yellow_ratio"]
        blue_ratio = analysis["blue_ratio"]

        if red_ratio > 0.02:
            return "red_ruler"  # MARMIOROBICI style
        elif yellow_ratio > 0.03:
            return "yellow_equipment"  # AL_AJIAL_FACTORY style
        elif blue_ratio > 0.02:
            return "blue_equipment"  # ERREBI style
        else:
            return "minimal_equipment"  # Clean background

    def get_optimal_parameters(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get optimal processing parameters based on image analysis."""
        stone_type = analysis["stone_type"]
        background_type = analysis["background_type"]

        # Base parameters
        params = {
            "alpha_threshold": 0.1,
            "morph_kernel_size": 5,
            "min_contour_ratio": 0.01,
            "use_preprocessing": False,
            "use_postprocessing": True,
            "rect_scale_factor": 5,
            "use_improved_contours": True,
            "fill_holes_aggressively": False,
        }

        # Adjust for stone type
        if stone_type == "light_uniform":
            params["alpha_threshold"] = 0.03  # Very sensitive for light marble
            params["morph_kernel_size"] = 11  # Larger kernel to connect areas
            params["rect_scale_factor"] = 4  # More detailed rectangle detection
            params["min_contour_ratio"] = 0.005  # More permissive
            params["fill_holes_aggressively"] = True
        elif stone_type == "light_veined":
            params["alpha_threshold"] = 0.05  # Moderate sensitivity
            params["morph_kernel_size"] = 9
            params["rect_scale_factor"] = 4
            params["min_contour_ratio"] = 0.008
            params["fill_holes_aggressively"] = True
        elif stone_type == "dark_stone":
            params["alpha_threshold"] = 0.12  # Less sensitive for dark stone
            params["morph_kernel_size"] = 7
            params["rect_scale_factor"] = 6  # Coarser for dark stone
            params["min_contour_ratio"] = 0.005  # More permissive for complex shapes
        elif stone_type == "high_texture":
            params["alpha_threshold"] = 0.08
            params["morph_kernel_size"] = 7
            params["use_postprocessing"] = True

        # Adjust for background type
        if background_type == "red_ruler":
            params["use_preprocessing"] = True
            params["ruler_colors"] = ["red"]
            params["min_contour_ratio"] = 0.005  # More permissive for red ruler cases
        elif background_type == "yellow_equipment":
            params["use_preprocessing"] = True
            params["ruler_colors"] = ["yellow"]
            params["min_contour_ratio"] = 0.015  # More strict filtering
        elif background_type == "blue_equipment":
            params["use_preprocessing"] = True
            params["ruler_colors"] = ["blue"]
            params["min_contour_ratio"] = 0.01

        return params

    def preprocess_image_for_rembg(
        self, image: np.ndarray, params: Dict[str, Any]
    ) -> np.ndarray:
        """Preprocess image to help rembg focus on the slab."""
        if not params.get("use_preprocessing", False):
            return image

        ruler_colors = params.get("ruler_colors", [])
        processed_image = image.copy()
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Create mask for ruler areas
        ruler_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        for color in ruler_colors:
            if color == "red":
                mask1 = cv2.inRange(
                    hsv, np.array([0, 80, 80]), np.array([15, 255, 255])
                )
                mask2 = cv2.inRange(
                    hsv, np.array([165, 80, 80]), np.array([180, 255, 255])
                )
                color_mask = cv2.bitwise_or(mask1, mask2)
            elif color == "yellow":
                color_mask = cv2.inRange(
                    hsv, np.array([15, 100, 100]), np.array([35, 255, 255])
                )
            elif color == "blue":
                color_mask = cv2.inRange(
                    hsv, np.array([100, 50, 50]), np.array([130, 255, 255])
                )
            else:
                continue

            ruler_mask = cv2.bitwise_or(ruler_mask, color_mask)

        # Replace ruler areas with neutral background
        if np.sum(ruler_mask) > 0:
            processed_image[ruler_mask > 0] = [240, 240, 240]

        return processed_image

    def postprocess_mask(self, mask: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Post-process the detection mask to improve results."""
        if not params.get("use_postprocessing", True):
            return mask

        kernel_size = params.get("morph_kernel_size", 5)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )

        # Fill holes and connect areas
        processed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Aggressive hole filling for light stones
        if params.get("fill_holes_aggressively", False):
            # Use larger kernel for aggressive filling
            large_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size + 4, kernel_size + 4)
            )
            processed_mask = cv2.morphologyEx(
                processed_mask, cv2.MORPH_CLOSE, large_kernel
            )

        # Remove small noise
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel)

        return processed_mask

    def detect_slab_adaptive(
        self, image_path: str, rembg_function
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Main adaptive detection function."""
        # Load and prepare image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]

        # Downscale for processing
        small_w, small_h = orig_w // 4, orig_h // 4
        img_small = cv2.resize(
            img_rgb, (small_w, small_h), interpolation=cv2.INTER_AREA
        )

        # Analyze image characteristics
        analysis = self.analyze_image_characteristics(img_small)
        params = self.get_optimal_parameters(analysis)

        print("Image Analysis:")
        print(f"  Stone type: {analysis['stone_type']}")
        print(f"  Background type: {analysis['background_type']}")
        print(
            f"  Parameters: alpha={params['alpha_threshold']}, kernel={params['morph_kernel_size']}"
        )

        # Preprocess if needed
        img_for_rembg = self.preprocess_image_for_rembg(img_small, params)

        # Apply rembg
        processed_img_small = rembg_function(img_for_rembg)
        processed_img_small = np.array(processed_img_small).copy()
        if processed_img_small.shape[-1] == 3:
            processed_img_small = cv2.cvtColor(processed_img_small, cv2.COLOR_RGB2RGBA)

        alpha_small = processed_img_small[:, :, 3] / 255.0

        # Create mask with adaptive threshold
        alpha_threshold = params["alpha_threshold"]
        mask_small = (alpha_small > alpha_threshold).astype(np.uint8) * 255

        # Post-process mask
        mask_small = self.postprocess_mask(mask_small, params)

        # Quality check
        slab_area_ratio = np.sum(mask_small > 0) / (
            mask_small.shape[0] * mask_small.shape[1]
        )
        print(f"  Detection ratio: {slab_area_ratio:.2f}")

        # Upscale back to original size
        mask_full = cv2.resize(
            mask_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
        )
        processed_img_up = cv2.resize(
            processed_img_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
        )

        return processed_img_up, {
            "mask": mask_full,
            "analysis": analysis,
            "parameters": params,
            "detection_ratio": slab_area_ratio,
        }
