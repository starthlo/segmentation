import tkinter as tk
from tkinter import filedialog, Label, Button, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import sys
from datetime import datetime
import xml.etree.ElementTree as ET
from rembg import remove
from adaptive_slab_detector import AdaptiveSlabDetector
from multi_strategy_detector import MultiStrategyDetector

# Set U2NET_HOME to bundled .u2net folder if running as a bundled app
if getattr(sys, "frozen", False):
    bundle_dir = sys._MEIPASS
    u2net_path = os.path.join(bundle_dir, ".u2net")
    os.environ["U2NET_HOME"] = u2net_path
else:
    u2net_path = os.path.expanduser("~/.u2net")
    os.environ["U2NET_HOME"] = u2net_path


# Import SAM (required)
try:
    from segment_anything import sam_model_registry, SamPredictor
    import torch

    SAM_AVAILABLE = True
    print("SAM (Segment Anything Model) loaded successfully")
except ImportError as e:
    SAM_AVAILABLE = False
    print(f"WARNING: SAM dependencies not fully installed: {e}")
    print("PyTorch is installing... Using U2NET-only mode temporarily")
    print("Once PyTorch installation completes, restart app for full SAM functionality")


class PerfectSlabDetector:
    """
    Perfect Slab Detection App
    Uses U2NET + SAM segmentation for perfect slab background removal
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Perfect Slab Detector - U2NET + SAM")
        self.root.geometry("1400x800")  # Compact size that fits all screens

        # Initialize variables
        self.file_path = None
        self.processed_image = None
        self.mask_image = None
        self.sam_predictor = None

        # Default DPI (250 DPI = ~9.84 pixels per mm)
        self.pixels_per_mm = tk.DoubleVar(value=9.84)  # 250 DPI default

        # Initialize Adaptive Slab Detector
        try:
            self.adaptive_detector = AdaptiveSlabDetector(rollers_folder="rollers")
            print("Adaptive Slab Detector initialized successfully")
            if self.adaptive_detector.ruler_templates:
                print(
                    f"Loaded {len(self.adaptive_detector.ruler_templates)} ruler templates"
                )
        except Exception as e:
            print(f"Warning: Could not initialize Adaptive Slab Detector: {e}")
            self.adaptive_detector = None

        # Initialize Multi-Strategy Detector
        self.multi_strategy = MultiStrategyDetector(self.adaptive_detector)
        print("Multi-Strategy Detector initialized")

        # Initialize SAM if available
        self._initialize_sam()

        # Create GUI
        self._create_gui()

    def _initialize_sam(self):
        """Initialize SAM model."""
        if not SAM_AVAILABLE:
            print("SAM not available - will use U2NET only until PyTorch is installed")
            self.sam_predictor = None
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing SAM on {device}")

        # Try to load SAM models in priority order
        model_configs = [
            ("vit_h", "sam_models/sam_vit_h_4b8939.pth"),
            ("vit_h", "sam_models/sam_vit_h.pth"),
            ("vit_h", "sam_vit_h.pth"),
            ("vit_l", "sam_models/sam_vit_l.pth"),
            ("vit_l", "sam_vit_l.pth"),
            ("vit_b", "sam_models/sam_vit_b.pth"),
            ("vit_b", "sam_vit_b.pth"),
        ]

        for model_type, model_path in model_configs:
            if os.path.exists(model_path):
                try:
                    print(f"Loading SAM model: {model_path}")
                    sam = sam_model_registry[model_type](checkpoint=model_path)
                    sam.to(device=device)
                    self.sam_predictor = SamPredictor(sam)
                    print(
                        f"SAM model {model_type} loaded successfully from {model_path}"
                    )
                    return
                except Exception as e:
                    print(f"Failed to load {model_path}: {e}")
                    continue

        print("WARNING: No SAM model found - using U2NET only")
        print("Download SAM model to sam_models/sam_vit_h.pth for full functionality")
        self.sam_predictor = None

    def _create_gui(self):
        """Create the GUI interface."""

        # Title
        title_label = Label(
            self.root, text="Perfect Slab Detector", font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=10)

        if SAM_AVAILABLE and self.sam_predictor:
            subtitle = "U2NET + SAM Segmentation for Perfect Slab Detection"
        else:
            subtitle = "Enhanced U2NET Detection with Intelligent Processing"
        subtitle_label = Label(self.root, text=subtitle, font=("Arial", 10))
        subtitle_label.grid(row=1, column=0, columnspan=3, pady=5)

        # Image display areas - Compact but clear visibility
        self.input_img_label = Label(
            self.root, text="Input Image", font=("Arial", 11, "bold")
        )
        self.input_img_label.grid(row=2, column=0, padx=5, pady=3)
        self.input_image_display = Label(self.root, bg="lightgray", width=50, height=25)
        self.input_image_display.grid(row=3, column=0, padx=5, pady=3)

        self.result_img_label = Label(
            self.root, text="Result Image", font=("Arial", 11, "bold")
        )
        self.result_img_label.grid(row=2, column=1, padx=5, pady=3)
        self.result_image_display = Label(
            self.root, bg="lightgray", width=50, height=25
        )
        self.result_image_display.grid(row=3, column=1, padx=5, pady=3)

        self.mask_img_label = Label(
            self.root, text="Detection Mask", font=("Arial", 11, "bold")
        )
        self.mask_img_label.grid(row=2, column=2, padx=5, pady=3)
        self.mask_image_display = Label(self.root, bg="lightgray", width=50, height=25)
        self.mask_image_display.grid(row=3, column=2, padx=5, pady=3)

        # Control buttons
        self.upload_btn = Button(
            self.root,
            text="Select Slab Image",
            command=self.upload_image,
            bg="lightblue",
            font=("Arial", 12),
        )
        self.upload_btn.grid(row=4, column=0, pady=10)

        # Main processing button
        if SAM_AVAILABLE:
            button_text = "Perfect Detection (U2NET + SAM)"
            button_color = "green"
        else:
            button_text = "Enhanced Detection (Intelligent U2NET)"
            button_color = "blue"

        self.process_btn = Button(
            self.root,
            text=button_text,
            command=self.perfect_detection,
            state=tk.DISABLED,
            bg=button_color,
            fg="white",
            font=("Arial", 11, "bold"),
        )
        self.process_btn.grid(row=4, column=1, pady=10)

        # Save button
        self.save_btn = Button(
            self.root,
            text="Save Results",
            command=self.save_results,
            state=tk.DISABLED,
            bg="orange",
            font=("Arial", 12),
        )
        self.save_btn.grid(row=4, column=2, pady=10)

        # Process XML folder button
        self.xml_btn = Button(
            self.root,
            text="Process XML Folder",
            command=self.process_xml_folder,
            bg="lightcyan",
            font=("Arial", 11),
        )
        self.xml_btn.grid(row=5, column=1, pady=10)

        # Pixel/mm ratio setting
        dpi_frame = tk.Frame(self.root)
        dpi_frame.grid(row=5, column=2, pady=10)

        dpi_label = Label(dpi_frame, text="Pixels/mm:", font=("Arial", 10))
        dpi_label.grid(row=0, column=0, padx=5)

        self.dpi_entry = tk.Entry(
            dpi_frame, textvariable=self.pixels_per_mm, width=8, font=("Arial", 10)
        )
        self.dpi_entry.grid(row=0, column=1, padx=5)

        dpi_info = Label(dpi_frame, text="(250 DPI)", font=("Arial", 8), fg="gray")
        dpi_info.grid(row=1, column=0, columnspan=2)

        # Status label
        self.status_label = Label(
            self.root,
            text="Ready - Select an image to begin",
            font=("Arial", 10),
            fg="blue",
        )
        self.status_label.grid(row=6, column=0, columnspan=3, pady=10)

    def upload_image(self):
        """Upload and display input image."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.JPG;*.jpg;*.jpeg;*.png")]
        )

        if file_path:
            self.file_path = file_path

            # Display input image - Compact but clear
            img = Image.open(file_path)
            img.thumbnail((400, 300))  # Compact size that shows details clearly
            img_tk = ImageTk.PhotoImage(img)
            self.input_image_display.config(image=img_tk)
            self.input_image_display.image = img_tk

            # Enable processing
            self.process_btn.config(state=tk.NORMAL)
            self.status_label.config(
                text=f"Image loaded: {os.path.basename(file_path)}"
            )

    def perfect_detection(self):
        """Perform perfect slab detection using U2NET + SAM/Enhancement."""
        if not self.file_path:
            messagebox.showerror("Error", "No image selected")
            return

        try:
            self.status_label.config(text="Processing... Please wait")
            self.root.update()

            # Load input image
            input_image = cv2.imread(self.file_path)
            orig_h, orig_w = input_image.shape[:2]

            # Stage 1: U2NET Background Removal
            self.status_label.config(text="Stage 1: U2NET background removal...")
            self.root.update()

            u2net_mask = self._apply_u2net(input_image)
            u2net_coverage = (np.sum(u2net_mask > 0) / (orig_w * orig_h)) * 100

            # Stage 2: SAM Segmentation or U2NET only
            if self.sam_predictor:
                self.status_label.config(text="Stage 2: SAM segmentation...")
                self.root.update()
                final_mask = self._apply_sam_segmentation(input_image, u2net_mask)
            else:
                self.status_label.config(
                    text="Stage 2: Intelligent U2NET enhancement..."
                )
                self.root.update()
                final_mask = self._enhance_u2net_mask(u2net_mask)

            # Stage 3: Calculate rectangles and generate results
            self.status_label.config(text="Stage 3: Calculating geometry...")
            self.root.update()

            external_rect, internal_rect = self._calculate_rectangles(final_mask)

            # Create results
            result_rgba = self._create_result_image(input_image, final_mask)
            mask_visualization = self._create_mask_visualization(
                final_mask, external_rect, internal_rect, orig_w, orig_h
            )

            # Display results
            self._display_results(result_rgba, mask_visualization)

            # Store for saving
            self.processed_image = result_rgba
            self.mask_image = mask_visualization
            self.external_rect = external_rect
            self.internal_rect = internal_rect

            # Enable saving
            self.save_btn.config(state=tk.NORMAL)

            final_coverage = (np.sum(final_mask > 0) / (orig_w * orig_h)) * 100

            # Display measurements in both pixels and mm
            ext_mm = self._pixels_to_mm(external_rect)
            int_mm = self._pixels_to_mm(internal_rect)

            # Include adaptive detection info if available
            status_text = f"Detection complete! Coverage: {final_coverage:.1f}%\n"
            if hasattr(self, "current_analysis"):
                status_text += f"Stone: {self.current_analysis['stone_type']}, BG: {self.current_analysis['background_type']}\n"
            if hasattr(self, "detection_strategy"):
                status_text += f"Strategy: {self.detection_strategy} (Q={self.detection_quality:.2f})\n"
            status_text += f"External: {external_rect[2]}x{external_rect[3]}px ({ext_mm[2]:.1f}x{ext_mm[3]:.1f}mm)\n"
            status_text += f"Internal: {internal_rect[2]}x{internal_rect[3]}px ({int_mm[2]:.1f}x{int_mm[3]:.1f}mm)"

            self.status_label.config(text=status_text)

        except Exception as e:
            messagebox.showerror("Error", f"Perfect detection failed: {str(e)}")
            self.status_label.config(text="Error occurred during processing")

    def _pixels_to_mm(self, rect):
        """Convert rectangle coordinates from pixels to mm."""
        if rect is None:
            return (0, 0, 0, 0)

        pixels_per_mm = self.pixels_per_mm.get()
        x, y, w, h = rect
        return (
            x / pixels_per_mm,  # x in mm
            y / pixels_per_mm,  # y in mm
            w / pixels_per_mm,  # width in mm
            h / pixels_per_mm,  # height in mm
        )

    def _apply_u2net(self, input_image):
        """Apply U2NET background removal with multi-strategy approach at full resolution."""

        orig_h, orig_w = input_image.shape[:2]
        img_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        
        print(f"Processing at full resolution: {orig_w}x{orig_h}")

        # Use adaptive detector if available
        if self.adaptive_detector:
            # Analyze image characteristics using full resolution
            # (or slightly downsampled only for analysis, not processing)
            analysis_w, analysis_h = max(orig_w // 2, 512), max(orig_h // 2, 512)
            img_analysis = cv2.resize(img_rgb, (analysis_w, analysis_h), interpolation=cv2.INTER_AREA)
            
            analysis = self.adaptive_detector.analyze_image_characteristics(img_analysis)
            params = self.adaptive_detector.get_optimal_parameters(analysis)

            # Log analysis results
            print(f"Image Analysis:")
            print(f"  Stone type: {analysis['stone_type']}")
            print(f"  Background type: {analysis['background_type']}")
            print(f"  Avg brightness: {analysis['avg_brightness']:.1f}")
            print(f"  Alpha threshold: {params['alpha_threshold']}")
            print(f"  Morph kernel size: {params['morph_kernel_size']}")

            # Store analysis and params for use in other methods
            self.current_analysis = analysis
            self.current_params = params
            
            # Scale morphology kernel for full resolution
            params_fullres = params.copy()
            params_fullres['morph_kernel_size'] = params['morph_kernel_size'] * 2  # Scale for full res
            
            # Use multi-strategy detector at FULL RESOLUTION
            u2net_mask, strategy_name, quality = self.multi_strategy.apply_u2net_with_strategies(
                img_rgb, analysis, params_fullres
            )
            
            # Store quality info
            self.detection_quality = quality
            self.detection_strategy = strategy_name
            
        else:
            # Fallback to simple processing at full resolution
            print("Using default parameters (adaptive detector not available)")
            alpha_threshold = 0.1
            
            processed_img = remove(img_rgb)
            processed_img = np.array(processed_img)

            if processed_img.shape[-1] == 3:
                processed_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2RGBA)

            alpha = processed_img[:, :, 3] / 255.0
            u2net_mask = (alpha > alpha_threshold).astype(np.uint8) * 255

        return u2net_mask

    def _apply_sam_segmentation(self, input_image, u2net_mask):
        """Apply SAM segmentation using U2NET mask as guidance."""

        try:
            # Set image for SAM
            image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            self.sam_predictor.set_image(image_rgb)

            # Extract prompts from U2NET mask
            contours, _ = cv2.findContours(
                u2net_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                print("No contours found in U2NET mask, returning original")
                return u2net_mask

            # Get largest contour (main slab)
            largest_contour = max(contours, key=cv2.contourArea)

            # Extract bounding box first (avoid redundant calculation)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Extract center point
            moments = cv2.moments(largest_contour)
            if moments["m00"] > 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
            else:
                # Fallback: use bounding box center
                cx, cy = x + w // 2, y + h // 2

            # Enhanced prompting strategy: Use multiple points for better segmentation
            input_points = []
            input_labels = []

            # 1. Center point (most important)
            input_points.append([cx, cy])
            input_labels.append(1)

            # 2. Add additional sample points along the contour for better guidance
            contour_len = len(largest_contour)
            if contour_len > 10:  # Only if we have enough contour points
                # Sample 4 points evenly distributed along the contour
                sample_indices = [
                    contour_len // 4,
                    contour_len // 2,
                    3 * contour_len // 4,
                    contour_len - 1,
                ]
                for idx in sample_indices:
                    pt = largest_contour[idx][0]
                    input_points.append([int(pt[0]), int(pt[1])])
                    input_labels.append(1)

            input_points = np.array(input_points)
            input_labels = np.array(input_labels)
            input_box = np.array([x, y, x + w, y + h])

            print(f"SAM prompts: {len(input_points)} positive points + bounding box")

            # Run SAM prediction
            masks, scores, logits = self.sam_predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                box=input_box,
                multimask_output=True,
            )

            # Select best mask
            best_mask_idx = np.argmax(scores)
            sam_mask = masks[best_mask_idx].astype(np.uint8) * 255

            print(f"SAM prediction score: {scores[best_mask_idx]:.3f}")

            # Validate SAM result
            sam_coverage = np.sum(sam_mask > 0) / sam_mask.size
            u2net_coverage = np.sum(u2net_mask > 0) / u2net_mask.size

            # Fallback to U2NET if SAM result is suspicious
            if sam_coverage < u2net_coverage * 0.3 or sam_coverage > 0.95:
                print(
                    f"SAM result suspicious (coverage: {sam_coverage:.1%}), using U2NET instead"
                )
                return self._post_process_mask(u2net_mask)

            # Post-process SAM result
            sam_mask = self._post_process_mask(sam_mask)

            # Final validation
            if np.sum(sam_mask > 0) == 0:
                print(
                    "SAM produced empty mask after post-processing, falling back to U2NET"
                )
                return self._post_process_mask(u2net_mask)

            return sam_mask

        except Exception as e:
            # Preserve original exception context
            print(f"SAM segmentation failed: {e}")
            import traceback

            traceback.print_exc()
            print("Falling back to U2NET mask")
            return self._post_process_mask(u2net_mask)

    def _enhance_u2net_mask(self, u2net_mask):
        """Significantly enhance U2NET mask for much better results using adaptive parameters."""
        print("Applying intelligent U2NET enhancement with adaptive parameters...")

        # Get original image for enhancement
        input_image = cv2.imread(self.file_path)
        h, w = input_image.shape[:2]

        # Use adaptive parameters if available
        if hasattr(self, "current_params"):
            params = self.current_params
            analysis = self.current_analysis
            kernel_size = params.get("morph_kernel_size", 15)
            fill_aggressively = params.get("fill_holes_aggressively", False)
            stone_type = analysis.get("stone_type", "medium_stone")
        else:
            kernel_size = 15
            fill_aggressively = False
            stone_type = "medium_stone"

        # Step 1: Improve U2NET mask with adaptive morphological operations
        enhanced_mask = u2net_mask.copy()

        # Close gaps in the slab detection with adaptive kernel size
        kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        enhanced_mask = cv2.morphologyEx(enhanced_mask, cv2.MORPH_CLOSE, kernel_close)

        # Fill holes inside the slab
        kernel_fill = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size + 10, kernel_size + 10)
        )
        enhanced_mask = cv2.morphologyEx(enhanced_mask, cv2.MORPH_CLOSE, kernel_fill)

        # Aggressive filling for light stones
        if fill_aggressively:
            large_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size + 15, kernel_size + 15)
            )
            enhanced_mask = cv2.morphologyEx(
                enhanced_mask, cv2.MORPH_CLOSE, large_kernel
            )
            print(f"Applied aggressive hole filling for {stone_type}")

        # Step 2: Color-based enhancement using slab characteristics
        if np.sum(enhanced_mask > 0) > 0:
            # Extract slab color properties
            mask_bool = enhanced_mask > 0
            slab_pixels = input_image[mask_bool]

            if len(slab_pixels) > 100:  # Need enough pixels for analysis
                slab_mean = np.mean(slab_pixels, axis=0)
                slab_std = np.std(slab_pixels, axis=0)

                # Adaptive tolerance based on stone type
                if stone_type == "light_uniform":
                    tolerance_factor = 3.0  # Wider tolerance for uniform light stones
                elif stone_type == "light_veined":
                    tolerance_factor = 2.8  # Wide tolerance for veined light stones
                elif stone_type == "dark_stone":
                    tolerance_factor = 2.2  # Narrower tolerance for dark stones
                else:
                    tolerance_factor = 2.5  # Default

                # Create color-based mask with adaptive tolerance
                color_lower = np.maximum(0, slab_mean - tolerance_factor * slab_std)
                color_upper = np.minimum(255, slab_mean + tolerance_factor * slab_std)

                color_mask = cv2.inRange(
                    input_image,
                    color_lower.astype(np.uint8),
                    color_upper.astype(np.uint8),
                )

                # Step 3: HSV-based stone detection with adaptive ranges
                hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)

                if stone_type in ["light_uniform", "light_veined"]:
                    # For light stones: wider saturation range, higher value range
                    lower_stone = np.array([0, 0, 40])
                    upper_stone = np.array([180, 100, 255])
                elif stone_type == "dark_stone":
                    # For dark stones: lower value range
                    lower_stone = np.array([0, 0, 20])
                    upper_stone = np.array([180, 150, 180])
                else:
                    # Default: moderate ranges
                    lower_stone = np.array([0, 0, 30])
                    upper_stone = np.array([180, 120, 220])

                hsv_mask = cv2.inRange(hsv, lower_stone, upper_stone)

                # Step 4: Combine all enhancement methods with adaptive weighting
                combined = np.zeros((h, w), dtype=np.float32)

                if stone_type in ["light_uniform", "light_veined"]:
                    # For light stones: rely more on color and HSV
                    combined += enhanced_mask.astype(np.float32) * 0.5
                    combined += color_mask.astype(np.float32) * 0.35
                    combined += hsv_mask.astype(np.float32) * 0.15
                else:
                    # For other stones: rely more on U2NET
                    combined += enhanced_mask.astype(np.float32) * 0.6
                    combined += color_mask.astype(np.float32) * 0.3
                    combined += hsv_mask.astype(np.float32) * 0.1

                # Threshold and clean
                _, final_mask = cv2.threshold(combined, 150, 255, cv2.THRESH_BINARY)
                final_mask = final_mask.astype(np.uint8)

                # Step 5: Select best contour and clean edges
                final_mask = self._post_process_mask(final_mask)

                # Step 6: Ensure we didn't make it worse
                original_coverage = (np.sum(u2net_mask > 0) / (h * w)) * 100
                enhanced_coverage = (np.sum(final_mask > 0) / (h * w)) * 100

                print(
                    f"Enhancement: {original_coverage:.1f}% -> {enhanced_coverage:.1f}% (stone: {stone_type})"
                )

                # Adaptive validation based on stone type
                if stone_type in ["light_uniform", "light_veined"]:
                    # More permissive for light stones
                    min_ratio = 0.4
                    max_coverage = 95
                else:
                    # Standard validation
                    min_ratio = 0.5
                    max_coverage = 90

                if (
                    enhanced_coverage > original_coverage * min_ratio
                    and enhanced_coverage < max_coverage
                ):
                    return final_mask
                else:
                    print(
                        f"Enhancement rejected (ratio: {enhanced_coverage / max(original_coverage, 1):.2f}), using post-processed U2NET"
                    )

        # Fallback: just post-process original
        return self._post_process_mask(enhanced_mask)

    def _refine_mask_with_edges(self, mask, input_image):
        """
        Refine mask using edge detection to separate slab from similar-colored background.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        
        # Detect strong edges (boundaries between slab and rollers)
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges to create boundary regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Use edges to refine the mask
        # Areas with strong edges near mask boundaries might be incorrect
        mask_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if mask_contours:
            # Create a refined mask
            refined_mask = mask.copy()
            
            # Check edge density around the mask boundary
            boundary_mask = np.zeros_like(mask)
            cv2.drawContours(boundary_mask, mask_contours, -1, 255, 10)
            
            # If there are strong edges at the boundary, the mask is likely good
            edge_at_boundary = cv2.bitwise_and(edges_dilated, boundary_mask)
            edge_density = np.sum(edge_at_boundary > 0) / max(np.sum(boundary_mask > 0), 1)
            
            print(f"  Edge density at boundary: {edge_density:.3f}")
            
            # If edge density is low, the mask might include roller regions
            # Use morphological erosion to pull back from uncertain areas
            if edge_density < 0.1:
                print("  Low edge density - applying conservative mask refinement")
                kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                refined_mask = cv2.erode(mask, kernel_erode, iterations=2)
            
            return refined_mask
        
        return mask

    def _post_process_mask(self, mask):
        """Post-process mask for clean results."""

        # Remove small noise
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)

        # Fill holes
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        # Keep only largest connected component
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            final_mask = np.zeros_like(mask)
            cv2.fillPoly(final_mask, [largest_contour], 255)

            # Refine with edge detection if we have the original image
            if hasattr(self, 'file_path') and self.file_path:
                try:
                    input_image = cv2.imread(self.file_path)
                    if input_image is not None and input_image.shape[:2] == mask.shape:
                        final_mask = self._refine_mask_with_edges(final_mask, input_image)
                except Exception as e:
                    print(f"  Could not apply edge refinement: {e}")

            # Smooth edges
            final_mask = cv2.GaussianBlur(final_mask, (3, 3), 1)
            _, final_mask = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)

            return final_mask

        return mask

    def _calculate_rectangles(self, mask):
        """Calculate external and internal rectangles."""

        # Downsample for faster processing
        h, w = mask.shape
        small_w, small_h = w // 4, h // 4
        mask_small = cv2.resize(
            mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST
        )

        # Find external rectangle
        contours, _ = cv2.findContours(
            mask_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        external_rect = (0, 0, 0, 0)
        internal_rect = (0, 0, 0, 0)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w_rect, h_rect = cv2.boundingRect(largest_contour)

            # Scale back to original size
            external_rect = (x * 4, y * 4, w_rect * 4, h_rect * 4)

            # Find internal rectangle using dynamic programming
            binary_mask_small = 255 - mask_small  # Invert for internal rectangle
            internal = self._find_largest_internal_rectangle(binary_mask_small)

            if internal:
                ix, iy, iw, ih = internal
                internal_rect = (ix * 4, iy * 4, iw * 4, ih * 4)

        return external_rect, internal_rect

    def _find_largest_internal_rectangle(self, binary_mask):
        """Find largest internal rectangle using dynamic programming."""

        if len(binary_mask.shape) > 2:
            binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)

        h, w = binary_mask.shape
        heights = np.zeros(w, dtype=int)
        max_area = 0
        best_rect = None

        for i in range(h):
            for j in range(w):
                if binary_mask[i, j] == 0:  # Foreground (slab area)
                    heights[j] += 1
                else:
                    heights[j] = 0

            # Find largest rectangle in histogram
            area, rect = self._largest_rectangle_in_histogram(heights, i)
            if area > max_area:
                max_area = area
                best_rect = rect

        return best_rect

    def _largest_rectangle_in_histogram(self, heights, row):
        """Find largest rectangle in histogram."""
        stack = []
        max_area = 0
        best_rect = None

        for i, h in enumerate(heights):
            start = i
            while stack and stack[-1][1] > h:
                idx, height = stack.pop()
                area = height * (i - idx)
                if area > max_area:
                    max_area = area
                    width = i - idx
                    best_rect = (idx, row - height + 1, width, height)
                start = idx
            stack.append((start, h))

        for idx, height in stack:
            area = height * (len(heights) - idx)
            if area > max_area:
                max_area = area
                width = len(heights) - idx
                best_rect = (idx, row - height + 1, width, height)

        return max_area, best_rect

    def _create_result_image(self, input_image, mask):
        """Create result image with transparent background."""

        result_rgba = np.zeros(
            (input_image.shape[0], input_image.shape[1], 4), dtype=np.uint8
        )
        result_rgba[:, :, :3] = input_image

        # Ensure mask is binary (0 or 255) for proper alpha channel
        alpha_mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        result_rgba[:, :, 3] = alpha_mask

        return result_rgba

    def _create_mask_visualization(
        self, mask, external_rect, internal_rect, orig_w, orig_h
    ):
        """Create mask visualization with rectangles."""

        # Create colored mask
        mask_colored = (
            np.ones((orig_h, orig_w, 3), dtype=np.uint8) * 255
        )  # White background
        mask_colored[mask == 0] = (0, 0, 0)  # Black for background

        # Draw external rectangle (green)
        if external_rect[2] > 0 and external_rect[3] > 0:
            x, y, w, h = external_rect
            cv2.rectangle(mask_colored, (x, y), (x + w, y + h), (0, 255, 0), 8)

        # Draw internal rectangle (red)
        if internal_rect[2] > 0 and internal_rect[3] > 0:
            x, y, w, h = internal_rect
            cv2.rectangle(mask_colored, (x, y), (x + w, y + h), (0, 0, 255), 6)

        # Add labels with both pixels and mm
        font = cv2.FONT_HERSHEY_SIMPLEX
        if external_rect[2] > 0:
            ext_mm = self._pixels_to_mm(external_rect)
            ext_text = f"Ext: {external_rect[2]}x{external_rect[3]}px ({ext_mm[2]:.1f}x{ext_mm[3]:.1f}mm)"
            cv2.putText(
                mask_colored,
                ext_text,
                (external_rect[0], max(external_rect[1] - 15, 30)),
                font,
                1.2,
                (0, 255, 0),
                3,
            )
        if internal_rect[2] > 0:
            int_mm = self._pixels_to_mm(internal_rect)
            int_text = f"Int: {internal_rect[2]}x{internal_rect[3]}px ({int_mm[2]:.1f}x{int_mm[3]:.1f}mm)"
            cv2.putText(
                mask_colored,
                int_text,
                (
                    internal_rect[0],
                    min(internal_rect[1] + internal_rect[3] + 35, orig_h - 15),
                ),
                font,
                1.2,
                (0, 0, 255),
                3,
            )

        return mask_colored

    def _display_results(self, result_rgba, mask_visualization):
        """Display results in GUI - MUCH LARGER."""

        # Display result image - Compact but clear
        result_rgb = cv2.cvtColor(result_rgba[:, :, :3], cv2.COLOR_BGR2RGB)
        result_pil = Image.fromarray(result_rgb)
        result_pil.thumbnail((400, 300))  # Compact size that shows details clearly
        result_tk = ImageTk.PhotoImage(result_pil)
        self.result_image_display.config(image=result_tk)
        self.result_image_display.image = result_tk

        # Display mask - Compact but clear
        mask_rgb = cv2.cvtColor(mask_visualization, cv2.COLOR_BGR2RGB)
        mask_pil = Image.fromarray(mask_rgb)
        mask_pil.thumbnail((400, 300))  # Compact size that shows details clearly
        mask_tk = ImageTk.PhotoImage(mask_pil)
        self.mask_image_display.config(image=mask_tk)
        self.mask_image_display.image = mask_tk

    def save_results(self):
        """Save processing results."""
        if self.processed_image is None or not self.file_path:
            messagebox.showerror("Error", "No results to save")
            return

        try:
            # Create result directory
            base_name = os.path.splitext(os.path.basename(self.file_path))[0]
            result_dir = f"result/{base_name}_slab"
            os.makedirs(result_dir, exist_ok=True)

            # Save processed image with white background (JPG doesn't support transparency)
            result_rgb = self.processed_image[:, :, :3]  # RGB channels
            alpha = self.processed_image[:, :, 3:4]  # Alpha channel

            # Create white background where alpha is 0 (transparent)
            white_bg = np.ones_like(result_rgb) * 255
            alpha_norm = alpha.astype(np.float32) / 255.0

            # Blend: result = alpha * foreground + (1-alpha) * background
            result_with_bg = (
                alpha_norm * result_rgb + (1 - alpha_norm) * white_bg
            ).astype(np.uint8)
            result_bgr = cv2.cvtColor(result_with_bg, cv2.COLOR_RGB2BGR)

            process_path = os.path.join(result_dir, "process.JPG")
            cv2.imwrite(process_path, result_bgr)

            # Save mask visualization
            mask_path = os.path.join(result_dir, "binary_mask.JPG")
            cv2.imwrite(mask_path, self.mask_image)

            # Generate XML
            self._generate_xml(result_dir, base_name)

            # Show success in status instead of popup
            self.status_label.config(text=f"SUCCESS: Results saved to: {result_dir}")
            print(f"SUCCESS: Results saved to: {result_dir}")

        except Exception as e:
            messagebox.showerror("Error", f"Save failed: {str(e)}")

    def _save_results_to_xml_folder(self, xml_folder, idslab):
        """Save processing results to folder in same directory as XML files."""
        if self.processed_image is None or not self.file_path:
            return

        try:
            # Create DIMENSION directory in parent folder of XML folder
            parent_folder = os.path.dirname(xml_folder)
            dimension_folder = os.path.join(parent_folder, "DIMENSION")
            os.makedirs(dimension_folder, exist_ok=True)

            # Create result directory inside DIMENSION folder
            result_dir = os.path.join(dimension_folder, f"{idslab}_slab")
            os.makedirs(result_dir, exist_ok=True)

            # Save processed image with white background
            result_rgb = self.processed_image[:, :, :3]  # RGB channels
            alpha = self.processed_image[:, :, 3:4]  # Alpha channel

            # Create white background where alpha is 0 (transparent)
            white_bg = np.ones_like(result_rgb) * 255
            alpha_norm = alpha.astype(np.float32) / 255.0

            # Blend: result = alpha * foreground + (1-alpha) * background
            result_with_bg = (
                alpha_norm * result_rgb + (1 - alpha_norm) * white_bg
            ).astype(np.uint8)
            result_bgr = cv2.cvtColor(result_with_bg, cv2.COLOR_RGB2BGR)

            process_path = os.path.join(result_dir, "process.JPG")
            cv2.imwrite(process_path, result_bgr)

            # Save mask visualization
            mask_path = os.path.join(result_dir, "binary_mask.JPG")
            cv2.imwrite(mask_path, self.mask_image)

            print(f"SUCCESS: Results saved to XML folder: {result_dir}")

        except Exception as e:
            print(f"Error saving results to XML folder: {str(e)}")

    def _generate_xml(self, result_dir, base_name):
        """Generate XML metadata."""

        external_rect = getattr(self, "external_rect", (0, 0, 0, 0))
        internal_rect = getattr(self, "internal_rect", (0, 0, 0, 0))

        # Create XML
        now = datetime.now()
        xml_path = os.path.join(result_dir, f"{base_name}_slab.xml")

        root = ET.Element("ROOT")
        ET.SubElement(root, "IMAGE_PATH").text = os.path.dirname(self.file_path)
        ET.SubElement(root, "DATE").text = now.strftime("%Y-%m-%d")
        ET.SubElement(root, "TIME").text = now.strftime("%H:%M:%S")

        # Parse company info from filename
        company_elem = ET.SubElement(root, "COMPANY")
        company_elem.set("ID", base_name.split("-")[0] if "-" in base_name else "")
        ET.SubElement(root, "BLOCK").text = (
            base_name.split("-")[1] if len(base_name.split("-")) > 1 else ""
        )
        ET.SubElement(root, "PROG").text = (
            base_name.split("-")[2] if len(base_name.split("-")) > 2 else ""
        )

        material_elem = ET.SubElement(root, "MATERIAL")
        material_elem.set("ID", "")
        material_elem.set("NAME", "")

        thickness_elem = ET.SubElement(root, "THICKNESS", REAL="2")
        thickness_elem.text = "2"

        ET.SubElement(root, "IDSLAB").text = base_name

        # Add rectangle attributes in pixels
        ET.SubElement(root, "ATTRIBUTE_EXTERNAL_X").text = str(external_rect[0])
        ET.SubElement(root, "ATTRIBUTE_EXTERNAL_Y").text = str(external_rect[1])
        ET.SubElement(root, "ATTRIBUTE_EXTERNAL_WIDTH").text = str(external_rect[2])
        ET.SubElement(root, "ATTRIBUTE_EXTERNAL_HEIGHT").text = str(external_rect[3])
        ET.SubElement(root, "ATTRIBUTE_INTERNAL_X").text = str(internal_rect[0])
        ET.SubElement(root, "ATTRIBUTE_INTERNAL_Y").text = str(internal_rect[1])
        ET.SubElement(root, "ATTRIBUTE_INTERNAL_WIDTH").text = str(internal_rect[2])
        ET.SubElement(root, "ATTRIBUTE_INTERNAL_HEIGHT").text = str(internal_rect[3])

        # Add rectangle attributes in mm
        ext_mm = self._pixels_to_mm(external_rect)
        int_mm = self._pixels_to_mm(internal_rect)

        ET.SubElement(root, "ATTRIBUTE_EXTERNAL_X_MM").text = f"{ext_mm[0]:.2f}"
        ET.SubElement(root, "ATTRIBUTE_EXTERNAL_Y_MM").text = f"{ext_mm[1]:.2f}"
        ET.SubElement(root, "ATTRIBUTE_EXTERNAL_WIDTH_MM").text = f"{ext_mm[2]:.2f}"
        ET.SubElement(root, "ATTRIBUTE_EXTERNAL_HEIGHT_MM").text = f"{ext_mm[3]:.2f}"
        ET.SubElement(root, "ATTRIBUTE_INTERNAL_X_MM").text = f"{int_mm[0]:.2f}"
        ET.SubElement(root, "ATTRIBUTE_INTERNAL_Y_MM").text = f"{int_mm[1]:.2f}"
        ET.SubElement(root, "ATTRIBUTE_INTERNAL_WIDTH_MM").text = f"{int_mm[2]:.2f}"
        ET.SubElement(root, "ATTRIBUTE_INTERNAL_HEIGHT_MM").text = f"{int_mm[3]:.2f}"

        # Add DPI information
        ET.SubElement(root, "PIXELS_PER_MM").text = f"{self.pixels_per_mm.get():.2f}"

        # Add adaptive detection metadata if available
        if hasattr(self, "current_analysis"):
            detection_elem = ET.SubElement(root, "DETECTION_INFO")
            ET.SubElement(detection_elem, "STONE_TYPE").text = str(
                self.current_analysis.get("stone_type", "")
            )
            ET.SubElement(detection_elem, "BACKGROUND_TYPE").text = str(
                self.current_analysis.get("background_type", "")
            )
            ET.SubElement(
                detection_elem, "AVG_BRIGHTNESS"
            ).text = f"{self.current_analysis.get('avg_brightness', 0):.2f}"

            if hasattr(self, "current_params"):
                ET.SubElement(
                    detection_elem, "ALPHA_THRESHOLD"
                ).text = f"{self.current_params.get('alpha_threshold', 0):.3f}"
                ET.SubElement(detection_elem, "MORPH_KERNEL_SIZE").text = str(
                    self.current_params.get("morph_kernel_size", 0)
                )
            
            if hasattr(self, "detection_strategy"):
                ET.SubElement(detection_elem, "DETECTION_STRATEGY").text = str(self.detection_strategy)
                ET.SubElement(detection_elem, "DETECTION_QUALITY").text = f"{self.detection_quality:.3f}"

        tree = ET.ElementTree(root)
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)

    def _update_xml_with_results(self, xml_path, tree, root):
        """Update existing XML file with new processing results."""

        external_rect = getattr(self, "external_rect", (0, 0, 0, 0))
        internal_rect = getattr(self, "internal_rect", (0, 0, 0, 0))

        # Update or create attribute elements in pixels
        self._update_or_create_element(
            root, "ATTRIBUTE_EXTERNAL_X", str(external_rect[0])
        )
        self._update_or_create_element(
            root, "ATTRIBUTE_EXTERNAL_Y", str(external_rect[1])
        )
        self._update_or_create_element(
            root, "ATTRIBUTE_EXTERNAL_WIDTH", str(external_rect[2])
        )
        self._update_or_create_element(
            root, "ATTRIBUTE_EXTERNAL_HEIGHT", str(external_rect[3])
        )
        self._update_or_create_element(
            root, "ATTRIBUTE_INTERNAL_X", str(internal_rect[0])
        )
        self._update_or_create_element(
            root, "ATTRIBUTE_INTERNAL_Y", str(internal_rect[1])
        )
        self._update_or_create_element(
            root, "ATTRIBUTE_INTERNAL_WIDTH", str(internal_rect[2])
        )
        self._update_or_create_element(
            root, "ATTRIBUTE_INTERNAL_HEIGHT", str(internal_rect[3])
        )

        # Update or create attribute elements in mm
        ext_mm = self._pixels_to_mm(external_rect)
        int_mm = self._pixels_to_mm(internal_rect)

        self._update_or_create_element(
            root, "ATTRIBUTE_EXTERNAL_X_MM", f"{ext_mm[0]:.2f}"
        )
        self._update_or_create_element(
            root, "ATTRIBUTE_EXTERNAL_Y_MM", f"{ext_mm[1]:.2f}"
        )
        self._update_or_create_element(
            root, "ATTRIBUTE_EXTERNAL_WIDTH_MM", f"{ext_mm[2]:.2f}"
        )
        self._update_or_create_element(
            root, "ATTRIBUTE_EXTERNAL_HEIGHT_MM", f"{ext_mm[3]:.2f}"
        )
        self._update_or_create_element(
            root, "ATTRIBUTE_INTERNAL_X_MM", f"{int_mm[0]:.2f}"
        )
        self._update_or_create_element(
            root, "ATTRIBUTE_INTERNAL_Y_MM", f"{int_mm[1]:.2f}"
        )
        self._update_or_create_element(
            root, "ATTRIBUTE_INTERNAL_WIDTH_MM", f"{int_mm[2]:.2f}"
        )
        self._update_or_create_element(
            root, "ATTRIBUTE_INTERNAL_HEIGHT_MM", f"{int_mm[3]:.2f}"
        )

        # Update DPI information
        self._update_or_create_element(
            root, "PIXELS_PER_MM", f"{self.pixels_per_mm.get():.2f}"
        )

        # Save updated XML
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)
        print(f"Updated XML: {xml_path}")

    def _update_or_create_element(self, root, tag, text):
        """Update existing element or create new one."""
        elem = root.find(tag)
        if elem is not None:
            elem.text = text
        else:
            ET.SubElement(root, tag).text = text

    def process_xml_folder(self):
        """Process all XML files in a folder."""
        folder = filedialog.askdirectory(title="Select XML Folder")
        if not folder:
            return

        try:
            xml_files = [f for f in os.listdir(folder) if f.endswith(".xml")]

            if not xml_files:
                messagebox.showinfo("No XMLs", "No XML files found in selected folder")
                return

            processed = 0
            total_files = len(xml_files)

            self.status_label.config(text=f"Processing {total_files} XML files...")
            self.root.update()

            for i, xml_file in enumerate(xml_files, 1):
                try:
                    # Show progress
                    print(f"Processing XML {i}/{total_files}: {xml_file}")
                    self.status_label.config(
                        text=f"Processing XML {i}/{total_files}: {xml_file}"
                    )
                    self.root.update()

                    xml_path = os.path.join(folder, xml_file)
                    tree = ET.parse(xml_path)
                    root = tree.getroot()

                    image_path_elem = root.find("IMAGE_PATH")
                    idslab_elem = root.find("IDSLAB")

                    if image_path_elem is not None and idslab_elem is not None:
                        image_dir = image_path_elem.text
                        idslab = idslab_elem.text

                        image_path = None

                        if os.path.exists(image_dir):
                            for file in os.listdir(image_dir):
                                if file.lower().endswith((".jpg", ".jpeg")):
                                    # Check if the base name matches part of IDSLAB
                                    base_name = os.path.splitext(file)[0]
                                    if base_name == os.path.splitext(idslab)[0]:
                                        image_path = os.path.join(image_dir, file)
                                        break

                        if image_path:
                            print(f"Processing image: {image_path}")

                            # Load and display the input image during processing
                            self.file_path = image_path
                            img = Image.open(image_path)
                            img.thumbnail(
                                (400, 300)
                            )  # Compact size for batch processing
                            img_tk = ImageTk.PhotoImage(img)
                            self.input_image_display.config(image=img_tk)
                            self.input_image_display.image = img_tk
                            self.root.update()

                            self.perfect_detection()

                            # Update XML with new coordinates (instead of creating new XML)
                            self._update_xml_with_results(xml_path, tree, root)

                            # Save results to folders in same directory as XML
                            self._save_results_to_xml_folder(folder, idslab)
                            processed += 1

                            self.status_label.config(
                                text=f"COMPLETED {processed}/{total_files}: {idslab}"
                            )
                            print(
                                f"COMPLETED image {processed}/{total_files}: {idslab}"
                            )
                            self.root.update()
                        else:
                            print(
                                f"Image not found for IDSLAB: {idslab} in directory: {image_dir}"
                            )

                except Exception as e:
                    print(f"Error processing {xml_file}: {e}")

            # Show completion status without popup
            self.status_label.config(
                text=f"BATCH COMPLETE: {processed}/{total_files} images processed"
            )
            print(f"BATCH COMPLETE: {processed}/{total_files} images processed")

        except Exception as e:
            messagebox.showerror("Error", f"Batch processing failed: {str(e)}")


def main():
    """Launch the Perfect Slab Detector application."""
    root = tk.Tk()
    app = PerfectSlabDetector(root)
    root.mainloop()


if __name__ == "__main__":
    main()
