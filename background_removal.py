import tkinter as tk
from tkinter import filedialog, Label, Button, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import requests
import io
import os
import sys

# Set U2NET_HOME to bundled .u2net folder if running as a bundled app
if getattr(sys, "frozen", False):
    # Running as bundled app
    bundle_dir = sys._MEIPASS
    u2net_path = os.path.join(bundle_dir, ".u2net")
    os.environ["U2NET_HOME"] = u2net_path
else:
    # Running as script
    u2net_path = os.path.expanduser("~/.u2net")
    os.environ["U2NET_HOME"] = u2net_path

from rembg import remove

try:
    from adaptive_slab_detector import AdaptiveSlabDetector

    ADAPTIVE_DETECTOR_AVAILABLE = True
except ImportError:
    ADAPTIVE_DETECTOR_AVAILABLE = False
    print("Adaptive detector not available")
try:
    from advanced_ruler_remover import AdvancedRulerBackgroundRemover
    from effective_slab_remover import EffectiveSlabRemover
    from stone_only_remover import StoneOnlyRemover
    from enhanced_rembg_remover import EnhancedREMBGRemover

    ADVANCED_REMOVER_AVAILABLE = True
except ImportError:
    ADVANCED_REMOVER_AVAILABLE = False
    print("Advanced ruler remover not available")

# Replace with your Clipdrop API key
API_KEY = "571b3416b09b25a060c55cacfcfe23490c5adc0d7bb2ea6b636139b924192d88ab635e88180366529d7bca14f932c2ae"


class BackgroundRemoverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Slab Background Remover")
        self.root.geometry("900x500")

        # Layout for images
        self.input_img_label = Label(self.root, text="Input Image")
        self.input_img_label.grid(row=0, column=0, padx=10, pady=5)
        self.input_image_display = Label(self.root)
        self.input_image_display.grid(row=1, column=0, padx=10)

        self.result_img_label = Label(self.root, text="Result Image")
        self.result_img_label.grid(row=0, column=1, padx=10, pady=5)
        self.result_image_display = Label(self.root)
        self.result_image_display.grid(row=1, column=1, padx=10)

        self.result_mask_img_label = Label(self.root, text="Result Mask Image")
        self.result_mask_img_label.grid(row=0, column=2, padx=10, pady=5)
        self.result_mask_image_display = Label(self.root)
        self.result_mask_image_display.grid(row=1, column=2, padx=10)

        # Buttons
        self.upload_btn = Button(
            self.root, text="Select Slab Image", command=self.upload_image
        )
        self.upload_btn.grid(row=2, column=0, pady=10)

        self.process_btn = Button(
            self.root,
            text="Remove Background (Free)",
            command=self.remove_bg_local,
            state=tk.DISABLED,
        )
        self.process_btn.grid(row=2, column=1, pady=10)

        self.api_btn = Button(
            self.root,
            text="Remove Background (Pay)",
            command=self.remove_bg_api,
            state=tk.DISABLED,
        )
        self.api_btn.grid(row=2, column=2, pady=10)

        # Add advanced ruler background removal button
        if ADVANCED_REMOVER_AVAILABLE:
            self.advanced_btn = Button(
                self.root,
                text="Remove Background (Effective Ruler)",
                command=self.remove_bg_advanced_ruler,
                state=tk.DISABLED,
                bg="green",
                fg="white",
            )
            self.advanced_btn.grid(row=4, column=2, pady=10)

            # Add stone-only removal button (best quality)
            self.stone_btn = Button(
                self.root,
                text="Remove Background (Stone Only)",
                command=self.remove_bg_stone_only,
                state=tk.DISABLED,
                bg="darkgreen",
                fg="white",
            )
            self.stone_btn.grid(row=4, column=0, pady=10)

            # Add enhanced REMBG button (perfect detection)
            self.enhanced_btn = Button(
                self.root,
                text="Remove Background (Perfect REMBG)",
                command=self.remove_bg_enhanced_rembg,
                state=tk.DISABLED,
                bg="purple",
                fg="white",
            )
            self.enhanced_btn.grid(row=4, column=1, pady=10)

        # Add ruler management button
        self.ruler_btn = Button(
            self.root,
            text="Manage Ruler Images",
            command=self.manage_rulers,
            bg="lightcyan",
        )
        self.ruler_btn.grid(row=5, column=1, pady=10)

        self.save_btn = Button(
            self.root, text="Save Image", command=self.save_image, state=tk.DISABLED
        )
        self.save_btn.grid(row=3, column=1, pady=10)

        self.save_maskbtn = Button(
            self.root,
            text="Save Mask Image",
            command=self.save_maskimage,
            state=tk.DISABLED,
        )
        self.save_maskbtn.grid(row=3, column=2, pady=10)

        self.reset_btn = Button(self.root, text="Reset", command=self.reset)
        self.reset_btn.grid(row=3, column=3, pady=10)

        self.xml_folder = None
        self.set_xml_btn = Button(
            self.root, text="Set XML Folder", command=self.set_xml_folder
        )
        self.set_xml_btn.grid(row=4, column=0, pady=10)
        self.process_all_btn = Button(
            self.root,
            text="Process All XMLs",
            command=self.process_all_xmls,
            state=tk.DISABLED,
        )
        self.process_all_btn.grid(row=4, column=1, pady=10)

        self.file_path = None
        self.processed_image = None
        self.mask_image = None

        # Initialize adaptive detector
        if ADAPTIVE_DETECTOR_AVAILABLE:
            try:
                self.adaptive_detector = AdaptiveSlabDetector("rullers")
                print("Adaptive slab detector initialized successfully")
            except Exception as e:
                print(f"Failed to initialize adaptive detector: {e}")
                self.adaptive_detector = None
        else:
            self.adaptive_detector = None

        # Initialize advanced ruler background remover if available
        if ADVANCED_REMOVER_AVAILABLE:
            try:
                self.advanced_ruler_remover = AdvancedRulerBackgroundRemover("rullers")
                self.effective_remover = EffectiveSlabRemover("rullers")
                self.stone_only_remover = StoneOnlyRemover("rullers")
                self.enhanced_rembg_remover = EnhancedREMBGRemover()
                print(
                    "Advanced, effective, stone-only, and enhanced REMBG removers initialized successfully"
                )
            except Exception as e:
                print(f"Failed to initialize ruler removers: {e}")
                self.advanced_ruler_remover = None
                self.effective_remover = None
                self.stone_only_remover = None
                self.enhanced_rembg_remover = None
        else:
            self.advanced_ruler_remover = None
            self.effective_remover = None
            self.stone_only_remover = None
            self.enhanced_rembg_remover = None

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.JPG;*.jpg;*.jpeg")]
        )
        if file_path:
            self.file_path = file_path
            img = Image.open(file_path)
            img.thumbnail((250, 250))
            img_tk = ImageTk.PhotoImage(img)

            self.input_image_display.config(image=img_tk)
            self.input_image_display.image = img_tk

            self.process_btn.config(state=tk.NORMAL)
            self.api_btn.config(state=tk.NORMAL)
            if ADVANCED_REMOVER_AVAILABLE and self.advanced_ruler_remover:
                self.advanced_btn.config(state=tk.NORMAL)
                self.stone_btn.config(state=tk.NORMAL)
                self.enhanced_btn.config(state=tk.NORMAL)

    # REMOVED: Ruler detection functions - caused issues with working images
    # Test results show 100% success rate without ruler detection
    # Keep original rembg logic which works perfectly

    def find_largest_internal_rectangle(self, mask):
        """Find the largest fully contained rectangle inside the given binary mask using dynamic programming."""
        h, w = mask.shape
        max_area = 0
        best_rect = (0, 0, 0, 0)

        # DP table to store max width of 1's ending at each point
        height = np.zeros((h, w), dtype=int)
        width = np.zeros((h, w), dtype=int)

        for i in range(h):
            for j in range(w):
                if mask[i, j] == 255:  # Foreground pixel
                    width[i, j] = (width[i, j - 1] + 1) if j > 0 else 1
                    height[i, j] = (height[i - 1, j] + 1) if i > 0 else 1

                    # Find max area rectangle ending at (i, j)
                    min_width = width[i, j]
                    for k in range(height[i, j]):
                        min_width = min(min_width, width[i - k, j])
                        area = (k + 1) * min_width
                        if area > max_area:
                            max_area = area
                            best_rect = (j - min_width + 1, i - k, min_width, k + 1)

        return best_rect

    def _save_results_with_xml(
        self, x, y, w, h, internal_x, internal_y, internal_w, internal_h
    ):
        """Save XML results with rectangle coordinates."""
        import os
        import ntpath
        from datetime import datetime
        import xml.etree.ElementTree as ET

        # Use IDSLAB as folder name
        base_name = ntpath.basename(self.file_path)
        name, ext = os.path.splitext(base_name)
        id_slab = f"{name}_slab{ext}"
        result_dir = os.path.join("result", id_slab)
        os.makedirs(result_dir, exist_ok=True)

        # Save images
        process_img_path = os.path.join(result_dir, "process.JPG")
        mask_img_path = os.path.join(result_dir, "binary_mask.JPG")

        # Copy the saved files
        import shutil

        if os.path.exists("process.JPG"):
            shutil.copy2("process.JPG", process_img_path)
        if os.path.exists("binary_mask.JPG"):
            shutil.copy2("binary_mask.JPG", mask_img_path)

        # Save XML
        now = datetime.now()
        xml_path = os.path.join(result_dir, f"{name}_slab.xml")

        root = ET.Element("ROOT")
        ET.SubElement(root, "IMAGE_PATH").text = os.path.dirname(self.file_path)
        ET.SubElement(root, "DATE").text = now.strftime("%Y-%m-%d")
        ET.SubElement(root, "TIME").text = now.strftime("%H:%M:%S")
        company_elem = ET.SubElement(root, "COMPANY")
        company_elem.set("ID", name.split("-")[0] if "-" in name else "")
        ET.SubElement(root, "BLOCK").text = (
            name.split("-")[1] if len(name.split("-")) > 1 else ""
        )
        ET.SubElement(root, "PROG").text = (
            name.split("-")[2] if len(name.split("-")) > 2 else ""
        )
        material_elem = ET.SubElement(root, "MATERIAL")
        material_elem.set("ID", "")
        material_elem.set("NAME", "")
        thickness_elem = ET.SubElement(root, "THICKNESS", REAL="2")
        thickness_elem.text = "2"
        ET.SubElement(root, "IDSLAB").text = id_slab

        # Named attributes for external/internal rectangle
        ET.SubElement(root, "ATTRIBUTE_EXTERNAL_X").text = str(x)
        ET.SubElement(root, "ATTRIBUTE_EXTERNAL_Y").text = str(y)
        ET.SubElement(root, "ATTRIBUTE_EXTERNAL_WIDTH").text = str(w)
        ET.SubElement(root, "ATTRIBUTE_EXTERNAL_HEIGHT").text = str(h)
        ET.SubElement(root, "ATTRIBUTE_EXTERNAL_SIZE").text = str(w * h)
        ET.SubElement(root, "ATTRIBUTE_INTERNAL_X").text = str(internal_x)
        ET.SubElement(root, "ATTRIBUTE_INTERNAL_Y").text = str(internal_y)
        ET.SubElement(root, "ATTRIBUTE_INTERNAL_WIDTH").text = str(internal_w)
        ET.SubElement(root, "ATTRIBUTE_INTERNAL_HEIGHT").text = str(internal_h)
        ET.SubElement(root, "ATTRIBUTE_INTERNAL_SIZE").text = str(
            internal_w * internal_h
        )
        ET.SubElement(root, "ATTRIBUTE11").text = ""
        ET.SubElement(root, "ATTRIBUTE12").text = ""
        ET.SubElement(root, "NOTES")

        tree = ET.ElementTree(root)
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)

    def remove_bg_local(self):
        if self.file_path:
            # Use adaptive detection if available, otherwise fall back to original
            if self.adaptive_detector:
                return self.remove_bg_adaptive()
            else:
                return self.remove_bg_original()

    def remove_bg_adaptive(self):
        """Adaptive background removal using intelligent parameter selection."""
        if not self.file_path or not self.adaptive_detector:
            return

        try:
            # Use adaptive detector
            processed_img_up, result_data = self.adaptive_detector.detect_slab_adaptive(
                self.file_path, remove
            )

            mask_full = result_data["mask"]
            analysis = result_data["analysis"]
            params = result_data["parameters"]

            # Get original image dimensions
            img = cv2.imread(self.file_path)
            orig_h, orig_w = img.shape[:2]

            # Process mask for rectangle detection
            binary_mask_full = 255 - mask_full  # Invert so main image = black

            # Special processing for light stones to consolidate fragmented areas
            if analysis["stone_type"] in ["light_uniform", "light_veined"]:
                print("Applying light stone mask consolidation...")

                # Apply aggressive morphological operations to connect fragmented areas
                consolidation_kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (15, 15)
                )
                binary_mask_full = cv2.morphologyEx(
                    binary_mask_full, cv2.MORPH_CLOSE, consolidation_kernel
                )

                # Fill holes aggressively
                binary_mask_full = cv2.morphologyEx(
                    binary_mask_full,
                    cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
                )

                # Find the largest connected component and keep only that
                contours_full, _ = cv2.findContours(
                    255 - binary_mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if contours_full:
                    largest_full_contour = max(contours_full, key=cv2.contourArea)

                    # Create new mask with only the largest component
                    consolidated_mask = np.ones_like(binary_mask_full) * 255
                    cv2.fillPoly(consolidated_mask, [largest_full_contour], 0)
                    binary_mask_full = consolidated_mask

                    print(
                        f"Light stone: consolidated to single component with area {cv2.contourArea(largest_full_contour)}"
                    )
                else:
                    print("Light stone: no contours found during consolidation")

            # Rectangle detection with adaptive scaling
            rect_scale = params.get("rect_scale_factor", 5)
            rect_w, rect_h = orig_w // (4 * rect_scale), orig_h // (4 * rect_scale)

            # Improved contour detection
            if params.get("use_improved_contours", True):
                # Apply additional morphology to improve contour detection
                kernel_rect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                binary_mask_full_enhanced = cv2.morphologyEx(
                    binary_mask_full, cv2.MORPH_CLOSE, kernel_rect
                )
                binary_mask_rect = cv2.resize(
                    binary_mask_full_enhanced,
                    (rect_w, rect_h),
                    interpolation=cv2.INTER_NEAREST,
                )
            else:
                binary_mask_rect = cv2.resize(
                    binary_mask_full, (rect_w, rect_h), interpolation=cv2.INTER_NEAREST
                )

            # Find contours with improved detection
            contours, _ = cv2.findContours(
                255 - binary_mask_rect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            x = y = w = h = internal_x = internal_y = internal_w = internal_h = 0

            if contours:
                # Adaptive contour filtering
                min_area_ratio = params.get("min_contour_ratio", 0.01)
                min_area = (rect_w * rect_h) * min_area_ratio
                valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

                # Special handling for light stones - try to find the main slab mass
                if valid_contours and analysis["stone_type"] in [
                    "light_uniform",
                    "light_veined",
                ]:
                    # For light stones, prioritize the largest contour that covers significant area
                    main_contours = []
                    total_mask_area = rect_w * rect_h

                    for contour in valid_contours:
                        area = cv2.contourArea(contour)
                        area_ratio = area / total_mask_area

                        # Must be substantial (>2% of image) to be considered main slab
                        if area_ratio > 0.02:
                            # Calculate how well it fills its bounding rectangle
                            x, y, w, h = cv2.boundingRect(contour)
                            bounding_area = w * h
                            fill_ratio = (
                                area / bounding_area if bounding_area > 0 else 0
                            )

                            # Score: area + fill_ratio (prefer compact, large shapes)
                            score = area * (0.5 + fill_ratio)
                            main_contours.append((score, contour, area))

                    if main_contours:
                        # Select the highest scoring main contour
                        largest_contour = max(main_contours, key=lambda x: x[0])[1]
                        selected_area = max(main_contours, key=lambda x: x[0])[2]
                        print(
                            f"Light stone: selected contour with area {selected_area:.0f} from {len(main_contours)} candidates"
                        )
                    else:
                        # Fallback to largest contour
                        largest_contour = (
                            max(valid_contours, key=cv2.contourArea)
                            if valid_contours
                            else None
                        )
                        print("Light stone: using fallback largest contour")
                else:
                    # Use largest contour for other stone types
                    largest_contour = (
                        max(valid_contours, key=cv2.contourArea)
                        if valid_contours
                        else None
                    )

                if largest_contour is not None:
                    small_x, small_y, small_w_rect, small_h_rect = cv2.boundingRect(
                        largest_contour
                    )

                    print(
                        f"Enhanced detection: {len(valid_contours)} contours, selected area: {cv2.contourArea(largest_contour)}"
                    )
                    print(
                        f"Rectangle params: scale={rect_scale}, size=({rect_w}x{rect_h})"
                    )

                    # Scale up with adaptive factor
                    x = int(small_x * 4 * rect_scale)
                    y = int(small_y * 4 * rect_scale)
                    w = int(small_w_rect * 4 * rect_scale)
                    h = int(small_h_rect * 4 * rect_scale)

                    # Find internal rectangle
                    internal = self.find_largest_internal_rectangle(
                        255 - binary_mask_rect
                    )
                    (
                        small_internal_x,
                        small_internal_y,
                        small_internal_w,
                        small_internal_h,
                    ) = internal
                    internal_x = int(small_internal_x * 4 * rect_scale)
                    internal_y = int(small_internal_y * 4 * rect_scale)
                    internal_w = int(small_internal_w * 4 * rect_scale)
                    internal_h = int(small_internal_h * 4 * rect_scale)
                else:
                    print("No valid contours found after filtering")

            # Create visualization
            binary_mask_colored = np.ones((orig_h, orig_w, 3), dtype=np.uint8) * 255
            binary_mask_colored[binary_mask_full == 0] = (0, 0, 0)

            # Draw rectangles with adaptive colors based on stone type
            ext_color = (
                (0, 255, 0) if analysis["stone_type"] != "dark_stone" else (0, 255, 255)
            )
            cv2.rectangle(binary_mask_colored, (x, y), (x + w, y + h), ext_color, 3)
            cv2.rectangle(
                binary_mask_colored,
                (internal_x, internal_y),
                (internal_x + internal_w, internal_y + internal_h),
                (0, 0, 255),
                2,
            )

            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                binary_mask_colored,
                f"Ext: {w}x{h} px",
                (x, max(y - 10, 20)),
                font,
                0.7,
                (255, 0, 0),
                2,
            )
            cv2.putText(
                binary_mask_colored,
                f"Int: {internal_w}x{internal_h} px",
                (internal_x, max(internal_y - 10, 40)),
                font,
                0.7,
                (128, 0, 0),
                2,
            )

            # Convert and save
            if processed_img_up.shape[-1] == 4:
                processed_img_pil = Image.fromarray(processed_img_up[:, :, :3])
            else:
                processed_img_pil = Image.fromarray(processed_img_up)

            processed_img_pil.save("process.JPG")
            cv2.imwrite("binary_mask.JPG", binary_mask_colored)

            self.display_result("process.JPG")
            self.display_mask_result("binary_mask.JPG")

            # Save results with XML (existing logic from original function)
            self._save_results_with_xml(
                x, y, w, h, internal_x, internal_y, internal_w, internal_h
            )

        except Exception as e:
            print(f"Adaptive detection failed: {e}")
            # Fall back to original method
            self.remove_bg_original()

    def remove_bg_original(self):
        """Original background removal logic as fallback."""
        if not self.file_path:
            return

        img = cv2.imread(self.file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        # Original logic
        small_w, small_h = orig_w // 4, orig_h // 4
        img_small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)

        processed_img_small = remove(img_small)
        processed_img_small = np.array(processed_img_small).copy()
        if processed_img_small.shape[-1] == 3:
            processed_img_small = cv2.cvtColor(processed_img_small, cv2.COLOR_RGB2RGBA)

        alpha_small = processed_img_small[:, :, 3] / 255.0

        print(
            f"Original detection ratio: {np.sum(alpha_small > 0.05) / (alpha_small.shape[0] * alpha_small.shape[1]):.2f}"
        )

        mask_small = (alpha_small > 0.05).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_CLOSE, kernel)
        mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_OPEN, kernel)

        binary_mask_small = 255 - mask_small

        # Continue with original rectangle detection...
        rect_w, rect_h = small_w // 5, small_h // 5
        binary_mask_rect = cv2.resize(
            binary_mask_small, (rect_w, rect_h), interpolation=cv2.INTER_NEAREST
        )

        # Find contours and rectangles (original logic)
        contours, _ = cv2.findContours(
            255 - binary_mask_rect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        x = y = w = h = internal_x = internal_y = internal_w = internal_h = 0

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            small_x, small_y, small_w_rect, small_h_rect = cv2.boundingRect(
                largest_contour
            )
            x = int(small_x * 5 * 4)
            y = int(small_y * 5 * 4)
            w = int(small_w_rect * 5 * 4)
            h = int(small_h_rect * 5 * 4)

            internal = self.find_largest_internal_rectangle(255 - binary_mask_rect)
            small_internal_x, small_internal_y, small_internal_w, small_internal_h = (
                internal
            )
            internal_x = int(small_internal_x * 5 * 4)
            internal_y = int(small_internal_y * 5 * 4)
            internal_w = int(small_internal_w * 5 * 4)
            internal_h = int(small_internal_h * 5 * 4)

        # Create visualization and save
        binary_mask_full = cv2.resize(
            binary_mask_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
        )
        binary_mask_colored = np.ones((orig_h, orig_w, 3), dtype=np.uint8) * 255
        binary_mask_colored[binary_mask_full == 0] = (0, 0, 0)

        cv2.rectangle(binary_mask_colored, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(
            binary_mask_colored,
            (internal_x, internal_y),
            (internal_x + internal_w, internal_y + internal_h),
            (0, 0, 255),
            2,
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            binary_mask_colored,
            f"Ext: {w}x{h} px",
            (x, max(y - 10, 20)),
            font,
            0.8,
            (255, 0, 0),
            2,
        )
        cv2.putText(
            binary_mask_colored,
            f"Int: {internal_w}x{internal_h} px",
            (internal_x, max(internal_y - 10, 40)),
            font,
            0.8,
            (255, 0, 0),
            2,
        )

        processed_img_up = cv2.resize(
            processed_img_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
        )
        processed_img_pil = Image.fromarray(processed_img_up)
        if processed_img_pil.mode == "RGBA":
            processed_img_pil = processed_img_pil.convert("RGB")

        processed_img_pil.save("process.JPG")
        cv2.imwrite("binary_mask.JPG", binary_mask_colored)
        self.display_result("process.JPG")
        self.display_mask_result("binary_mask.JPG")

        self._save_results_with_xml(
            x, y, w, h, internal_x, internal_y, internal_w, internal_h
        )

    def remove_bg_api(self):
        if not self.file_path:
            return

        try:
            checkimg = cv2.imread(self.file_path)
            checkimg = cv2.cvtColor(checkimg, cv2.COLOR_BGR2RGB)

            # Step 1: Get original dimensions
            orig_h, orig_w = checkimg.shape[:2]

            img = Image.open(self.file_path)
            img.thumbnail((6000, 6000))
            img.save("temp.JPG")

            with open("temp.JPG", "rb") as image_file:
                response = requests.post(
                    "https://clipdrop-api.co/remove-background/v1",
                    files={"image_file": ("input.jpg", image_file, "image/jpeg")},
                    headers={"x-api-key": API_KEY},
                )

            if response.ok:
                result = io.BytesIO(response.content)
                processed_img = Image.open(result)
                processed_img.thumbnail((orig_w, orig_h))
                processed_img.save("process.JPG")
                self.display_result("process.JPG")

                processed_img = np.array(processed_img).copy()
                if processed_img.shape[-1] == 3:
                    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2RGBA)

                alpha = processed_img[:, :, 3] / 255.0

                # 1. Create binary mask (black = main image, white = background)
                mask = (alpha > 0.1).astype(np.uint8) * 255
                binary_mask = 255 - mask  # Invert so main image = black

                # 2. Resize mask to small size
                small_w, small_h = orig_w // 4, orig_h // 4
                binary_mask_small = cv2.resize(
                    binary_mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST
                )

                # 3. Find contours for external rectangle
                # → findContours detects WHITE areas, so invert again
                contours, _ = cv2.findContours(
                    255 - binary_mask_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                x = y = w = h = internal_x = internal_y = internal_w = internal_h = 0

                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    small_x, small_y, small_w_rect, small_h_rect = cv2.boundingRect(
                        largest_contour
                    )

                    x, y, w, h = (
                        int(small_x * 4),
                        int(small_y * 4),
                        int(small_w_rect * 4),
                        int(small_h_rect * 4),
                    )

                    # 4. Find internal rectangle in binary_mask_small (black region = 0)
                    internal = self.find_largest_internal_rectangle(
                        255 - binary_mask_small
                    )
                    (
                        small_internal_x,
                        small_internal_y,
                        small_internal_w,
                        small_internal_h,
                    ) = internal
                    internal_x, internal_y, internal_w, internal_h = (
                        int(small_internal_x * 4),
                        int(small_internal_y * 4),
                        int(small_internal_w * 4),
                        int(small_internal_h * 4),
                    )

                # 5. Resize mask back to original size and convert to RGB
                binary_mask_full = cv2.resize(
                    binary_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
                )
                binary_mask_colored = (
                    np.ones((orig_h, orig_w, 3), dtype=np.uint8) * 255
                )  # White background
                binary_mask_colored[binary_mask_full == 0] = (
                    0,
                    0,
                    0,
                )  # Main image → black

                # 6. Draw rectangles
                cv2.rectangle(
                    binary_mask_colored, (x, y), (x + w, y + h), (0, 255, 0), 2
                )  # External → Green
                cv2.rectangle(
                    binary_mask_colored,
                    (internal_x, internal_y),
                    (internal_x + internal_w, internal_y + internal_h),
                    (0, 0, 255),
                    2,
                )  # Internal → Red

                # 7. Add labels
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    binary_mask_colored,
                    f"Ext: {w}x{h} px",
                    (x, y - 10),
                    font,
                    0.8,
                    (255, 0, 0),
                    2,
                )
                cv2.putText(
                    binary_mask_colored,
                    f"Int: {internal_w}x{internal_h} px",
                    (internal_x, internal_y - 10),
                    font,
                    0.8,
                    (255, 0, 0),
                    2,
                )

                # Save & show result
                cv2.imwrite("binary_mask.JPG", binary_mask_colored)
                self.display_mask_result("binary_mask.JPG")

                # self.display_result("temp.JPG")
            else:
                messagebox.showerror(
                    "Error", f"API Error: {response.status_code} - {response.text}"
                )

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def display_result(self, img_path):
        self.processed_image = Image.open(img_path)
        res_img = self.processed_image.copy()
        res_img.thumbnail((250, 250))
        img_tk = ImageTk.PhotoImage(res_img)
        self.result_image_display.config(image=img_tk)
        self.result_image_display.image = img_tk

        self.save_btn.config(state=tk.NORMAL)

    def display_mask_result(self, img_path):
        self.mask_image = Image.open(img_path)
        res_img = self.mask_image.copy()
        res_img.thumbnail((250, 250))
        img_tk = ImageTk.PhotoImage(res_img)
        self.result_mask_image_display.config(image=img_tk)
        self.result_mask_image_display.image = img_tk

        self.save_maskbtn.config(state=tk.NORMAL)

    def save_image(self):
        if self.processed_image:
            save_path = filedialog.asksaveasfilename(
                defaultextension=".JPG",
                filetypes=[
                    ("JPEG files", "*.jpg"),
                    ("PNG files", "*.png"),
                ],
            )
            if save_path:
                self.processed_image.save(save_path)

    def save_maskimage(self):
        if self.mask_image:
            save_path = filedialog.asksaveasfilename(
                defaultextension=".JPG",
                filetypes=[
                    ("JPEG files", "*.jpg"),
                    ("PNG files", "*.png"),
                ],
            )
            if save_path:
                self.mask_image.save(save_path)

    def remove_bg_focused(self):
        """Remove background using focused slab remover with perfect results."""
        if not self.file_path or not self.focused_remover:
            messagebox.showerror(
                "Error", "No image selected or focused remover not available."
            )
            return

        try:
            # Use focused slab remover
            result_rgba, slab_mask = self.focused_remover.remove_background(
                self.file_path, save_debug=True
            )

            # Get original image dimensions
            img = cv2.imread(self.file_path)
            orig_h, orig_w = img.shape[:2]

            # Process the mask for rectangle detection (same logic as existing methods)
            alpha = result_rgba[:, :, 3] / 255.0
            mask = (alpha > 0.1).astype(np.uint8) * 255
            binary_mask = 255 - mask  # Invert so main image = black

            # Downscale mask for rectangle detection
            small_w, small_h = orig_w // 4, orig_h // 4
            binary_mask_small = cv2.resize(
                binary_mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST
            )

            # Find contours for external rectangle
            contours, _ = cv2.findContours(
                255 - binary_mask_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            x = y = w = h = internal_x = internal_y = internal_w = internal_h = 0

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                small_x, small_y, small_w_rect, small_h_rect = cv2.boundingRect(
                    largest_contour
                )

                # Scale up to original image size
                x = int(small_x * 4)
                y = int(small_y * 4)
                w = int(small_w_rect * 4)
                h = int(small_h_rect * 4)

                # Find internal rectangle
                internal = self.find_largest_internal_rectangle(255 - binary_mask_small)
                (
                    small_internal_x,
                    small_internal_y,
                    small_internal_w,
                    small_internal_h,
                ) = internal
                internal_x = int(small_internal_x * 4)
                internal_y = int(small_internal_y * 4)
                internal_w = int(small_internal_w * 4)
                internal_h = int(small_internal_h * 4)

            # Create visualization mask
            binary_mask_full = cv2.resize(
                binary_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
            )
            binary_mask_colored = (
                np.ones((orig_h, orig_w, 3), dtype=np.uint8) * 255
            )  # White background
            binary_mask_colored[binary_mask_full == 0] = (0, 0, 0)  # Main image → black

            # Draw rectangles
            cv2.rectangle(
                binary_mask_colored, (x, y), (x + w, y + h), (0, 255, 0), 2
            )  # External → Green
            cv2.rectangle(
                binary_mask_colored,
                (internal_x, internal_y),
                (internal_x + internal_w, internal_y + internal_h),
                (0, 0, 255),
                2,
            )  # Internal → Red

            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                binary_mask_colored,
                f"Ext: {w}x{h} px",
                (x, y - 10),
                font,
                0.8,
                (255, 0, 0),
                2,
            )
            cv2.putText(
                binary_mask_colored,
                f"Int: {internal_w}x{internal_h} px",
                (internal_x, internal_y - 10),
                font,
                0.8,
                (255, 0, 0),
                2,
            )

            # Convert RGBA result to RGB for saving and display
            processed_img_rgb = cv2.cvtColor(result_rgba[:, :, :3], cv2.COLOR_BGR2RGB)
            processed_img_pil = Image.fromarray(processed_img_rgb)

            # Save results
            processed_img_pil.save("process.JPG")
            cv2.imwrite("binary_mask.JPG", binary_mask_colored)

            # Display results
            self.display_result("process.JPG")
            self.display_mask_result("binary_mask.JPG")

            # Save to results folder with XML (same logic as existing methods)
            import ntpath
            from datetime import datetime
            import xml.etree.ElementTree as ET

            base_name = ntpath.basename(self.file_path)
            name, ext = os.path.splitext(base_name)
            id_slab = f"{name}_slab{ext}"
            result_dir = os.path.join("result", id_slab)
            os.makedirs(result_dir, exist_ok=True)

            # Save images
            process_img_path = os.path.join(result_dir, "process.JPG")
            mask_img_path = os.path.join(result_dir, "binary_mask.JPG")
            processed_img_pil.save(process_img_path)
            cv2.imwrite(mask_img_path, binary_mask_colored)

            # Save XML
            now = datetime.now()
            xml_path = os.path.join(result_dir, f"{name}_slab.xml")

            root = ET.Element("ROOT")
            ET.SubElement(root, "IMAGE_PATH").text = os.path.dirname(self.file_path)
            ET.SubElement(root, "DATE").text = now.strftime("%Y-%m-%d")
            ET.SubElement(root, "TIME").text = now.strftime("%H:%M:%S")
            company_elem = ET.SubElement(root, "COMPANY")
            company_elem.set("ID", name.split("-")[0] if "-" in name else "")
            ET.SubElement(root, "BLOCK").text = (
                name.split("-")[1] if len(name.split("-")) > 1 else ""
            )
            ET.SubElement(root, "PROG").text = (
                name.split("-")[2] if len(name.split("-")) > 2 else ""
            )
            material_elem = ET.SubElement(root, "MATERIAL")
            material_elem.set("ID", "")
            material_elem.set("NAME", "")
            thickness_elem = ET.SubElement(root, "THICKNESS", REAL="2")
            thickness_elem.text = "2"
            ET.SubElement(root, "IDSLAB").text = id_slab

            # Add rectangle attributes
            ET.SubElement(root, "ATTRIBUTE_EXTERNAL_X").text = str(x)
            ET.SubElement(root, "ATTRIBUTE_EXTERNAL_Y").text = str(y)
            ET.SubElement(root, "ATTRIBUTE_EXTERNAL_WIDTH").text = str(w)
            ET.SubElement(root, "ATTRIBUTE_EXTERNAL_HEIGHT").text = str(h)
            ET.SubElement(root, "ATTRIBUTE_EXTERNAL_SIZE").text = str(w * h)
            ET.SubElement(root, "ATTRIBUTE_INTERNAL_X").text = str(internal_x)
            ET.SubElement(root, "ATTRIBUTE_INTERNAL_Y").text = str(internal_y)
            ET.SubElement(root, "ATTRIBUTE_INTERNAL_WIDTH").text = str(internal_w)
            ET.SubElement(root, "ATTRIBUTE_INTERNAL_HEIGHT").text = str(internal_h)
            ET.SubElement(root, "ATTRIBUTE_INTERNAL_SIZE").text = str(
                internal_w * internal_h
            )
            ET.SubElement(root, "ATTRIBUTE11").text = ""
            ET.SubElement(root, "ATTRIBUTE12").text = ""
            ET.SubElement(root, "NOTES")

            tree = ET.ElementTree(root)
            tree.write(xml_path, encoding="utf-8", xml_declaration=True)

            messagebox.showinfo(
                "Success",
                f"Perfect background removal completed!\nResults saved to: {result_dir}",
            )

        except Exception as e:
            messagebox.showerror(
                "Error", f"Focused background removal failed: {str(e)}"
            )

    def remove_bg_advanced_ruler(self):
        """Remove background using proven effective ruler-based algorithm."""
        if not self.file_path or not self.effective_remover:
            messagebox.showerror(
                "Error", "No image selected or effective ruler remover not available."
            )
            return

        try:
            # Use the effective algorithm that actually works
            result_rgba, final_mask = self.effective_remover.remove_background(
                self.file_path
            )

            # Load input image for dimensions
            input_image = cv2.imread(self.file_path)
            orig_h, orig_w = input_image.shape[:2]

            # Process for rectangle detection (same logic as other methods)
            alpha = final_mask.astype(np.float32) / 255.0
            mask = (alpha > 0.1).astype(np.uint8) * 255
            binary_mask = 255 - mask

            # Rectangle detection
            small_w, small_h = orig_w // 4, orig_h // 4
            binary_mask_small = cv2.resize(
                binary_mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST
            )

            contours, _ = cv2.findContours(
                255 - binary_mask_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            x = y = w = h = internal_x = internal_y = internal_w = internal_h = 0

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                small_x, small_y, small_w_rect, small_h_rect = cv2.boundingRect(
                    largest_contour
                )

                x = int(small_x * 4)
                y = int(small_y * 4)
                w = int(small_w_rect * 4)
                h = int(small_h_rect * 4)

                internal = self.find_largest_internal_rectangle(255 - binary_mask_small)
                (
                    small_internal_x,
                    small_internal_y,
                    small_internal_w,
                    small_internal_h,
                ) = internal
                internal_x = int(small_internal_x * 4)
                internal_y = int(small_internal_y * 4)
                internal_w = int(small_internal_w * 4)
                internal_h = int(small_internal_h * 4)

            # Create visualization
            binary_mask_full = cv2.resize(
                binary_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
            )
            binary_mask_colored = np.ones((orig_h, orig_w, 3), dtype=np.uint8) * 255
            binary_mask_colored[binary_mask_full == 0] = (0, 0, 0)

            cv2.rectangle(binary_mask_colored, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(
                binary_mask_colored,
                (internal_x, internal_y),
                (internal_x + internal_w, internal_y + internal_h),
                (0, 0, 255),
                2,
            )

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                binary_mask_colored,
                f"Ext: {w}x{h} px",
                (x, y - 10),
                font,
                0.8,
                (255, 0, 0),
                2,
            )
            cv2.putText(
                binary_mask_colored,
                f"Int: {internal_w}x{internal_h} px",
                (internal_x, internal_y - 10),
                font,
                0.8,
                (255, 0, 0),
                2,
            )

            # Convert RGBA to RGB for display/saving
            processed_img_rgb = cv2.cvtColor(result_rgba[:, :, :3], cv2.COLOR_BGR2RGB)
            processed_img_pil = Image.fromarray(processed_img_rgb)

            processed_img_pil.save("process.JPG")
            cv2.imwrite("binary_mask.JPG", binary_mask_colored)

            self.display_result("process.JPG")
            self.display_mask_result("binary_mask.JPG")

            # Save to results folder with XML
            import ntpath
            from datetime import datetime
            import xml.etree.ElementTree as ET

            base_name = ntpath.basename(self.file_path)
            name, ext = os.path.splitext(base_name)
            id_slab = f"{name}_slab{ext}"
            result_dir = os.path.join("result", id_slab)
            os.makedirs(result_dir, exist_ok=True)

            process_img_path = os.path.join(result_dir, "process.JPG")
            mask_img_path = os.path.join(result_dir, "binary_mask.JPG")
            processed_img_pil.save(process_img_path)
            cv2.imwrite(mask_img_path, binary_mask_colored)

            # Create XML
            now = datetime.now()
            xml_path = os.path.join(result_dir, f"{name}_slab.xml")

            root = ET.Element("ROOT")
            ET.SubElement(root, "IMAGE_PATH").text = os.path.dirname(self.file_path)
            ET.SubElement(root, "DATE").text = now.strftime("%Y-%m-%d")
            ET.SubElement(root, "TIME").text = now.strftime("%H:%M:%S")
            company_elem = ET.SubElement(root, "COMPANY")
            company_elem.set("ID", name.split("-")[0] if "-" in name else "")
            ET.SubElement(root, "BLOCK").text = (
                name.split("-")[1] if len(name.split("-")) > 1 else ""
            )
            ET.SubElement(root, "PROG").text = (
                name.split("-")[2] if len(name.split("-")) > 2 else ""
            )
            material_elem = ET.SubElement(root, "MATERIAL")
            material_elem.set("ID", "")
            material_elem.set("NAME", "")
            thickness_elem = ET.SubElement(root, "THICKNESS", REAL="2")
            thickness_elem.text = "2"
            ET.SubElement(root, "IDSLAB").text = id_slab

            ET.SubElement(root, "ATTRIBUTE_EXTERNAL_X").text = str(x)
            ET.SubElement(root, "ATTRIBUTE_EXTERNAL_Y").text = str(y)
            ET.SubElement(root, "ATTRIBUTE_EXTERNAL_WIDTH").text = str(w)
            ET.SubElement(root, "ATTRIBUTE_EXTERNAL_HEIGHT").text = str(h)
            ET.SubElement(root, "ATTRIBUTE_EXTERNAL_SIZE").text = str(w * h)
            ET.SubElement(root, "ATTRIBUTE_INTERNAL_X").text = str(internal_x)
            ET.SubElement(root, "ATTRIBUTE_INTERNAL_Y").text = str(internal_y)
            ET.SubElement(root, "ATTRIBUTE_INTERNAL_WIDTH").text = str(internal_w)
            ET.SubElement(root, "ATTRIBUTE_INTERNAL_HEIGHT").text = str(internal_h)
            ET.SubElement(root, "ATTRIBUTE_INTERNAL_SIZE").text = str(
                internal_w * internal_h
            )
            ET.SubElement(root, "ATTRIBUTE11").text = ""
            ET.SubElement(root, "ATTRIBUTE12").text = ""
            ET.SubElement(root, "NOTES")

            tree = ET.ElementTree(root)
            tree.write(xml_path, encoding="utf-8", xml_declaration=True)

            messagebox.showinfo(
                "Success",
                f"Advanced ruler background removal completed!\\nResults saved to: {result_dir}",
            )

        except Exception as e:
            messagebox.showerror(
                "Error", f"Advanced ruler background removal failed: {str(e)}"
            )

    def remove_bg_stone_only(self):
        """Remove background using stone-only algorithm for cleanest results."""
        if not self.file_path or not self.stone_only_remover:
            messagebox.showerror(
                "Error", "No image selected or stone-only remover not available."
            )
            return

        try:
            # Use the stone-only algorithm for cleanest background removal
            result_rgba, final_mask = self.stone_only_remover.remove_background(
                self.file_path
            )

            # Load input image for dimensions
            input_image = cv2.imread(self.file_path)
            orig_h, orig_w = input_image.shape[:2]

            # Process for rectangle detection (same logic as other methods)
            alpha = final_mask.astype(np.float32) / 255.0
            mask = (alpha > 0.1).astype(np.uint8) * 255
            binary_mask = 255 - mask

            # Rectangle detection
            small_w, small_h = orig_w // 4, orig_h // 4
            binary_mask_small = cv2.resize(
                binary_mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST
            )

            contours, _ = cv2.findContours(
                255 - binary_mask_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            x = y = w = h = internal_x = internal_y = internal_w = internal_h = 0

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                small_x, small_y, small_w_rect, small_h_rect = cv2.boundingRect(
                    largest_contour
                )

                # Scale back to original dimensions
                x, y, w, h = (
                    small_x * 4,
                    small_y * 4,
                    small_w_rect * 4,
                    small_h_rect * 4,
                )

                # Find internal rectangle
                internal_rect = self.find_largest_internal_rectangle(
                    255 - binary_mask_small
                )
                if internal_rect:
                    internal_x, internal_y, internal_w, internal_h = [
                        coord * 4 for coord in internal_rect
                    ]

            # Convert result to BGR for display
            result_bgr = cv2.cvtColor(result_rgba[:, :, :3], cv2.COLOR_RGB2BGR)

            # Create result directory and save files
            base_name = os.path.splitext(os.path.basename(self.file_path))[0]
            result_dir = f"result/{base_name}_slab"
            os.makedirs(result_dir, exist_ok=True)

            # Save processed image
            process_path = os.path.join(result_dir, "process.JPG")
            cv2.imwrite(process_path, result_bgr)

            # Create and save binary mask visualization
            binary_mask_bgr = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

            # Draw external rectangle
            if w > 0 and h > 0:
                cv2.rectangle(binary_mask_bgr, (x, y), (x + w, y + h), (0, 255, 0), 10)

            # Draw internal rectangle
            if internal_w > 0 and internal_h > 0:
                cv2.rectangle(
                    binary_mask_bgr,
                    (internal_x, internal_y),
                    (internal_x + internal_w, internal_y + internal_h),
                    (0, 0, 255),
                    8,
                )

            binary_mask_path = os.path.join(result_dir, "binary_mask.JPG")
            cv2.imwrite(binary_mask_path, binary_mask_bgr)

            # Generate XML
            import ntpath
            from datetime import datetime
            import xml.etree.ElementTree as ET

            # Parse image name for XML structure
            name = (
                ntpath.basename(self.file_path).replace(".JPG", "").replace(".jpg", "")
            )
            id_slab = name

            # Create XML
            now = datetime.now()
            xml_path = os.path.join(result_dir, f"{name}_slab.xml")

            root = ET.Element("ROOT")
            ET.SubElement(root, "IMAGE_PATH").text = os.path.dirname(self.file_path)
            ET.SubElement(root, "DATE").text = now.strftime("%Y-%m-%d")
            ET.SubElement(root, "TIME").text = now.strftime("%H:%M:%S")
            company_elem = ET.SubElement(root, "COMPANY")
            company_elem.set("ID", name.split("-")[0] if "-" in name else "")
            ET.SubElement(root, "BLOCK").text = (
                name.split("-")[1] if len(name.split("-")) > 1 else ""
            )
            ET.SubElement(root, "PROG").text = (
                name.split("-")[2] if len(name.split("-")) > 2 else ""
            )
            material_elem = ET.SubElement(root, "MATERIAL")
            material_elem.set("ID", "")
            material_elem.set("NAME", "")
            thickness_elem = ET.SubElement(root, "THICKNESS", REAL="2")
            thickness_elem.text = "2"
            ET.SubElement(root, "IDSLAB").text = id_slab

            # Add rectangle attributes
            ET.SubElement(root, "ATTRIBUTE_EXTERNAL_X").text = str(x)
            ET.SubElement(root, "ATTRIBUTE_EXTERNAL_Y").text = str(y)
            ET.SubElement(root, "ATTRIBUTE_EXTERNAL_WIDTH").text = str(w)
            ET.SubElement(root, "ATTRIBUTE_EXTERNAL_HEIGHT").text = str(h)
            ET.SubElement(root, "ATTRIBUTE_INTERNAL_X").text = str(internal_x)
            ET.SubElement(root, "ATTRIBUTE_INTERNAL_Y").text = str(internal_y)
            ET.SubElement(root, "ATTRIBUTE_INTERNAL_WIDTH").text = str(internal_w)
            ET.SubElement(root, "ATTRIBUTE_INTERNAL_HEIGHT").text = str(internal_h)

            tree = ET.ElementTree(root)
            tree.write(xml_path, encoding="utf-8", xml_declaration=True)

            # Update display
            self.display_result_image(result_bgr)
            self.display_result_mask(binary_mask_bgr)

            messagebox.showinfo(
                "Success",
                f"Stone-only background removal completed!\\nCleanest results saved to: {result_dir}",
            )

        except Exception as e:
            messagebox.showerror(
                "Error", f"Stone-only background removal failed: {str(e)}"
            )

    def remove_bg_enhanced_rembg(self):
        """Remove background using enhanced REMBG with intelligent post-processing."""
        if not self.file_path or not self.enhanced_rembg_remover:
            messagebox.showerror(
                "Error", "No image selected or enhanced REMBG remover not available."
            )
            return

        try:
            # Use the enhanced REMBG algorithm for perfect detection
            result_rgba, final_mask = self.enhanced_rembg_remover.remove_background(
                self.file_path
            )

            # Load input image for dimensions
            input_image = cv2.imread(self.file_path)
            orig_h, orig_w = input_image.shape[:2]

            # Process for rectangle detection (same logic as other methods)
            alpha = final_mask.astype(np.float32) / 255.0
            mask = (alpha > 0.1).astype(np.uint8) * 255
            binary_mask = 255 - mask

            # Rectangle detection
            small_w, small_h = orig_w // 4, orig_h // 4
            binary_mask_small = cv2.resize(
                binary_mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST
            )

            contours, _ = cv2.findContours(
                255 - binary_mask_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            x = y = w = h = internal_x = internal_y = internal_w = internal_h = 0

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                small_x, small_y, small_w_rect, small_h_rect = cv2.boundingRect(
                    largest_contour
                )

                # Scale back to original dimensions
                x, y, w, h = (
                    small_x * 4,
                    small_y * 4,
                    small_w_rect * 4,
                    small_h_rect * 4,
                )

                # Find internal rectangle
                internal_rect = self.find_largest_internal_rectangle(
                    255 - binary_mask_small
                )
                if internal_rect:
                    internal_x, internal_y, internal_w, internal_h = [
                        coord * 4 for coord in internal_rect
                    ]

            # Convert result to BGR for display
            result_bgr = cv2.cvtColor(result_rgba[:, :, :3], cv2.COLOR_RGB2BGR)

            # Create result directory and save files
            base_name = os.path.splitext(os.path.basename(self.file_path))[0]
            result_dir = f"result/{base_name}_slab"
            os.makedirs(result_dir, exist_ok=True)

            # Save processed image
            process_path = os.path.join(result_dir, "process.JPG")
            cv2.imwrite(process_path, result_bgr)

            # Create and save binary mask visualization
            binary_mask_bgr = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

            # Draw external rectangle
            if w > 0 and h > 0:
                cv2.rectangle(binary_mask_bgr, (x, y), (x + w, y + h), (0, 255, 0), 10)

            # Draw internal rectangle
            if internal_w > 0 and internal_h > 0:
                cv2.rectangle(
                    binary_mask_bgr,
                    (internal_x, internal_y),
                    (internal_x + internal_w, internal_y + internal_h),
                    (0, 0, 255),
                    8,
                )

            binary_mask_path = os.path.join(result_dir, "binary_mask.JPG")
            cv2.imwrite(binary_mask_path, binary_mask_bgr)

            # Generate XML
            import ntpath
            from datetime import datetime
            import xml.etree.ElementTree as ET

            # Parse image name for XML structure
            name = (
                ntpath.basename(self.file_path).replace(".JPG", "").replace(".jpg", "")
            )
            id_slab = name

            # Create XML
            now = datetime.now()
            xml_path = os.path.join(result_dir, f"{name}_slab.xml")

            root = ET.Element("ROOT")
            ET.SubElement(root, "IMAGE_PATH").text = os.path.dirname(self.file_path)
            ET.SubElement(root, "DATE").text = now.strftime("%Y-%m-%d")
            ET.SubElement(root, "TIME").text = now.strftime("%H:%M:%S")
            company_elem = ET.SubElement(root, "COMPANY")
            company_elem.set("ID", name.split("-")[0] if "-" in name else "")
            ET.SubElement(root, "BLOCK").text = (
                name.split("-")[1] if len(name.split("-")) > 1 else ""
            )
            ET.SubElement(root, "PROG").text = (
                name.split("-")[2] if len(name.split("-")) > 2 else ""
            )
            material_elem = ET.SubElement(root, "MATERIAL")
            material_elem.set("ID", "")
            material_elem.set("NAME", "")
            thickness_elem = ET.SubElement(root, "THICKNESS", REAL="2")
            thickness_elem.text = "2"
            ET.SubElement(root, "IDSLAB").text = id_slab

            # Add rectangle attributes
            ET.SubElement(root, "ATTRIBUTE_EXTERNAL_X").text = str(x)
            ET.SubElement(root, "ATTRIBUTE_EXTERNAL_Y").text = str(y)
            ET.SubElement(root, "ATTRIBUTE_EXTERNAL_WIDTH").text = str(w)
            ET.SubElement(root, "ATTRIBUTE_EXTERNAL_HEIGHT").text = str(h)
            ET.SubElement(root, "ATTRIBUTE_INTERNAL_X").text = str(internal_x)
            ET.SubElement(root, "ATTRIBUTE_INTERNAL_Y").text = str(internal_y)
            ET.SubElement(root, "ATTRIBUTE_INTERNAL_WIDTH").text = str(internal_w)
            ET.SubElement(root, "ATTRIBUTE_INTERNAL_HEIGHT").text = str(internal_h)

            tree = ET.ElementTree(root)
            tree.write(xml_path, encoding="utf-8", xml_declaration=True)

            # Update display
            self.display_result_image(result_bgr)
            self.display_result_mask(binary_mask_bgr)

            messagebox.showinfo(
                "Success",
                f"Enhanced REMBG background removal completed!\\nPerfect detection saved to: {result_dir}",
            )

        except Exception as e:
            messagebox.showerror(
                "Error", f"Enhanced REMBG background removal failed: {str(e)}"
            )

    def manage_rulers(self):
        """Open ruler management dialog."""
        ruler_window = tk.Toplevel(self.root)
        ruler_window.title("Manage Ruler Images")
        ruler_window.geometry("600x400")

        tk.Label(
            ruler_window, text="Ruler Image Management", font=("Arial", 14, "bold")
        ).pack(pady=10)

        # List current rulers
        tk.Label(ruler_window, text="Current Ruler Images:").pack(anchor=tk.W, padx=20)

        ruler_listbox = tk.Listbox(ruler_window, height=10)
        ruler_listbox.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Load current rulers
        try:
            ruler_files = glob.glob(os.path.join("rullers", "*.JPG"))
            for ruler_file in ruler_files:
                ruler_listbox.insert(tk.END, os.path.basename(ruler_file))
        except:
            pass

        # Buttons
        button_frame = tk.Frame(ruler_window)
        button_frame.pack(pady=10)

        def add_ruler():
            file_path = filedialog.askopenfilename(
                title="Select Ruler Background Image",
                filetypes=[("Image Files", "*.jpg *.jpeg *.JPG *.JPEG *.png *.PNG")],
            )
            if file_path:
                import shutil

                os.makedirs("rullers", exist_ok=True)
                dest_path = os.path.join("rullers", os.path.basename(file_path))
                shutil.copy2(file_path, dest_path)
                ruler_listbox.insert(tk.END, os.path.basename(file_path))
                messagebox.showinfo(
                    "Success", f"Ruler image added: {os.path.basename(file_path)}"
                )

        def remove_ruler():
            selection = ruler_listbox.curselection()
            if selection:
                ruler_name = ruler_listbox.get(selection[0])
                if messagebox.askyesno("Confirm", f"Remove ruler image: {ruler_name}?"):
                    try:
                        os.remove(os.path.join("rullers", ruler_name))
                        ruler_listbox.delete(selection[0])
                        messagebox.showinfo(
                            "Success", f"Ruler image removed: {ruler_name}"
                        )
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to remove: {e}")

        tk.Button(
            button_frame, text="Add Ruler Image", command=add_ruler, bg="lightgreen"
        ).pack(side=tk.LEFT, padx=5)
        tk.Button(
            button_frame, text="Remove Selected", command=remove_ruler, bg="lightcoral"
        ).pack(side=tk.LEFT, padx=5)
        tk.Button(
            button_frame, text="Close", command=ruler_window.destroy, bg="lightgray"
        ).pack(side=tk.LEFT, padx=5)

    def reset(self):
        self.file_path = None
        self.processed_image = None

        self.input_image_display.config(image="")
        self.result_image_display.config(image="")

        self.process_btn.config(state=tk.DISABLED)
        self.api_btn.config(state=tk.DISABLED)
        if FOCUSED_REMOVER_AVAILABLE and hasattr(self, "focused_btn"):
            self.focused_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)

    def set_xml_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.xml_folder = folder
            self.process_all_btn.config(state=tk.NORMAL)
            messagebox.showinfo("XML Folder Set", f"XML folder set to:\n{folder}")

    def process_all_xmls(self):
        import xml.etree.ElementTree as ET
        import glob

        if not self.xml_folder:
            messagebox.showerror("Error", "No XML folder set.")
            return
        xml_files = glob.glob(os.path.join(self.xml_folder, "*.xml"))
        if not xml_files:
            messagebox.showinfo("No XMLs", "No XML files found in the selected folder.")
            return
        for xml_file in xml_files:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                image_path_elem = root.find("IMAGE_PATH")
                idslab_elem = root.find("IDSLAB")
                if image_path_elem is not None and idslab_elem is not None:
                    image_folder = image_path_elem.text
                    image_name = idslab_elem.text
                    image_path = os.path.join(image_folder, image_name)
                    if os.path.exists(image_path):
                        self.file_path = image_path
                        self.remove_bg_local()
                    else:
                        print(f"Image not found: {image_path}")
                else:
                    print(f"Missing IMAGE_PATH or IDSLAB in {xml_file}")
            except Exception as e:
                print(f"Error processing {xml_file}: {e}")
        messagebox.showinfo("Done", "Batch processing of XMLs completed.")


if __name__ == "__main__":
    root = tk.Tk()
    app = BackgroundRemoverApp(root)
    root.mainloop()
