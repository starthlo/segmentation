"""
Multi-strategy detection with automatic fallback for robust slab detection
"""
import cv2
import numpy as np
from rembg import remove


class MultiStrategyDetector:
    """
    Tries multiple detection strategies and automatically selects the best one
    based on quality metrics.
    """
    
    def __init__(self, adaptive_detector=None):
        self.adaptive_detector = adaptive_detector
        self.debug_mode = True  # Save intermediate results for debugging
        
    def apply_u2net_with_strategies(self, img_rgb, analysis, params):
        """
        Apply U2NET with multiple strategies and pick the best result.
        Works at full resolution for maximum accuracy.
        Returns: (best_mask, strategy_name, quality_score)
        """
        h, w = img_rgb.shape[:2]
        print(f"  Multi-strategy detection at {w}x{h} resolution")
        
        strategies = []
        
        # Strategy 1: With preprocessing (adaptive)
        strategies.append(("Adaptive+Preprocess", lambda: self._strategy_with_preprocess(img_rgb, params)))
        
        # Strategy 2: Without preprocessing
        strategies.append(("NoPreprocess", lambda: self._strategy_no_preprocess(img_rgb, params)))
        
        # Strategy 3: Conservative (higher threshold)
        strategies.append(("Conservative", lambda: self._strategy_conservative(img_rgb, params)))
        
        # Strategy 4: Aggressive (lower threshold) - only for light stones
        if analysis.get('stone_type') in ['light_uniform', 'light_veined']:
            strategies.append(("Aggressive", lambda: self._strategy_aggressive(img_rgb, params)))
        
        # Strategy 5: Multi-threshold fusion
        strategies.append(("Multi-Threshold", lambda: self._strategy_multi_threshold(img_rgb, params)))
        
        # Strategy 6: Edge-guided (uses edge detection)
        strategies.append(("Edge-Guided", lambda: self._strategy_edge_guided(img_rgb, params)))
        
        best_mask = None
        best_quality = 0.0
        best_strategy = None
        results = []
        
        print(f"\n  === TESTING {len(strategies)} STRATEGIES (Full Resolution) ===")
        
        for i, (strategy_name, strategy_func) in enumerate(strategies, 1):
            try:
                print(f"    [{i}/{len(strategies)}] Testing {strategy_name}...", end=' ', flush=True)
                
                mask = strategy_func()
                
                # Evaluate quality
                if self.adaptive_detector:
                    quality_metrics = self.adaptive_detector.evaluate_mask_quality(mask, img_rgb)
                    quality = quality_metrics['overall']
                else:
                    quality = self._simple_quality_score(mask)
                    quality_metrics = {'overall': quality, 'raw_coverage': np.sum(mask > 0) / mask.size}
                
                coverage = quality_metrics['raw_coverage']
                
                print(f"Q={quality:.3f}, Cov={coverage:.1%}")
                
                results.append({
                    'name': strategy_name,
                    'mask': mask,
                    'quality': quality,
                    'coverage': coverage
                })
                
                # Save debug images (downsampled to save space)
                if self.debug_mode:
                    h, w = mask.shape
                    mask_small = cv2.resize(mask, (w//2, h//2), interpolation=cv2.INTER_NEAREST)
                    cv2.imwrite(f'debug_strategy_{strategy_name}.jpg', mask_small)
                
                if quality > best_quality:
                    best_quality = quality
                    best_mask = mask
                    best_strategy = strategy_name
                    
            except Exception as e:
                print(f"    {strategy_name:20s}: FAILED - {e}")
                continue
        
        # Fallback: if all strategies produce poor quality, try ensemble
        if best_quality < 0.3:
            print(f"    All strategies poor quality, trying ensemble...")
            ensemble_mask = self._ensemble_strategies(results)
            if self.adaptive_detector:
                ensemble_quality = self.adaptive_detector.evaluate_mask_quality(ensemble_mask, img_rgb_small)['overall']
            else:
                ensemble_quality = self._simple_quality_score(ensemble_mask)
            
            if ensemble_quality > best_quality:
                best_mask = ensemble_mask
                best_quality = ensemble_quality
                best_strategy = "Ensemble"
                print(f"    {'Ensemble':20s}: quality={ensemble_quality:.3f} [SELECTED]")
        
        print(f"  ✓ SELECTED: {best_strategy} (quality={best_quality:.3f})")
        return best_mask, best_strategy, best_quality
    
    def _strategy_with_preprocess(self, img_rgb, params):
        """Strategy: With adaptive preprocessing"""
        if self.adaptive_detector:
            img_preprocessed = self.adaptive_detector.preprocess_image_for_rembg(img_rgb, params)
        else:
            img_preprocessed = img_rgb
        
        processed = remove(img_preprocessed)
        processed = np.array(processed)
        
        if processed.shape[-1] == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_RGB2RGBA)
        
        alpha = processed[:, :, 3] / 255.0
        mask = (alpha > params['alpha_threshold']).astype(np.uint8) * 255
        
        if self.adaptive_detector:
            mask = self.adaptive_detector.postprocess_mask(mask, params)
        
        return mask
    
    def _strategy_no_preprocess(self, img_rgb, params):
        """Strategy: No preprocessing"""
        processed = remove(img_rgb)
        processed = np.array(processed)
        
        if processed.shape[-1] == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_RGB2RGBA)
        
        alpha = processed[:, :, 3] / 255.0
        mask = (alpha > params['alpha_threshold']).astype(np.uint8) * 255
        
        if self.adaptive_detector:
            mask = self.adaptive_detector.postprocess_mask(mask, params)
        
        return mask
    
    def _strategy_conservative(self, img_rgb, params):
        """Strategy: Conservative (higher threshold, less aggressive)"""
        if self.adaptive_detector:
            img_preprocessed = self.adaptive_detector.preprocess_image_for_rembg(img_rgb, params)
        else:
            img_preprocessed = img_rgb
        
        processed = remove(img_preprocessed)
        processed = np.array(processed)
        
        if processed.shape[-1] == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_RGB2RGBA)
        
        alpha = processed[:, :, 3] / 255.0
        conservative_threshold = min(params['alpha_threshold'] * 1.5, 0.25)
        mask = (alpha > conservative_threshold).astype(np.uint8) * 255
        
        # Less aggressive postprocessing
        conservative_params = params.copy()
        conservative_params['morph_kernel_size'] = max(5, params['morph_kernel_size'] - 2)
        conservative_params['fill_holes_aggressively'] = False
        
        if self.adaptive_detector:
            mask = self.adaptive_detector.postprocess_mask(mask, conservative_params)
        
        return mask
    
    def _strategy_aggressive(self, img_rgb, params):
        """Strategy: Aggressive (lower threshold, more aggressive)"""
        if self.adaptive_detector:
            img_preprocessed = self.adaptive_detector.preprocess_image_for_rembg(img_rgb, params)
        else:
            img_preprocessed = img_rgb
        
        processed = remove(img_preprocessed)
        processed = np.array(processed)
        
        if processed.shape[-1] == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_RGB2RGBA)
        
        alpha = processed[:, :, 3] / 255.0
        aggressive_threshold = max(params['alpha_threshold'] * 0.6, 0.02)
        mask = (alpha > aggressive_threshold).astype(np.uint8) * 255
        
        # More aggressive postprocessing
        aggressive_params = params.copy()
        aggressive_params['morph_kernel_size'] = params['morph_kernel_size'] + 4
        aggressive_params['fill_holes_aggressively'] = True
        
        if self.adaptive_detector:
            mask = self.adaptive_detector.postprocess_mask(mask, aggressive_params)
        
        return mask
    
    def _strategy_multi_threshold(self, img_rgb, params):
        """Strategy: Multi-threshold fusion"""
        if self.adaptive_detector:
            img_preprocessed = self.adaptive_detector.preprocess_image_for_rembg(img_rgb, params)
        else:
            img_preprocessed = img_rgb
        
        processed = remove(img_preprocessed)
        processed = np.array(processed)
        
        if processed.shape[-1] == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_RGB2RGBA)
        
        alpha = processed[:, :, 3] / 255.0
        base_threshold = params['alpha_threshold']
        
        # Try multiple thresholds
        mask_low = (alpha > base_threshold * 0.7).astype(np.float32)
        mask_mid = (alpha > base_threshold).astype(np.float32)
        mask_high = (alpha > base_threshold * 1.3).astype(np.float32)
        
        # Weighted fusion - emphasize mid threshold
        fused = (mask_low * 0.2 + mask_mid * 0.5 + mask_high * 0.3)
        mask = (fused > 0.4).astype(np.uint8) * 255
        
        if self.adaptive_detector:
            mask = self.adaptive_detector.postprocess_mask(mask, params)
        
        return mask
    
    def _strategy_edge_guided(self, img_rgb, params):
        """Strategy: Edge-guided detection"""
        if self.adaptive_detector:
            img_preprocessed = self.adaptive_detector.preprocess_image_for_rembg(img_rgb, params)
        else:
            img_preprocessed = img_rgb
        
        processed = remove(img_preprocessed)
        processed = np.array(processed)
        
        if processed.shape[-1] == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_RGB2RGBA)
        
        alpha = processed[:, :, 3] / 255.0
        mask = (alpha > params['alpha_threshold']).astype(np.uint8) * 255
        
        # Use edges to refine
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Remove mask regions that don't align with edges
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Keep largest contour
            largest = max(contours, key=cv2.contourArea)
            refined_mask = np.zeros_like(mask)
            cv2.drawContours(refined_mask, [largest], -1, 255, -1)
            
            # Check edge alignment
            boundary = np.zeros_like(mask)
            cv2.drawContours(boundary, [largest], -1, 255, 2)
            
            edge_overlap = cv2.bitwise_and(boundary, edges_dilated)
            if np.sum(edge_overlap) < np.sum(boundary) * 0.1:
                # Poor edge alignment, erode conservatively
                refined_mask = cv2.erode(refined_mask, kernel, iterations=2)
            
            mask = refined_mask
        
        if self.adaptive_detector:
            mask = self.adaptive_detector.postprocess_mask(mask, params)
        
        return mask
    
    def _ensemble_strategies(self, results):
        """Combine multiple strategies using voting"""
        if not results:
            return np.zeros((100, 100), dtype=np.uint8)
        
        # Get top 3 strategies
        results_sorted = sorted(results, key=lambda x: x['quality'], reverse=True)
        top_strategies = results_sorted[:min(3, len(results_sorted))]
        
        if not top_strategies:
            return results[0]['mask']
        
        # Voting ensemble
        h, w = top_strategies[0]['mask'].shape
        vote_map = np.zeros((h, w), dtype=np.float32)
        
        for result in top_strategies:
            weight = result['quality']  # Weight by quality
            vote_map += (result['mask'] > 0).astype(np.float32) * weight
        
        # Normalize and threshold
        if len(top_strategies) > 0:
            vote_map /= sum([r['quality'] for r in top_strategies])
        
        ensemble_mask = (vote_map > 0.5).astype(np.uint8) * 255
        
        return ensemble_mask
    
    def _simple_quality_score(self, mask):
        """Simple quality score without adaptive detector"""
        h, w = mask.shape
        coverage = np.sum(mask > 0) / (h * w)
        
        # Penalize too small or too large
        coverage_score = 1.0
        if coverage < 0.1:
            coverage_score = coverage / 0.1
        elif coverage > 0.85:
            coverage_score = (1.0 - coverage) / 0.15
        
        # Check contour quality
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        largest_area = max([cv2.contourArea(c) for c in contours])
        total_area = np.sum(mask > 0)
        contour_score = largest_area / max(total_area, 1)
        
        return (coverage_score * 0.5 + contour_score * 0.5)
