import torch
import numpy as np
import cv2

class AGSoft_Mask_Fix:
    """
    Расширенная обработка маски:
    - эрозия/дилатация,
    - заполнение дыр,
    - удаление шума,
    - сглаживание,
    - размытие,
    - расширение маски (expand) с опцией сглаживания углов (tapered_corners),
    - бинаризация по порогу (threshold).

    Поддержка CPU/CUDA.
    Превью выводится через выход IMAGE.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "erode_dilate": ("INT", {"default": 0, "min": -256, "max": 256, "step": 1}),
                "fill_holes": ("BOOLEAN", {"default": False}),
                "remove_isolated_pixels": ("INT", {"default": 0, "min": 0, "max": 32, "step": 1}),
                "smooth": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1}),
                "blur": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 256.0, "step": 0.5}),
                "expand": ("INT", {"default": 0, "min": -256, "max": 256, "step": 1}),
                "tapered_corners": ("BOOLEAN", {"default": True}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "device": (["cpu", "cuda"], {"default": "cpu"}),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask_out", "mask_preview")
    FUNCTION = "fix_mask"
    CATEGORY = "AGSoft/Mask"

    def fix_mask(self, mask, erode_dilate, fill_holes, remove_isolated_pixels, smooth, blur, expand, tapered_corners, threshold, device):
        try:
            target_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
            mask = mask.to("cpu")  # OpenCV only on CPU
            mask_np = mask.numpy()

            if mask_np.ndim == 2:
                mask_np = mask_np[None, ...]

            batch_size = mask_np.shape[0]
            processed_masks = []

            for i in range(batch_size):
                m = mask_np[i]
                m_uint8 = (m * 255).astype(np.uint8)

                # --- Эрозия / Дилатация ---
                if erode_dilate != 0:
                    kernel_size = max(1, abs(erode_dilate) // 8 + 1)
                    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
                    if erode_dilate > 0:
                        m_uint8 = cv2.dilate(m_uint8, kernel, iterations=1)
                    else:
                        m_uint8 = cv2.erode(m_uint8, kernel, iterations=1)

                # --- Заполнение дыр ---
                if fill_holes:
                    h, w = m_uint8.shape
                    mask_flood = np.zeros((h + 2, w + 2), np.uint8)
                    im_flood = m_uint8.copy()
                    cv2.floodFill(im_flood, mask_flood, (0, 0), 255)
                    im_flood_inv = cv2.bitwise_not(im_flood)
                    m_uint8 = m_uint8 | im_flood_inv

                # --- Удаление изолированных пикселей ---
                if remove_isolated_pixels > 0:
                    k = min(2 * remove_isolated_pixels + 1, 15)
                    kernel = np.ones((k, k), np.uint8)
                    m_uint8 = cv2.morphologyEx(m_uint8, cv2.MORPH_OPEN, kernel)

                # --- Сглаживание ---
                smooth_iter = min(smooth // 16, 8)
                for _ in range(smooth_iter):
                    m_uint8 = cv2.GaussianBlur(m_uint8, (3, 3), 0)
                    _, m_uint8 = cv2.threshold(m_uint8, 127, 255, cv2.THRESH_BINARY)

                # --- Размытие ---
                if blur > 0:
                    m_uint8 = cv2.GaussianBlur(m_uint8, (0, 0), blur)

                # --- Expand (GrowMask logic) ---
                if expand != 0:
                    if tapered_corners:
                        sigma = max(0.5, abs(expand) * 0.1)
                        m_uint8 = cv2.GaussianBlur(m_uint8, (0, 0), sigma)
                        _, m_uint8 = cv2.threshold(m_uint8, 127, 255, cv2.THRESH_BINARY)
                    else:
                        kernel_size = max(1, abs(expand) // 4 + 1)
                        kernel = np.ones((kernel_size, kernel_size), np.uint8)
                        if expand > 0:
                            m_uint8 = cv2.dilate(m_uint8, kernel, iterations=1)
                        else:
                            m_uint8 = cv2.erode(m_uint8, kernel, iterations=1)

                # --- Thresholding (бинаризация) ---
                m_float = m_uint8.astype(np.float32) / 255.0
                m_binary = (m_float >= threshold).astype(np.float32)  

                processed_masks.append(m_binary)

            result_mask = torch.from_numpy(np.stack(processed_masks, axis=0)).to(target_device)
            preview = result_mask.unsqueeze(-1).repeat(1, 1, 1, 3)

            return (result_mask, preview)

        except Exception as e:
            raise RuntimeError(f"AGSoft_Mask_Fix error: {e}")


# --- РЕГИСТРАЦИЯ ---
NODE_CLASS_MAPPINGS = {
    "AGSoft_Mask_Fix": AGSoft_Mask_Fix
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoft_Mask_Fix": "AGSoft Mask Fix"
}