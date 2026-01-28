#!/usr/bin/env python3
"""
Simple test to verify crop size limiting logic without imports.
"""

class CropRegion:
    def __init__(self, x_min: int, y_min: int, x_max: int, y_max: int):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

def test_crop_limiting_logic():
    """Test the crop size limiting logic we implemented."""
    
    print("Testing crop size limiting logic...")
    
    # Simulate a large crop (like from a scene-wide tile)
    original_crop = CropRegion(0, 0, 11310, 17310)
    max_crop_size = 3072
    img_width = 11310
    img_height = 17310
    
    print(f"Original crop: {original_crop.width}x{original_crop.height}")
    
    # Apply the same logic as in our modified compute_tile_crop_for_camera
    clamped = original_crop  # Already within image bounds
    
    # Apply maximum crop size limit to prevent OOM
    if max_crop_size and max_crop_size > 0:
        crop_width = clamped.width
        crop_height = clamped.height
        
        # If crop is too large, center a smaller crop within the projected region
        if crop_width > max_crop_size or crop_height > max_crop_size:
            print(f"Limiting crop size from {crop_width}x{crop_height} to max {max_crop_size}")
            
            # Calculate center of the crop
            center_x = (clamped.x_min + clamped.x_max) // 2
            center_y = (clamped.y_min + clamped.y_max) // 2
            
            # Determine new crop size (maintain aspect ratio if possible)
            if crop_width > crop_height:
                new_width = max_crop_size
                new_height = min(max_crop_size, int(crop_height * max_crop_size / crop_width))
            else:
                new_height = max_crop_size
                new_width = min(max_crop_size, int(crop_width * max_crop_size / crop_height))
            
            print(f"New crop size: {new_width}x{new_height} (centered at {center_x},{center_y})")
            
            # Center the new crop
            new_x_min = center_x - new_width // 2
            new_x_max = new_x_min + new_width
            new_y_min = center_y - new_height // 2
            new_y_max = new_y_min + new_height
            
            print(f"Before bounds check: ({new_x_min},{new_y_min})-({new_x_max},{new_y_max})")
            
            # Ensure we stay within image bounds
            if new_x_min < 0:
                new_x_max -= new_x_min
                new_x_min = 0
            if new_y_min < 0:
                new_y_max -= new_y_min
                new_y_min = 0
            if new_x_max > img_width:
                new_x_min -= (new_x_max - img_width)
                new_x_max = img_width
            if new_y_max > img_height:
                new_y_min -= (new_y_max - img_height)
                new_y_max = img_height
            
            # Final clamp
            new_x_min = max(0, new_x_min)
            new_y_min = max(0, new_y_min)
            new_x_max = min(img_width, new_x_max)
            new_y_max = min(img_height, new_y_max)
            
            final_crop = CropRegion(
                x_min=int(new_x_min),
                y_min=int(new_y_min),
                x_max=int(new_x_max),
                y_max=int(new_y_max)
            )
            
            print(f"Final limited crop: {final_crop.width}x{final_crop.height} "
                  f"({final_crop.x_min},{final_crop.y_min})-({final_crop.x_max},{final_crop.y_max})")
            
            # Verify the result
            if final_crop.width <= max_crop_size and final_crop.height <= max_crop_size:
                print("✅ SUCCESS: Crop size limiting is working correctly!")
                return True
            else:
                print("❌ FAILED: Final crop is still too large")
                return False
        else:
            print("Original crop is already within size limits")
            return True

if __name__ == "__main__":
    success = test_crop_limiting_logic()
    print(f"Test result: {'PASS' if success else 'FAIL'}")