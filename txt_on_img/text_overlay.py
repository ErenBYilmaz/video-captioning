from cv2watermark import (
    merge_image_percentage_height,
    merge_image_percentage_width,
    merge_image_percentage_height_position,
    merge_image_percentage_width_position,
    merge_image,
)

i1 = r"/Users/ronaldseidel/PycharmProjects/venv_waterkant.coding23/data/Frame at 00:00:16.jpg"
i2 = r"/Users/ronaldseidel/PycharmProjects/venv_waterkant.coding23/data/Frame at 00:00:16_text_overlay.png"

merg1 = merge_image(
    back=i1, front=i2, x=10, y=-1, save_path="/Users/ronaldseidel/PycharmProjects/venv_waterkant.coding23/data/Frame at 00:00:16_with-text_overlay2.jpg")