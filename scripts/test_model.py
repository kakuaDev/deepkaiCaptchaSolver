import numpy as np
from deepkaiCaptchaSolver.core import ApplyModel, LABEL_ENCODER
import cv2
import glob

if __name__ == "__main__":
    model_weights_path = "weights.h5"
    target_images_path = glob.glob("train/*.png")

    # image data size
    img_width = 200
    img_height = 80

    AM = ApplyModel(model_weights_path, img_width, img_height)

    for target_image_path in target_images_path:
        image = cv2.imread(target_image_path)
        target_img: np.array = cv2.resize(
            image, (img_width, img_height), interpolation=cv2.INTER_AREA)

        print(AM.predict(target_image_path))

        cv2.imshow("Target", target_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
