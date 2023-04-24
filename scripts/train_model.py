import glob
from deepkaiCaptchaSolver.core import CaptchaSolver


if __name__ == "__main__":
    # Training image data path
    train_img_path_list = glob.glob("./train/*")
    model_weights_path = "model/weights.h5"

    # Training image data size
    img_width = 200
    img_height = 80

    # Creating an instance that creates a model
    cs = CaptchaSolver(
        train_img_path_list,
        img_width=img_width,
        img_height=img_height,
        n_samples=2000
    )

    # Performing model training
    history, model = cs.train_model("weights.h5", epochs=10, load_weights=True)
