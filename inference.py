from pathlib import Path

from glob import glob
import SimpleITK
import os
from konfai.utils.dataset import Dataset

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")

def run():
    dataset = Dataset("./Dataset/", "mha")
    dataset.write("CT", "Case", load_image_file_as_image(location=INPUT_PATH / "images/thoracic-abdominal-ct"))

    os.system("konfai PREDICTION -y --gpu 0 --num_workers 0 --config ./resources/Prediction_TS.yml --MODEL ./resources/Model/M291.pt")
    os.system("konfai PREDICTION -y --config ./resources/Prediction.yml --MODEL ./resources/Model/FT_0.pt:./resources/Model/FT_1.pt:./resources/Model/FT_2.pt:./resources/Model/FT_3.pt:./resources/Model/FT_4.pt --gpu 0")

    write_image_as_image_file(OUTPUT_PATH / "images/pdac-segmentation", SimpleITK.ReadImage("./Predictions/Curvas_1/Dataset/Case/Seg.mha"))
    write_image_as_image_file(OUTPUT_PATH / "images/pdac-confidence", SimpleITK.ReadImage("./Predictions/Curvas_1/Dataset/Case/Prob.mha"))

    return 0

def load_image_file_as_image(location):
    input_files = (
        glob(str(location / "*.tif"))
        + glob(str(location / "*.tiff"))
        + glob(str(location / "*.mha"))
    )
    return SimpleITK.ReadImage(input_files[0])

def write_image_as_image_file(location, image):
    location.mkdir(parents=True, exist_ok=True)
    SimpleITK.WriteImage(
        image,
        location / "output.mha",
        useCompression=True,
    )

if __name__ == "__main__":
    run()
