import os
import subprocess
from pathlib import Path


def download_models(output_folder="./models"):
    assert "INTEL_OPENVINO_DIR" in os.environ, "OpenVINO workspace not initialized"
    OPENVINO_dir = Path(os.environ["INTEL_OPENVINO_DIR"])
    downloader_script = OPENVINO_dir.joinpath("deployment_tools/tools/model_downloader/downloader.py")
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)
    with open("./models.txt", "r") as f:
        for model in f.readlines():
            try:
                cmd = f'''python "{downloader_script}"
                --name {model.strip()}
                -o {output_folder}
                '''
                cmd = " ".join([line.strip() for line in cmd.splitlines()])
                print(subprocess.check_output(cmd, shell=True).decode())
            except Exception as ex:
                print("Error downloading the model {} : {}".format(model, ex))


if __name__ == '__main__':
    download_models()