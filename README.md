## Installation

check https://github.com/danielgatis/rembg?tab=readme-ov-file
Go to onnxruntime.ai and check the installation matrix.
https://onnxruntime.ai/getting-started

If you see this error - `WARNING: There was an error checking the latest version of pip`, upgrade the PIP version using this command:
WARNING: There was an error checking the latest version of pip


If you see this error on runtime - `ImportError: DLL load failed while importing onnxruntime_pybind11_state: A dynamic link library (DLL) initialization routine failed.`, 
Install Microsoft Visual C++ Redistributable (x64)
Download & install this first:
https://aka.ms/vs/17/release/vc_redist.x64.exe - This is required for onnxruntime on Windows.


## SAM_MODEL

Download the SAM model and store it in the **sam_models** folder.
https://huggingface.co/HCMUE-Research/SAM-vit-h/blob/main/sam_vit_h_4b8939.pth
or
https://www.kaggle.com/datasets/simayyamuruysal/sam-vit-h-4b8939-pth