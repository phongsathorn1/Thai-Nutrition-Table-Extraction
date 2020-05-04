# Thai Nutrition Table Extraction
[Thai-Nutrition-Table-Extraction on GitHub](https://github.com/Phongsathron/Thai-Nutrition-Table-Extraction)

This project is created to extract data form Nutrition Table to help user collect data about nutrition to use with Health care application, Fitness app, etc.

**NOTICE** Thank to [text-detection-ctpn](https://github.com/eragonruan/text-detection-ctpn) and [Tesseract](https://github.com/tesseract-ocr/tesseract), we using their source code for detect and recognize text in nutrition table.

---
## Requirement
- Computer running **Linux** or **MacOS**
- **Python 3.7.1** or later
- **Pip 20.1** or later

## Setup
1. Install Python libraries.

```bash
pip install -r requirements.txt
```

2. Check directory `text_detection/checkpoints_mlt`. If directory not exists, download the file from [google drive](https://drive.google.com/file/d/1HcZuB_MHqsKhKEKpfF1pEU85CYy4OlWO/view?usp=sharing) or [baidu yun](https://pan.baidu.com/s/1BNHt_9fiqRPGmEXPaxaFXw). Then extract file and put `checkpoints_mlt/` in `text-detection/`.

3. Setup `nms` and `bbox`. Because of the libraries are written in Cython, hence you have to build the library by using follow command.

```bash
cd text_detection/utils/bbox
chmod +x make.sh
./make.sh
```

4. [Install Tesseract](https://tesseract-ocr.github.io/tessdoc/Home.html) by following [this document](https://tesseract-ocr.github.io/tessdoc/Home.html).

5. Install Tesseract pretrained to supporting Thai language by going to [this page](https://github.com/tesseract-ocr/tessdata_best) and download `tha.traineddata`. Then set the `TESSDATA_PREFIX` environment variable and put file in `ESSDATA_PREFIX/tessdata/tha.traineddata`.

## Dataset
- The Thai Nutrition Table images are in `images/` directory.

## Demo
- Run `main.py` to see result.
```bash
python main.py
```
