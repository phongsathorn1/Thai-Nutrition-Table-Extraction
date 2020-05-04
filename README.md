# Thai Nutrition Table Extraction
This project is created to extract data form Nutrition Table to help user collect data about nutrition to use with Health care application, Fitness app, etc.

**NOTIC** Thank to [text-detection-ctpn]() and [Tesseract](), we using their source code for detect and recognize text in nutrition table

---
## Requirement
- Computer running MacOS or Linux
- Python 3.7.1 or later
- Pip 20.1 or later

## Setup
1. Install Python libraries

```bash
    pip install -r requirements.txt
```

2. Check directory `text_detection/checkpoints_mlt`. If directory not exists, download the file from [googl drive](https://drive.google.com/file/d/1HcZuB_MHqsKhKEKpfF1pEU85CYy4OlWO/view?usp=sharing) or [baidu yun](https://pan.baidu.com/s/1BNHt_9fiqRPGmEXPaxaFXw). Then extract file and put `checkpoints_mlt/` in `text-detection/`

3. Setup `nms` and `bbox`. Because of the libraries are written in Cython, hence you have to build the library by using follow command.

```bash
cd text_detection/utils/bbox
chmod +x make.sh
./make.sh
```

## Demo
- Run `main.py` to see result
```bash
python main.py
```
