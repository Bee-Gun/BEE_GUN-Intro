# 말벌인식 프로젝트

---

- 목차
    1. 왜 꿀벌 프로젝트인가?
    2. 데이터 셋 구축
        
        1) 동영상 추출
        
        2) 프레임 추출
        
        3) 라벨링 작업 - 가이드라인 작성
        
    3. What is **Yolo**
    4. YoloV5 code review 
        
        1) inference Code
        
        2) train Code
        
    

알 수 없는 원인으로 꿀벌 개체가 줄어드는 현상이 전 세계적으로 발생하고 있다. 우리나라도 예외가 아니며 미세먼지, 전자기파, 기생충 등 수많은 원인이 거론되고 있는 가운데 `검은등 말벌`이 눈에 띄게 꿀벌 개체를 줄이는데 일조하고 있다.

한국에도 2011년 중국에서 들어온 `검은등 말벌`이 토종꿀벌 생태계를 위협하고 있고, 양봉 농가의 피해(추산 1700억)가 증가하고 있는 실정이다.

### □ 꿀벌과 꿀벌을 위협하는 말벌들

**꿀벌**
**검은등 말벌**
**장수말벌**

---

 말벌이 꿀벌을 사냥하는 방법과 이유

[김병만×배정남×박군, 등검은말벌 영상에 느끼는 인간의 책임감!ㅣ공생의 법칙(symbiosis) ㅣSBS ENTER..mp4](%E1%84%86%E1%85%A1%E1%86%AF%E1%84%87%E1%85%A5%E1%86%AF%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%89%200caf5/%EA%B9%80%EB%B3%91%EB%A7%8C%EB%B0%B0%EC%A0%95%EB%82%A8%EB%B0%95%EA%B5%B0_%EB%93%B1%EA%B2%80%EC%9D%80%EB%A7%90%EB%B2%8C_%EC%98%81%EC%83%81%EC%97%90_%EB%8A%90%EB%81%BC%EB%8A%94_%EC%9D%B8%EA%B0%84%EC%9D%98_%EC%B1%85%EC%9E%84%EA%B0%90!%E3%85%A3%EA%B3%B5%EC%83%9D%EC%9D%98_%EB%B2%95%EC%B9%99(symbiosis)_%E3%85%A3SBS_ENTER..mp4)

출처 : SBS 공생의법칙

현재 말벌에 대한 대책은?

**양봉농가의 대책**을 살펴보면 아날로그적이다.

말벌 트랩설치 또는 벌통 앞을 지키는 수준이고, 사실상 속수무책이다.

---

이런 상황에서 인공지능 기술로 Detecting 하고, 꿀벌 속에서 말벌만 잡아낼 수는 없을까??

![KakaoTalk_Photo_2022-02-14-14-51-24.png](%E1%84%86%E1%85%A1%E1%86%AF%E1%84%87%E1%85%A5%E1%86%AF%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%89%200caf5/KakaoTalk_Photo_2022-02-14-14-51-24.png)

![2AE3803C-0A41-4B46-946D-4F78BE57A7AD_1_201_a.jpeg](%E1%84%86%E1%85%A1%E1%86%AF%E1%84%87%E1%85%A5%E1%86%AF%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%89%200caf5/2AE3803C-0A41-4B46-946D-4F78BE57A7AD_1_201_a.jpeg)

이런 문제 상황에서 사람이 직접 벌통을 지키는것 보다. 인공지능 기술을 사용하여 작업을 하는것이 훨씬 사람들의 생활을 풍요롭게할것이다.

---

우선, 일반적인 인공지능서비스가 만들어지는 과정을 살펴보면,

<aside>
💡 Data 수집 → 전처리 → Modeling → 학습 → Service(Wed Service or Embeding)

</aside>

Data 수집은 첫 번째 해야 할 작업이다. 하지만 개인이 데이터를 구하기란 쉽지 않다.  그래서 인공지능 모델링 공부를 할 때는 Kaggle Data, AI hub 등 주어진 대량의 데이터를 통해 학습할 수 있다. 그러나 **현실적인 문제를 해결하기 위해서는 그에 적합한 데이터를 구할 필요가 있고** 직접 데이터를 가공 및 처리하는 능력이 필요하다.

### □ 작업과정

<aside>
💡 동영상 및 이미지 수집 → 프레임 추출 → 라벨링 작업 → Modeling 학습

</aside>

이렇게 학습된 작업물은 어떻게 사용 가능할까요? 

AI Modeling을 통한 Detection 후, Web Service 및 실물 임베딩을 통한 물리적 구현정도로 생각해 볼 수 있다.  

### □ 동영상 추출

---

**유튜브 영상를 추출하는 두 가지 방법**

- 웹사이트를 통한 방법
    
    [https://www.youtube.com/](https://www.youtube.com/) ← 유튜브 주소창
    
    [https://www.ssyoutube.com/](https://www.ssyoutube.com/) ← ss를 추가해준다.
    
    그리고 다운로드를 하는 방법
    
- 파이썬 코드를 통한 방법
    
    ```python
    #Pytube 설치
    pip install --upgrade "git+https://github.com/nficano/pytube.git"
    ```
    
    ```python
    import os 
    import pytube # pip install pytube 
    from pytube.cli import on_progress 
    url = "https://www.youtube.com/watch?v=xP9hvIYHMC8" 
    yt = pytube.YouTube(url, on_progress_callback=on_progress) 
    print(yt.streams)
    ```
    
    ```python
    urls = ['유튜브 URL(1)','유튜브 URL(2)',]
    # 해당 유튜브 영상 URL 주소를 LIST로 받는다.
    ```
    
    ```python
    save_dir = "/content/drive/MyDrive/Data_2nd/VDO" # 저장경로
    for url in url_:
      print(url)
      yt = pytube.YouTube(url, on_progress_callback=on_progress) 
      print(yt.streams)
      yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc().first().download(save_dir)
    ```
    
    파이썬 코드를 이용하면 URL주소를 대량으로 모아서 한번에 다운받을 수 있다.
    
    참고자료 :[Pytube](https://pytube.io/en/latest/) 
    

---

저작권 문제 :  동영상 저작자에게 미리 양해를 구하고 사용한 데이터 입니다.

### □ 프레임 추출

(해커톤 중 편의대로 코드 이름 및 변수명 제작 - 추후 수정필요)

```python
import cv2
import os

vido_folder = '/content/drive/MyDrive/Data/VDO'
img_folder = '/content/drive/MyDrive/Data/img'
Height_resolution = 720 (해상도)
```

폴더경로 설정하는 코드 입니다. 

비디오 폴더 설정 / 이미지 폴더 설정 / 해상도 설정

```python
# Load all videos
v_list = [v for v in os.listdir(v_folder) if v.endswith('mp4')]
for idx, v in enumerate(v_list):
    v_list[idx] = os.path.join(v_folder, v)
print(len(v_list))
print(v_list)
```

```python
# Get frames
count = 0
for v_path in v_list:
    vidcap = cv2.VideoCapture(v_path)
    if vidcap.isOpened():
        while True:
            ret, image = vidcap.read()
            if ret:
                h, w, c = image.shape
                nw = int(w*resolution/h)
                image = cv2.resize(image, (nw, resolution)) # resize
                if(int(vidcap.get(1)) % 30 == 0):
                    cv2.imwrite("%s/%05d.png" % (i_folder,count), image)
                    count += 1
            else:
                break
        print("Number of width pixels : ", nw)
        print('Saved frame %d.jpg' % (count-1))
        vidcap.release()
```

### □ 라벨링 작업

**라벨링방법에는** 점찍는 포인트, 박스로 대상을 지정하는 **b-box**, 인체 관절 등에 점과 선으로 연결하는 **스켈레톤**, b-box보다 세밀하게 대상을 지정하는 **폴리곤**, 전체를 면으로 분할하는 **세그멘트**, 이미지가 아닌 문서로 하는 **텍스트**, 음성녹음을 적는 **전사** 등이 있습니다.

이 중에서 우리는 **b-box**을 활용합니다.

[https://github.com/tzutalin/labelImg](https://github.com/tzutalin/labelImg)

- 설치하는 방법
    
    **macOS**
    
    ```python
    brew install qt  # Install qt-5.x.x by Homebrew
    brew install libxml2
    
    or using pip
    
    pip3 install pyqt5 lxml # Install qt and lxml by pip
    
    make qt5py3
    python3 labelImg.py
    python3 labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]
    ```
    
    ```python
    pip install labelImg
    labelImg
    labelImg [IMAGE_PATH] [PRE-DEFINED CLASS FILE]
    ```
    
- Class 설정

---

 Yolo format .txt파일 살펴보기

```python
[class, x_center, y_center, width, height]
```

---

□ **가이드라인 작성**

충분한 데이터 셋을 얻기위해서는 많은 시간과 노력이 필요합니다. 

여러명이 일관성있는 라벨링 작업을 하기위해서는  `가이드라인`을 작성해야합니다.

공통된 목표 설정 및 예외적인 상황에서 어떻게 라벨링 할 것인지 작업자들의 합의가 이뤄져야지

추후 안정적인 인공지능모델 학습에 기여할 것 입니다.

- 가이드라인 예시
    
    (미 작성)
    

### □ Yolov5

[YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/)

[https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)


- Wandb
    
    ```python
    !pip install wandb
    ```
    
    ```python
    !wandb login
    ```
    

(1) Inference Code

- Yolov5 코드 다운로드

```python
!git clone https://github.com/ultralytics/yolov5.git
```

- Yolov5 코드 다운 확인

```python
!ls -a
```

- 코드 구동을 위한 환경설정
    
    (1) 경로 이동
    

```python
# 절대 경로
cd '/content/drive/MyDrive/code/yolov5'
# 상대 경로 
cd './yolov5'
```

      (2) 라이브러리 설치

```python
!pip install -r requirements.txt
```

- pre-trained 가중치 다운로드
    - 가중치 공유 주소 (n드라이브, 구글드라이브 등)
        
        [best.pt](%E1%84%86%E1%85%A1%E1%86%AF%E1%84%87%E1%85%A5%E1%86%AF%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%89%200caf5/best.pt)
        
    - 가중치 저장 경로 정해주기
    
- Pre-trained 가중치를 활용하여 말벌 인식 inference 진행

```python
!python detect.py --source '/content/drive/MyDrive/Hackathon Data/Crawling' --weights 'custom_n_train_valid_100_720/exp/weights/best.pt' --imgsz 720
```

--source ‘테스트 이미지 셋 폴더경로’ ,  --weights ‘Pre-trained 가중치경로’

![inference 결과 예시](%E1%84%86%E1%85%A1%E1%86%AF%E1%84%87%E1%85%A5%E1%86%AF%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%89%200caf5/13833.png)

inference 결과 예시

(2) Train Code

- 데이터셋 경로 설정
    - custom/*/images/에 이미지를 넣고, custom/*/labels/에 라벨 데이터 넣음
    - 주의: images 안에 있는 파일과, labels 안에 있는 파일은  대응이 되어야함

![그림1.png](%E1%84%86%E1%85%A1%E1%86%AF%E1%84%87%E1%85%A5%E1%86%AF%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%89%200caf5/%EA%B7%B8%EB%A6%BC1.png)

```python
# parent
# ├── yolov5
# └── datasets
#     └── custom
#					└── train
#							├── images
#							└── labels
#					└── valid
#							├── images
#							└── labels
#					└── test
#							├── images
#							└── labels
```

- data 설정 파일 생성 및 작성
    - custom.yaml 파일을 생성하여, yolov5/data 폴더에 넣어줌

```yaml
train: custom/train/images/  # train images (relative to 'path') 128 images
val: custom/valid/images/ # val images (relative to 'path') 128 images

# Classes
nc: 2  # number of classes
names: ['honey', 'hornet']  # class names list
```

- 제공한 pre-trained 가중치를 기반으로 custom 데이터로 학습 진행

```python
!python train.py --img 720 --batch 16 --epochs 100 --data data/custom__.yaml --cfg models/yolov5s.yaml --weights weights/yolov5s.pt --project bee_n_label2_valid!python train.py --img 720 --batch 16 --epochs 100 --data data/custom__.yaml --cfg models/yolov5s.yaml --weights weights/yolov5s.pt
```

--img 이미지세로필셀수, --batch 배치사이즈, --epochs 에폭수, --data 데이터설정파일경로 --cfg 모델config파일경로, --weights pre-trained가중치경로

□ 프로젝트 목적에 따라 모델 사이즈 결정 

![Untitled](%E1%84%86%E1%85%A1%E1%86%AF%E1%84%87%E1%85%A5%E1%86%AF%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%89%200caf5/Untitled%204.png)

최종 결과

---

[KakaoTalk_Video_2022-02-15-18-52-34.mp4](%E1%84%86%E1%85%A1%E1%86%AF%E1%84%87%E1%85%A5%E1%86%AF%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%89%200caf5/KakaoTalk_Video_2022-02-15-18-52-34.mp4)

[KakaoTalk_Video_2022-02-15-18-48-44.mp4](%E1%84%86%E1%85%A1%E1%86%AF%E1%84%87%E1%85%A5%E1%86%AF%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%89%200caf5/KakaoTalk_Video_2022-02-15-18-48-44.mp4)
