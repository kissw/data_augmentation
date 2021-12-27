## Data Augmentation

머신러닝 학습시 Overffiting 방지를 위해 Data Augmentation 하는 프로그램  </br></br>

## 사용법
### 파라미터
```yaml
origin_img_path: ./img # 원본 이미지 경로
origin_txt_path: ./img # 원본 이미지 라벨파일 경로. Yolo의 경우 보통 이미지와 라벨파일을 한 폴더에 넣어놓기때문에 보통 origin_img_path 와 같게하면됨
save_img_path: ./img # 어그멘테이션 이미지 저장 경로
file_ext: JPG #이미지 확장자
# 저장시 이름은 "원본이미지이름_(숫자)" 형태로 저장됨
```
</br>

Imagaug library
```python
https://github.com/aleju/imgaug # 참고
```
```python
def augmentations(self, images):
    
    seq1 = iaa.Sequential([
        iaa.AverageBlur(k=(3,3)) 
    ])
    seq2 = iaa.ChannelShuffle(p=1.0)
    seq3 = iaa.Dropout((0.05, 0.1), per_channel=0.5)
    seq4 = iaa.Sequential([
        iaa.Add((-10,10)),
        iaa.Multiply((0.7, 1.3))
    ])
    ...
```
</br>

### 명령어
conda 환경설치 방법은 상위 폴더를 참조
```
(base) $ conda activate watt

(3w) $ python src/data_augmentation_bbox.py
```
