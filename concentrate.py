import cv2
import tensorflow.keras
import numpy as np
import winsound as ws
## 이미지 전처리
def preprocessing(frame):
    # 사이즈 조정
    size = (28, 28)
    frame_resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    
    # 이미지 정규화
    frame_normalized = (frame_resized.astype(np.float32) / 127.0) - 1
    
    # 이미지 차원 재조정 - 예측을 위해 reshape 해줍니다.
    frame_reshaped = frame_normalized.reshape((1, 28, 28, 3))
    
    return frame_reshaped

beep = 1

def beepsound(beep):
    freq = 2000    # range : 37 ~ 32767
    dur = 1000     # ms
    # winsound.Beep(frequency, duration)
   
    if beep==0: ws.Beep(freq, dur) 


str = "공부에 집중하세요!!"

## 학습된 모델 불러오기
model_filename = 'concentrate.h5'
model = tensorflow.keras.models.load_model(model_filename)

# 카메라 캡쳐 객체, 0=내장 카메라
capture = cv2.VideoCapture(0)

# 캡쳐 프레임 사이즈 조절
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 280)

sleep_cnt = 1 
while True:
    ret, frame = capture.read()
    if ret == True: 
        print("read success!")

    # 이미지 뒤집기
    frame_fliped = cv2.flip(frame, 1)
    
    # 이미지 출력
    cv2.imshow("VideoFrame", frame_fliped)
    
    # 1초마다 검사하며, videoframe 창으로 아무 키나 누르게 되면 종료
    if cv2.waitKey(200) > 0: 
        break
    
    # 데이터 전처리
    preprocessed = preprocessing(frame_fliped)

    # 예측
    prediction = model.predict(preprocessed)
   
    
    if prediction[0,0] > prediction[0,1]:
        print('딴짓 상태')
    
        sleep_cnt += 1
        
       
        if sleep_cnt % 30 == 0:
            sleep_cnt = 1
            print('공부에 집중하세요!!')
            beep=0
            beepsound(beep)
            

             
    else:
        print('공부 상태')
        sleep_cnt = 1
      
# 카메라 객체 반환
capture.release() 
# 화면에 나타난 윈도우들을 종료
cv2.destroyAllWindows()