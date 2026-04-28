import cv2
import numpy as np
import glob
import os

# 1. 체스보드 설정
# 인쇄한 체스보드의 '내부 교차점' 개수를 입력합니다. (예: 가로 9개, 세로 6개)
CHECKERBOARD = (9, 6)
# 서브픽셀 단위로 코너를 정밀하게 찾기 위한 종료 조건 설정
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 세계 좌표계에서의 3D 점들을 저장할 리스트 (예: (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0))
objpoints = []
# 이미지 평면에서의 2D 점들을 저장할 리스트
imgpoints = [] 

# 체스보드의 3D 좌표 생성
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# 2. 촬영한 이미지 불러오기
# 'calibration_images' 폴더 안에 촬영한 jpg 사진들을 넣어두세요.
images_path = 'calibration_images/*.jpg'
images = glob.glob(images_path)

if len(images) == 0:
    print("오류: 캘리브레이션용 이미지를 찾을 수 없습니다. 경로를 확인해주세요.")
    exit()

print(f"총 {len(images)}장의 이미지를 분석합니다...")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 체스보드 코너 찾기
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, 
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                             cv2.CALIB_CB_FAST_CHECK + 
                                             cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    # 코너를 찾았다면
    if ret == True:
        objpoints.append(objp)
        # 서브픽셀 수준으로 코너 위치를 정밀하게 조정
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        
        # 코너를 이미지에 그려서 화면에 보여줌 (확인용)
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Chessboard Detection', img)
        cv2.waitKey(100) # 0.1초 대기

cv2.destroyAllWindows()

# 3. 카메라 캘리브레이션 수행
print("캘리브레이션 연산 중... 잠시만 기다려주세요.")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 4. 결과 출력 및 저장
print("\n=== 캘리브레이션 완료 ===")
print("\n[카메라 매트릭스 (mtx)]")
print(mtx)
print("\n[왜곡 계수 (dist)]")
print(dist)

# numpy 배열 형태로 저장 (나중에 실시간 스트리밍 코드에서 불러오기 위함)
np.savez('camera_calib_data.npz', mtx=mtx, dist=dist)
print("\n'camera_calib_data.npz' 파일로 매트릭스와 왜곡 계수가 저장되었습니다.")

# 5. 왜곡 보정 테스트 (첫 번째 이미지를 펴서 보여줌)
if len(images) > 0:
    test_img = cv2.imread(images[0])
    h,  w = test_img.shape[:2]
    # 최적의 카메라 매트릭스 도출 (크롭 방지)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    
    # 왜곡 펴기
    undistorted_img = cv2.undistort(test_img, mtx, dist, None, newcameramtx)
    
    # 결과 비교
    cv2.imshow('Original (Distorted)', test_img)
    cv2.imshow('Undistorted', undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()