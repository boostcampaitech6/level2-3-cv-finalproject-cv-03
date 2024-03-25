<img src="https://github.com/boostcampaitech6/level2-3-cv-finalproject-cv-03/assets/79782180/3b823d07-4688-46a9-8a54-7025a3538fb1" width=20%>

# WatchDuck

### **프로젝트 주제**

실시간 CCTV 도난 의심 행동 감지 앱

### 선정 배경

24시간 무인으로 운영되는 매장에서의 도난 범죄에 신속히 대응하기 위해 특정 매장에 특화된 도난 의심 행동 감지 모델을 만들고자 함

### 기대 효과

- 실시간으로 도난 의심 알림을 받아 실제 도난이 발생하기 전 조치를 취할 수 있음
- 도난 의심 행동 구간을 따로 확인할 수 있어 모든 CCTV 영상을 볼 필요가 없어 사용자 피로도 감소에도 효과적임

### 1. Dataset

[AI-Hub 실내 사람 이상행동 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71550)

- Format: 영상 이미지
- Labeling: 물건을 가방이나 주머니에 넣는 순간을 Doubt, 그 외 모든 구간은 Normal
- 801개의 데이터 중 무인매장에 적합한 583개의 데이터를 선별 후 사용

### 2. Model

🔧 `Pytorch`
![model](https://github.com/boostcampaitech6/level2-3-cv-finalproject-cv-03/assets/79782180/3c94a7e0-8f08-4f50-a4a0-1cb08227bcc2)

- 구조: CNN+RNN
    - CNN: 클립 영상의 프레임 별 feature 추출
    - RNN: CNN의 output을 입력으로 받아 영상 속 행동 맥락을 파악한 후 결과를 예측
    
    ⇒ **MobileNet_v2 + GRU**
    

### 3. Frontend

🔧 `Figma` `React Native` `Expo`

- Figma: 디자인 툴로 사용, 댓글 기능 활용 1차, 2차 피드백을 거치며 디자인/기능적 합의
- Expo: Android, iOS, Web 모두 같은 코드로 호환 가능한 개발
- 푸시 알림: Polling 방식을 사용하여 새 도난 탐지 기록 발생 시 푸시 알림 발생

**3.1.페이지 목록**

| 페이지명 | 페이지 상세 | 기능 |
| --- | --- | --- |
| 회원가입 |  | 1. 약관 동의<br>2. 이메일 인증<br>3. 비밀번호 입력<br>4. URL 등록<br>5. 매장 이름 등록<br>6. 사용자 이름 등록 |
| 로그인 |  | 1. 로그인 |
| Tab1 도난 탐지 기록 | 도난 탐지 기록 리스트 | 기본 화면<br>1. 검색 기능(날짜, CCTV 이름으로 검색) |
|  | 도난 탐지 기록별 상세 | 리스트의 각 아이템 클릭 시 상세 페이지로 이동<br>1. 피드백 남기기<br>2. 영상 다운로드<br>3. 기록 삭제하기 |
| Tab2 실시간 스트리밍 |  | 최대 16개 CCTV 동시 스트리밍 가능<br>1. 각 CCTV 클릭 시 전체화면 |
| Tab3 설정화면 | CCTV 설정 | 1. CCTV 등록<br>2. CCTV 수정(이름, URL)<br>3. CCTV 삭제 |
|  | 알림/동영상 설정 | 1. 알림 임계값 조회 및 수정<br>2. 저장되는 동영상 시간 조회 및 수정 |
|  | 개인정보 설정 | 1. 비밀번호 수정 (기존 비밀번호 확인 후 새 비밀번호 등록)<br>2. 매장명 수정 |

### 4. Backend

🔧 `FastAPI` `Redis`

**4.1. DB(PostgreSQL) & ORM(SQLAlchemy**)
- DB Schema

![DB](https://github.com/boostcampaitech6/level2-3-cv-finalproject-cv-03/assets/79782180/371b7b00-fa28-48b8-b26e-d58738a6b486)

**4.2. API**
- a.b.c 형식으로 Versioning하여 API 명세서 관리
  - a : 배포
  - b : 백-프론트 합의
  - c : 백 또는 프론트에서 개별 업데이트 
- member, cctv, streaming, settings의 네 개 분류로, 총 20개의 API 이용


### 5. Architecture

5.1. **Streaming Architecture**


5.2. **System Architecture**


# 팀 소개



<div align="center">
<table>
  <tr>
    <td align="center"><a href="https://github.com/jinjero"><img src="https://avatars.githubusercontent.com/u/146058962?v=4" width="100px;" alt=""/><br /><sub><b>박진영</b></sub><br />
    </td>
    <td align="center"><a href="https://github.com/rudeuns"><img src="https://avatars.githubusercontent.com/u/151593264?v=4" width="100px;" alt=""/><br /><sub><b>선경은</b></sub><br />
    </td>
    <td align="center"><a href="https://github.com/hyunseo-k"><img src="https://avatars.githubusercontent.com/u/79782180?v=4" width="100px;" alt=""/><br /><sub><b>강현서</b></sub><br />
    </td>
    <td align="center"><a href="https://github.com/Jungtaxi"><img src="https://avatars.githubusercontent.com/u/18082001?v=4" width="100px;" alt=""/><br /><sub><b>김정택</b></sub><br />
    </td>
    <td align="center"><a href="https://github.com/rsl82"><img src="https://avatars.githubusercontent.com/u/90877240?v=4" width="100px;" alt=""/><br /><sub><b>이선우</b></sub><br />
    </td>
    <td align="center"><a href="https://github.com/ChoeHyeonWoo"><img src="https://avatars.githubusercontent.com/u/78468396?v=4" width="100px;" alt=""/><br /><sub><b>최현우</b></sub><br />
    </td>
  </tr>
  <tr>
    <td><b>Model</b> </td>
    <td><b>Model</b> </td>
    <td><b>Frontend</b> </td>
    <td><b>Frontend</b> </td>
    <td><b>Backend</b> </td>
    <td><b>Backend</b> </td>
  </tr>
</table>
</div>
