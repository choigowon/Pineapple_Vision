# 

---

# 시각장애인용 온디바이스 비전 AI 웨어러블 보조 장치

이 장치는 시각장애인이 스마트폰을 손으로 조작할 필요 없이(**No-Touch**), 안경처럼 착용만 하면 주변 환경을 AI가 읽어주고 음성으로 안내해 주는 첨단 보조 기기입니다.

---

## 1. 프로젝트 개요 및 3대 핵심 특징

인터넷 연결 없이도 실시간으로 사물을 인식하는 **'온디바이스(On-device) AI'** 기술을 라즈베리 파이 5에 구현하여, 실제 일상에서 착용 가능한 형태로 제작하는 것이 목표입니다.

- **손이 자유로운 인터페이스 (Hands-Free):** 카메라가 사용자의 시선을 대신하고 음성으로만 상호작용 할 수 있습니다.
- **지연 없는 빠른 인식 (Low Latency):** 움직이는 물체를 왜곡 없이 찍는 '글로벌 셔터 카메라'와 최신 '라즈베리 파이 5'의 연산 능력을 결합해 실시간으로 위험 요소를 알려줍니다.
- **안전한 청취 환경 (Bone Conduction):** 귀를 막지 않는 **골전도 이어폰**을 사용합니다. AI의 안내를 들으면서도 자동차 소리나 발소리 같은 주변의 생존 정보를 동시에 들을 수 있어 안전합니다.

---

## 2. 장치의 두뇌와 몸체 (시스템 및 웨어러블 설계)

### ⚙️ 시스템 구조와 데이터 흐름

데이터는 다음의 과정을 거쳐 사용자에게 전달됩니다.

1. **입력:** 카메라가 주변 영상을 초당 수십 장 캡처합니다.
2. **전처리:** 라즈베리 파이가 영상을 분석하기 좋게 다듬습니다.
3. **추론:** AI 모델(ONNX/TensorFlow Lite)이 "저것은 전신주다", "계단이다"라고 판단합니다.
4. **출력:** TTS 엔진이 글자를 목소리로 바꿔 골전도 이어폰으로 들려줍니다.

### 👕 착용 및 하드웨어 구성

- **카메라 장착:** 카메라를 골전도 이어폰 쪽에 고정합니다. 이렇게 하면 사용자가 고개를 돌리는 방향(시선)과 카메라의 방향이 일치하여 정확한 안내가 가능합니다.
- **본체 휴대:** 열이 많이 발생하는 라즈베리 파이와 배터리는 아크릴 케이스에 넣어 안전하게 결합한 뒤, 주머니나 가방에 넣고 케이블로 연결하여 휴대합니다.
- **열 관리 및 보호:** 라즈베리 파이 5의 발열을 식히기 위해 전용 **액티브 쿨러**를 상시 가동합니다. 아크릴 케이스는 외부 충격을 막아주며, 배터리와 본체 사이 간격을 두어 열이 서로 전달되지 않게 설계했습니다.

---

## 3. 인공지능(AI) 기반 주요 기능

단순한 인식을 넘어 보행에 꼭 필요한 기능을 제공합니다.

- **실시간 객체 탐지:** 전신주, 계단, 차량 등 보행 중 부딪힐 수 있는 장애물을 인식하고 거리를 계산해 경고합니다.
- **텍스트 인식 (OCR):** 카메라가 비추는 곳의 표지판, 식당 메뉴판, 약봉지 글자 등을 읽어주는 부가 기능을 탑재했습니다.

---

## 4. 기기 사양 및 사용 환경 (Spec)

사용자가 체감하는 물리적인 크기와 무게입니다.

| **구분** | **상세 내용** | **비고** |
| --- | --- | --- |
| **전체 무게** | 약 **250g** | 본체(45g) + 쿨러(20g) + 배터리(185g) |
| **본체 두께** | 약 **5cm** | 기판 + 케이스 + 배터리 합계 |
| **사용 시간** | 약 **3~5시간** | 10,000mAh 보조배터리 기준 (연속 사용 시) |
| **오디오 엔진** | gTTS(온라인) / espeak(오프라인) | 환경에 맞게 하이브리드 운영 가능 |

---

## 5. 제작 비용 상세 내역 (2026년 3월 기준)

부품 하나하나의 단가와 구매처를 포함한 총 예산입니다.

| **항목** | **금액** | **주요 특징** | **구매처** |
| --- | --- | --- | --- |
| **라즈베리 파이 5 (4GB)** | 137,610원 | 장치의 두뇌 (디바이스마트) | https://www.devicemart.co.kr/goods/view?no=15215450&srsltid=AfmBOootC3af1xM5TJM4gNUmrjPvQX5Idgdi4burDfG6vpS7Ydlg6EuR |
| **글로벌 셔터 카메라** | 33,509원 | 움직임 왜곡 없는 고성능 카메라 (알리) | https://ko.aliexpress.com/item/1005006975839693.html?spm=a2g0o.productlist.main.4.8d206ab9CDAHw7&aem_p4p_detail=20260318084314576397603924240000455575&algo_pvid=6e3bdf6a-2d08-45bb-a909-47221b99e8b5&algo_exp_id=6e3bdf6a-2d08-45bb-a909-47221b99e8b5-3&pdp_ext_f=%7B%22order%22%3A%2274%22%2C%22spu_best_type%22%3A%22price%22%2C%22eval%22%3A%221%22%2C%22fromPage%22%3A%22search%22%7D&pdp_npi=6%40dis%21KRW%2149915%2133509%21%21%21224.51%21150.72%21%402140c1e917738485940484874e8eab%2112000038910951591%21sea%21KR%212723324665%21ACX%211%210%21n_tag%3A-29919%3Bd%3A55966132%3Bm03_new_user%3A-29894%3BpisId%3A5000000201545892&curPageLogUid=n4vlRNV3grvh&utparam-url=scene%3Asearch%7Cquery_from%3A%7Cx_object_id%3A1005006975839693%7C_p_origin_prod%3A&search_p4p_id=20260318084314576397603924240000455575_1 |
| **SD 카드 (64GB)** | 31,220원 | OS 및 AI 모델 저장용 (쿠팡) | https://www.coupang.com/vp/products/8859296539?itemId=25832507597&vendorItemId=92819202638&src=1042503&spec=10304025&addtag=400&ctag=8859296539&lptag=8859296539-25832507597&itime=20260319153615&pageType=PRODUCT&pageValue=8859296539&wPcid=17679722986579637990639&wRef=www.google.com&wTime=20260319153615&redirect=landing&gclid=Cj0KCQjwmunNBhDbARIsAOndKpnnm1KBCnbeX3xR_273SpAu5uP3BwVJbsx3Yp2ySokqxIquDQaejCAaAi3nEALw_wcB&mcid=940393ab4433459d9f30e07f131f05f8&campaignid=22815108882&adgroupid= |
| **골전도 이어폰** | 29,930원 | 주변 소리 청취 가능 (11번가) | https://www.11st.co.kr/products/8873273053?vkey=RLHIS0HZX24SKUXZ3933LKVUU4XMJ4&utm_term=&utm_campaign=%B4%D9%C0%BDpc_%B0%A1%B0%DD%BA%F1%B1%B3%B1%E2%BA%BB&utm_source=%B4%D9%C0%BD_PC_PCS&utm_medium=%B0%A1%B0%DD%BA%F1%B1%B3 |
| **보조배터리 (10000mAh)** | 22,800원 | 전원 공급 장치 (네이버 쇼핑) | https://smartstore.naver.com/pokoshop/products/11415432806?NaPm=ct%3Dmmw85572%7Cci%3Dshopn%7Ctr%3Ddana%7Chk%3D6fe8d487a4a6c363184840ab244c12dd537c244f%7Ctrx%3Dundefined |
| **액티브 쿨러** | 7,700원 | CPU 열 식힘용 (디바이스마트) | https://www.devicemart.co.kr/goods/view?no=15276241 |
| **아크릴 케이스** | 7,390원 | 하드웨어 보호용 (쿠팡) | https://www.coupang.com/vp/products/8417003992?itemId=24341502513&vendorItemId=91356992570&src=1042503&spec=10304025&addtag=400&ctag=8417003992&lptag=8417003992-24341502513&itime=20260319005416&pageType=PRODUCT&pageValue=8417003992&wPcid=17679722986579637990639&wRef=www.google.com&wTime=20260319005416&redirect=landing&gclid=Cj0KCQjwmunNBhDbARIsAOndKpkRyGZar8IQ7tMsekFY_5L3Vrcj1k1QmOjSt3Hj-laRvgDhjKOX1p8aArfYEALw_wcB&mcid=b6e7bd66766e42749a84cccbb67eb3bd&campaignid=22815108882&adgroupid= |
| **실리콘 고무밴드** | 4,480원 | 부품 고정 및 결합용 (쿠팡) | https://www.coupang.com/vp/products/8518602874?itemId=24662975055&vendorItemId=91671753571&q=%EC%8B%A4%EB%A6%AC%EC%BD%98+%EA%B3%A0%EB%AC%B4%EC%A4%84&searchId=5c9eb27013164203&sourceType=search&itemsCount=36&searchRank=1&rank=1&traceId=mmw63bmj |
| **PD 60W 케이블** | 3,990원 | 안정적인 전력 공급 (네이버 쇼핑) | https://brand.naver.com/redbean/products/9909562988?NaPm=ct%3Dmmw5hb8r%7Cci%3Dshopn%7Ctr%3Ddana%7Chk%3D1a960ee2343d5c690c95c0bd4969106ae1dce37c%7Ctrx%3Dundefined |
| **SD 카드 리더기** | 5,600원 | 데이터 전송 및 세팅용 (쿠팡) | https://www.coupang.com/vp/products/7704899510?itemId=20638672088&vendorItemId=87712314364&src=1032034&spec=10305197&addtag=400&ctag=7704899510&lptag=I20638672088&itime=20260319000308&pageType=PRODUCT&pageValue=7704899510&wPcid=17679722986579637990639&wRef=prod.danawa.com&wTime=20260319000308&redirect=landing&mcid=42f4c44034e44dcda7ee4f677514b94b |
| **총 합계** | **284,229원** |  |  |

---

## 6. 프로젝트의 의의

이 장치는 **① 30만 원 미만의 저비용 ② 인터넷이 필요 없는 보안성 ③ 실제 착용 가능한 디자인**이라는 세 마리 토끼를 잡았습니다. 이를 통해 기술이 단순히 연구실에 머물지 않고, 시각장애인의 일상 속 안전한 보행을 돕는 실질적인 솔루션이 될 수 있음을 보여줍니다.
