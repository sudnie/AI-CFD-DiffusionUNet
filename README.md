# HighRes AI-CFD DiffusionUNet (고해상도 유동장 예측)

**관련 학술 논문**:  
JIN ZHEXU\(^1\), 신정훈\(^2\), 조금원\(^3*\) (금오공과대학교, 2024).  
"High-Resolution Diffusion Model을 이용한 익형 유동장 예측"

## 프로젝트 구조

```
AI-CFD-DiffusionUnet-release/
├── Code/
│   ├── highres_steady/           # 정상류 NACA0012 (Ch3)
│   │   ├── code/                 # 소스 코드
│   │   │   ├── highres_model.py       # 모델 프레임워크 (HighResDiffusionUNet)
│   │   │   ├── highres_train.py       # 훈련 스크립트
│   │   │   ├── highres_prepare_data.py# 데이터 전처리
│   │   │   ├── highres_crop_data.py   # 데이터 크롭 도구
│   │   │   ├── highres_infer.py       # 추론 스크립트
│   │   │   ├── highres_infer_full.py  # 전체 테스트셋 추론
│   │   │   ├── highres_infer_epochs.py# 다모델 비교
│   │   │   └── highres_eval_cp.py     # Cp 계수 평가
│   │   └── results/            # 결과 출력 디렉토리
│   │
│   └── highres_unsteady/       # 비정상류 Eppler 387 (Ch2)
│       ├── code/                 # 소스 코드
│       │   ├── highres_model.py       # 모델 프레임워크
│       │   ├── highres_train.py       # 훈련 스크립트
│       │   ├── highres_prepare_data.py# 데이터 전처리
│       │   ├── highres_infer.py       # 추론 스크립트
│       │   ├── highres_infer_cp.py    # Cp 추론
│       │   ├── highres_infer_vortex.py# 와류 추론
│       │   └── highres_infer_animation.py # 애니메이션 추론
│       └── results/            # 결과 출력 디렉토리
│
├── Database/                     # CFD 데이터 디렉토리 (사용자가 배치)
│   ├── Simdata_airfoil_Steady_FC/         # 정상류 데이터
│   └── Simdata_airfoil_unsteady/        # 비정상류 데이터
│
├── requirements.txt
└── .gitignore
```

## 핵심 설명

### Ch3: 정상류 NACA0012 (고해상도 DiffusionUNet)
- **highres_model.py** - `HighResDiffusionUNet` 모델 정의 (4단 U-Net + AdaIN)
- **highres_train.py** - 정상류 훈련 스크립트 (2000 epoch, batch 16, lr 7e-4)
- **highres_prepare_data.py** - 데이터 전처리 (401×81 C-그리드 → 60×301)
- **highres_crop_data.py** - 데이터 컷actoring 도구
- **highres_infer.py** - 단일 인스턴스 추론 (DDIM 50-ste samples)
- **highres_infer_full.py** - 전체 테스트셋 추론
- **highres_infer_epochs.py** - 다모델 비교
- **highres_eval_cp.py** - 공기역학 계수 평가
- **highres_metrics.py** - 오류 지표 도구 (NumPy 구현)

### Ch2: 비정상류 Eppler 387 (DiffusionUNet)
- **highres_model.py** - `DiffusionUNet` 모델 정의 (대형 C-그리드)
- **highres_train.py** - 비정상류 훈련 스크립트 (5000 epoch, batch 48, lr 2e-4)
- **highres_prepare_data.py** - 데이터 전처리 (145×689 C-그리드 → 36×172 down4)
- **highres_infer.py** - 단일 인스턴스 추론 (DDIM 50-ste samples)
- **highres_infer_cp.py** - Cp 계수 추론
- **highres_infer_vortex.py** - 와류 분석 추론
- **highres_infer_animation.py** - 애니메이션 생성 추론
- **highres_metrics.py** - 오류 지표 도구 (NumPy 구현)

## 설치 및 의존성
```bash
pip install torch numpy pandas matplotlib scipy tqdm
```

## 경로 설정 방법

1. **데이터베이스 (`Database/`)**:
   - 원본 CFD 데이터를 프로젝트 루트 디렉토리의 `Database/` 하위 디렉토리에 배치하세요
   - 정상류 데이터: `Database/Simdata_airfoil_Steady_FC/`
   - 비정상류 데이터: `Database/Simdata_airfoil_unsteady/`

2. **결과 출력**:
   - 각 서브시스템의 `results/` 디렉토리가 자동으로 생성됩니다
   - 훈련 및 추론 출력 파일은 해당 디렉토리에 저장됩니다

3. **스크립트 실행**:
   - `Code/highres_X` 디렉토리에서 실행하세요
   - 데이터 경로는 프로젝트 루트 `Database/` 를 참조합니다

## 사용예

### 정상류 훈련 (Ch3)
```bash
cd Code/highres_steady/code
python highres_train.py
```

### 비정상류 훈련 (Ch2)
```bash
cd Code/highres_unsteady/code
python highres_train.py
```

### 비정상류 추론 (Ch2 예시)
```bash
cd Code/highres_unsteady/code
python highres_infer.py
```

## 주의사항
- 모든 스크립트는 UTF-8 인코딩입니다
- GPU 환경에서 실행을 권장하며, CUDA 11.x / 12.x 에 적합합니다

## 인용
본 배포판은 다음 논문 및 작업에 기반합니다:
- 중국어 보고서: "확산 모델을 이용한 익형 유동장 예측 기술 연구"
- 한국어 논문: "High-Resolution Diffusion Model 을 이용한 익형 유동장 예측" (JIN ZHEXU, 신정훈, 조금원)

---
**버전**: v1.0  
**배포**: 2024