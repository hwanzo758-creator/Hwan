# GitHub 저장소 정리 — 리뷰어 4 L1.5 대응

## 문제 진단

리뷰어 4의 2차 코멘트에 이렇게 적혀 있습니다.

> "a reproducibility commitment that is **stated but not yet delivered**"

확인해보니 **1차 리비전 때 준비한 `github_release/` 폴더가 실제로 push되지 않았습니다.** 온라인 저장소(`hwanzo758-creator/UAV-Multispectral-Super-Resolution`)와 로컬 준비본이 다릅니다.

| | 온라인 저장소 (현재) | 로컬 `1st_revision/github_release/` |
|---|---|---|
| 파이프라인 | `train_sr_all_crops_Joint_5ch_x234_SR.py` | `Final_version5_revision.py` |
| 지원 모델 (README) | **6개** (HAT·DAT 없음) | 8개 |
| 평가/메트릭 코드 | 없음 | 포함 |
| scene index | 없음 | `scene_index_template.csv` |
| 그림 생성 스크립트 | 없음 | 3종 |
| README 파일명 | `build_light_cabbage_gangwon_dataset.py` 로 기재 (실제는 `build_uav_5band_dataset.py`) | 정합 |

리뷰어가 저장소를 직접 열어봤다면 README의 "Supported SR Models" 표에 HAT·DAT가 없는 것을 봤을 것이고, 원고가 "코드를 공개했다"고 주장하는 것과 어긋납니다.

---

## 올릴 내용 (`2nd_revision/github_release_v2/`)

```
Final_version6_mambair.py     메인 파이프라인 — 9개 방법 + 모든 리비전 실험
selftest_mambair.py           MambaIR 사전 점검 (스캔 정확성·파라미터·메모리)
check_env.py                  라이브러리/GPU 확인
Make_NDVI_maps.py             단일 씬 오차맵 생성
make_rgb_grids.py             RGB·false-color·zoom 그리드 생성
make_fig2_boxplot.py          Figure 2 박스플롯
scene_index_template.csv      scene index 스키마
requirements.txt
README.md                     9개 방법 기준으로 갱신 + 재현 명령 + 항목별 코드 위치
README_REVISION.md
RELEASE_CHECKLIST.md
CITATION.cff / LICENSE / .gitignore / .gitattributes
```

**README에 새로 넣은 두 절**

1. **Reproducing the Reported Numbers** — 환경변수만으로 각 표를 재현하는 명령. 소스 수정이 필요 없습니다.
2. **Where the Reviewer-Requested Items Live in the Code** — R4가 명시적으로 요구한 항목(데이터 무결성 필터, 패치 품질 임계값, 마스킹, 메트릭 구현, 고정 시드)이 코드 어디에 있는지 표로 매핑. **리뷰어가 확인하기 쉽게 만드는 게 핵심입니다.**

---

## push 절차

```powershell
cd "D:\2026\논문\UAV_multispectral_SR_manuscript\Revision\2nd_revision\github_release_v2"

git init
git remote add origin https://github.com/hwanzo758-creator/UAV-Multispectral-Super-Resolution.git
git fetch origin
git checkout -b main
git add -A
git commit -m "Release revision code: nine SR methods incl. MambaIR, spectral metrics, robustness experiments"
git push -f origin main
```

기존 이력을 남기고 싶으면 `-f` 대신 clone 후 파일을 덮어쓰고 커밋하세요.

```powershell
git clone https://github.com/hwanzo758-creator/UAV-Multispectral-Super-Resolution.git repo
# github_release_v2 내용을 repo/ 에 복사한 뒤
cd repo && git add -A && git commit -m "..." && git push
```

---

## push 전 확인

- [ ] `Final_version6_mambair.py`의 `DATASET_DIR` / `OUTPUT_ROOT` 기본값이 **개인 경로(`/home/whanjo/...`)** 로 남아 있습니다. 플레이스홀더로 바꾸거나, 환경변수로 덮어쓰는 방식이라는 점을 README에서 명확히 하세요.
- [ ] `scene_index.csv` 실제 파일을 올릴지 결정. AI Hub 재배포 정책상 어려우면 template만 두고, **원고 Data Availability 문구도 "scene index를 공개한다"가 아니라 "scene index 스키마와 재생성 규칙을 공개한다"로 맞춰야** 합니다. 지금 원고는 "the scene-index file are publicly available"이라고 적혀 있어 실제와 어긋납니다.
- [ ] `.npy`, `.pth`, 체크포인트, 생성 이미지가 커밋되지 않는지 `.gitignore` 확인
- [ ] 저장소 Description 설정: `UAV five-band multispectral super-resolution benchmark: spectral-consistency metrics, VI preservation, nine SR methods including MambaIR, degradation and noise robustness.`
- [ ] `CITATION.cff`에 저장소 URL 반영 (완료), 논문 DOI는 게재 후 추가
- [ ] 공개 후 `v1.0.0` 태그

## 원고에서 함께 고칠 곳

현재 Data Availability Statement:

> "the training and evaluation code (data-integrity filters, patch-quality thresholds, valid-pixel masking, metric implementations, and the fixed random seed) together with the scene-index file are publicly available on GitHub at ... (accessed on 28 July 2026)."

- `scene-index file`을 실제로 올리지 않을 거면 문구 수정 필요
- 접속일(28 July 2026)을 최종 제출일로 갱신
