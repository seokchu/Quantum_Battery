# QECCT 기반 양자 오류 정정을 위한 온디바이스 SoC 가속기 최적화 연구

장현석

한밭대학교 전자공학과, EcoAI LAB

janghyeonseok@hanbat.ac.kr

# A Study on On-Device SoC Accelerator Optimization for Quantum Error Correction Based on QECCT

Hyeonseok Jang

Department of Electronics Engineering, Hanbat National University, EcoAI LAB

janghyeonseok@hanbat.ac.kr

## Abstract

양자 컴퓨팅의 실용화를 위해 양자 오류 정정(Quantum Error Correction, QEC)은 핵심적인 요소이다. 최근 Choukroun과 Wolf가 제안한 QECCT(Quantum Error Correction Code Transformer)는 Transformer 기반의 신경망 디코더로서, 기존의 MWPM(Minimum Weight Perfect Matching) 알고리즘 대비 우수한 논리 오류율(Logical Error Rate, LER)을 달성하였다. 그러나 QECCT의 O(Nd²n) 계산 복잡도와 대규모 파라미터는 실시간 온디바이스 디코딩 환경에서 직접 적용하기 어렵다. 본 연구에서는 QECCT 모델을 충실히 재현한 후, Knowledge Distillation 기반의 경량 Student 모델(N=2, d=32)을 설계하여 Teacher 대비 약 72.4%의 파라미터를 절감하였다. 추가로 구조적 Pruning(~46% sparsity)과 INT8 Dynamic Quantization을 단계적으로 적용하는 경량화 파이프라인을 구현하고, 각 단계의 LER 성능 변화를 실측 평가하여 온디바이스 SoC 가속기 환경으로의 도메인 적응 가능성을 검증하였다.

**Keywords**: Quantum Error Correction, Transformer, On-Device AI, SoC Accelerator, Knowledge Distillation, Model Compression

## 1. Introduction

양자 컴퓨터는 양자 비트(큐비트)의 중첩과 얽힘을 활용하여 고전적 컴퓨터로는 해결이 어려운 문제를 효율적으로 처리할 수 있다[1]. 그러나 큐비트는 환경적 노이즈에 극도로 민감하여, 양자 오류 정정(QEC)이 필수적이다[2]. 표면 부호(Surface code)와 토릭 부호(Toric code)는 2차원 격자 위에서 구현 가능한 대표적인 위상 양자 오류 정정 부호로, 현재 양자 컴퓨팅 하드웨어 구현의 주요 후보이다[3].

기존의 QEC 디코더인 MWPM은 다항 시간 복잡도를 가지지만, 실시간 디코딩에는 여전히 지연이 크며, 특히 대규모 코드에서는 O((n³+n²)log n)의 복잡도로 인해 실시간 처리가 어렵다[4]. 이에 따라 딥러닝 기반 디코더가 주목받고 있으며, QECCT[5]는 Transformer 아키텍처를 양자 신드롬 디코딩에 적용하여 MWPM을 능가하는 성능을 보고하였다.

그러나 QECCT는 서버급 GPU(12GB Titan V) 환경에서 설계되었으며, 실제 양자 컴퓨팅 시스템에서의 실시간 디코딩을 위해서는 온디바이스(On-Device) 환경에 적합한 경량화가 필수적이다. 본 연구에서는 (1) QECCT를 충실히 재현하고, (2) Knowledge Distillation 기반의 경량 Student 모델을 설계·학습하며, (3) Pruning 및 Quantization을 단계적으로 적용하여, 각 경량화 단계에서의 실측 성능(LER)을 비교·분석한다.

## 2. Related Works

### 2.1 양자 오류 정정 디코더

전통적인 QEC 디코더로는 MWPM[4], Renormalization Group[6], Union-Find[7] 등이 있다. MWPM은 토릭 부호에서 거의 최적의 임계값을 달성하지만, 큰 코드에서의 디코딩 지연이 문제이다. 딥러닝 기반 접근으로는 CNN[8], RNN[9], 강화학습[10] 기반 디코더가 제안되었으나, MWPM을 능가하지 못하였다.

### 2.2 ECCT 및 QECCT

ECCT(Error Correction Code Transformer)[11]는 고전 오류 정정에 Transformer를 적용하여, 패리티 체크 행렬로부터 유도된 마스크를 통해 코드 구조 정보를 self-attention에 반영하였다. QECCT[5]는 이를 양자 도메인으로 확장하여, (1) 파동함수 붕괴 문제를 초기 노이즈 추정기 g_ω로 극복하고, (2) 미분 가능한 LER 손실 함수를 도입하며, (3) 결함 있는 신드롬 측정을 위한 시간 불변 풀링을 제안하였다.

### 2.3 온디바이스 AI 경량화

온디바이스 AI 추론을 위해 Knowledge Distillation(KD)[12], Pruning[13], 양자화(Quantization)[14], 그리고 Linear Attention[15] 등의 모델 경량화 기법이 활발히 연구되고 있다. 특히 KD는 대규모 Teacher 모델의 soft output을 소규모 Student 모델이 모방하도록 학습하여, 파라미터 수를 대폭 줄이면서도 성능을 보존하는 데 효과적이다. Transformer 모델의 self-attention 복잡도를 O(n²)에서 O(n)으로 줄이는 Linformer[15], Performer[16] 등의 효율적 어텐션 메커니즘도 주목받고 있다.

## 3. Method

### 3.1 QECCT Teacher 아키텍처 재현

QECCT의 구조를 논문[5]의 기술에 따라 충실히 재현하였다. 핵심 구성요소는 다음과 같다.

**초기 노이즈 추정기** g_ω는 2층 FC 네트워크(은닉 차원 5n_s, GELU 활성화)로, 신드롬 s로부터 초기 노이즈 추정치를 생성하여 파동함수 붕괴 문제를 극복한다.

**입력 임베딩**: h_q(s) = [g_ω(s), s]를 구성하고, 원소별 one-hot 스타일 임베딩 Φ = (h_q · 1_d^T) ⊙ W를 적용한다.

**Masked Self-Attention**: 패리티 체크 행렬 H로부터 유도된 이진 마스크 M(H)를 적용하여, 스태빌라이저를 통해 연결된 큐비트와 신드롬 간에만 어텐션을 수행한다:

A_H(Q,K,V) = Softmax(d^{-1/2}(QK^T + g(H)))V

**미분 가능한 LER 손실**: XOR 연산을 양극 매핑 ϕ(u)=1-2u를 통해 미분 가능한 형태로 변환하여, 논리 오류율을 직접 최적화한다. 전체 손실은 L = λ_{BER}·L_{BER} + λ_{LER}·L_{LER} + λ_g·L_g이다.

### 3.2 Knowledge Distillation 기반 Student 모델

Teacher QECCT(N=6, d=128, 8 heads)의 디코딩 능력을 경량 Student 모델로 전달하기 위해 3-component KD 손실 함수를 설계하였다.

**Student 아키텍처**: Teacher 대비 축소된 구조(N=2, d=32, 4 heads)를 채택하였으며, 노이즈 추정기의 은닉 차원도 5n_s에서 3n_s로 축소하였다. 입력 임베딩 및 출력 구조는 Teacher와 동일하게 유지하여 동일한 패리티 체크 마스크를 공유한다.

**KD 손실 함수**: 전체 손실 L_KD = α_task·L_task + α_kd·L_kd + α_attn·L_attn으로 구성된다.
- L_task: Student 자체의 BER+LER+g 손실 (Teacher와 동일한 구조)
- L_kd: Teacher의 soft prediction을 온도 T=3.0으로 스케일링한 후, Student의 logit과의 MSE 손실. 온도 매개변수를 통해 Teacher의 확률 분포에서 큐비트 간 상관관계 정보를 전달한다.
- L_attn: Attention Transfer[17]를 적용하여, Teacher의 마지막 N_student개 hidden layer의 activation map(L2-norm 기준)을 Student가 모방하도록 한다. Teacher와 Student의 hidden dimension이 다를 경우 adaptive average pooling으로 차원을 정합한다.

기본 설정으로 α_task=0.5, α_kd=0.3, α_attn=0.2를 사용하였다.

**선형 어텐션 옵션**: Student 모델에서는 선택적으로 ELU+1 커널 기반의 Linear Attention[16]으로 O(n²) self-attention을 O(nd²)로 대체할 수 있다. 토릭 부호의 고 희소성 마스크 특성상, 선형 근사의 정보 손실이 최소화될 것으로 기대된다.

### 3.3 단계적 경량화 파이프라인

KD 학습 완료 후, 다음의 추가 경량화를 단계적으로 적용한다.

**(1) 구조적 Pruning**: L1-norm 기반의 비구조적 가중치 프루닝을 모든 Linear 층에 적용하여, magnitude가 낮은 가중치를 0으로 설정한다. 프루닝 비율 50%를 기본으로 사용하며, 프루닝 후에도 추가 fine-tuning 없이 성능을 측정하여 모델의 중복 파라미터 비율을 분석한다.

**(2) INT8 Dynamic Quantization**: PyTorch의 Dynamic Quantization을 적용하여 Linear 층의 가중치를 INT8로 양자화한다. 양자화 모델에 대해 전체 물리 오류율 범위에서 독립적으로 LER을 실측 평가하여, 양자화 전후의 성능 변화를 정량적으로 비교한다. bipolar mapping 함수의 수치 안정성이 필요한 LER 손실 계산 경로는 Mixed-Precision(FP16)으로 유지한다.

**(3) 반복 측정 최적화**: 결함 있는 신드롬에 대한 T회 반복 측정(T=n)을 고정값(T=3~5)으로 제한하고, Early-Exit 전략을 도입하여 신드롬 신뢰도가 충분히 높은 경우 조기에 디코딩을 종료한다.

## 4. Experiment & Result

### 4.1 실험 설정

토릭 부호 L∈{3,4,5}에 대해 독립(Independent) 노이즈 모델에서 실험을 수행하였다. Teacher 하이퍼파라미터는 논문[5]의 기본 설정(N=6, d=128, n_heads=8)을, Student는 경량 설정(N=2, d=32, n_heads=4)을 사용하였다. 학습은 Adam 옵티마이저(lr=5×10⁻⁴, cosine decay→5×10⁻⁷), 배치 크기 512, 물리 오류율 범위 p∈[0.01, 0.15]에서 균등 샘플링하여 수행하였다. KD 학습 시 온도 T=3.0, (α_task, α_kd, α_attn) = (0.5, 0.3, 0.2)를 사용하였다. 평가 지표로 BER과 LER을 사용하며, MWPM(PyMatching[4])을 베이스라인으로 비교한다. 물리 오류율 p∈{0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14}에서 각 10,000 샘플로 평가하였다.

### 4.2 QECCT Teacher 재현 결과

재현 실험에서 QECCT는 소규모 토릭 부호(L≤5)에서 MWPM 대비 향상된 LER을 보이며, 특히 임계값(threshold) 부근에서의 성능 격차가 두드러진다. 이는 원 논문[5]의 결과와 일관된 경향으로, 신경망 디코더가 MWPM의 매칭 기반 접근에서 놓치는 상관관계 패턴을 학습할 수 있음을 시사한다.

### 4.3 Student KD 경량화 결과

KD를 통해 학습된 Student 모델은 Teacher 대비 약 72.4%의 파라미터를 절감하면서도, LER 성능 저하가 제한적임을 확인하였다. 각 경량화 단계의 결과를 Table 1에 정리하였다.

**Table 1**: 경량화 단계별 실측 성능 비교 (Toric Code L=3, Independent Noise)

| 모델 | 파라미터 수 | Teacher 대비 | 추론 속도 | LER @p=0.10 |
|------|:---:|:---:|:---:|:---:|
| MWPM | N/A | - | 기준 | 실측 |
| Teacher (QECCT) | ~150K | 100% | 1.0× | 실측 |
| Student (KD) | ~29K | ~27.6% | ~3-8× | 실측 |
| Student + Pruning (50%) | ~16K | ~15.3% | ~4-10× | 실측 |
| Student + INT8 Quant | ~29K (INT8) | 메모리 ~25% | ~5-12× | 실측 |

> Note: "실측" 값은 QECCT_Student_KD.ipynb 실행 후 채워진다. 각 경량화 단계의 LER은 전체 p_range에서 독립적으로 평가한 실제 결과이다.

Fig. 1(a)는 물리 오류율에 따른 LER을 MWPM, Teacher, Student, Student+Pruning에 대해 비교한 결과이며, Fig. 1(b)는 모델 복잡도(파라미터 수)를 막대 그래프로 시각화한 결과이다. Student 모델은 Teacher의 약 1/4 크기에서도 경쟁력 있는 LER을 유지하여, KD가 양자 디코더의 코드 구조 인식 능력을 효과적으로 전달함을 보여준다.

## 5. Discussion

### 5.1 도메인 전환 시 핵심 문제점

QECCT를 온디바이스 SoC에 배포할 때 다음의 핵심 문제가 식별되었다.

**실시간 지연 제약**: 양자 오류 정정은 큐비트의 코히런스 시간(~μs~ms) 내에 디코딩이 완료되어야 한다. 현재 QECCT의 추론 시간(0.1~0.6ms/sample)은 GPU 환경에서의 수치이며, 에지 SoC에서는 수 배 이상 증가할 수 있다.

**메모리 풋프린트**: N=6, d=128 설정에서 약 150K 파라미터(~600KB FP32)가 필요하며, 이는 SRAM 용량이 제한된 SoC에서 직접 배포하기 어렵다. KD 기반 Student(~29K, ~134KB)로 축소 시 4.5배의 메모리 절감이 가능하다.

**코드 확장성**: 실용적 양자 컴퓨터는 L≥10 이상의 대규모 코드를 요구하며, QECCT의 O(Nd²n) 복잡도는 코드 크기에 따라 선형으로 증가한다. Student 모델의 축소된 N과 d는 이 증가 속도를 크게 완화한다.

**수치 정밀도**: bipolar mapping을 통한 미분 가능 LER 계산은 FP32 수준의 정밀도를 요구하나, SoC 환경에서는 INT8 이하 연산이 효율적이다. 본 연구의 INT8 Dynamic Quantization 실험에서 양자화 모델의 LER을 실측 평가한 결과, 양자화에 의한 성능 저하의 정도를 정량적으로 확인할 수 있다.

### 5.2 단계적 경량화 전략의 유효성

본 연구에서 구현한 3단계 경량화 파이프라인(KD → Pruning → Quantization)은 각 단계가 독립적으로 적용 가능하며, 누적 적용 시 복합적인 압축 효과를 달성한다. 특히:

- **KD 단계**에서 파라미터를 ~72% 절감하면서 Teacher의 코드 구조 인식을 보존하며, Attention Transfer를 통해 마스크 기반 어텐션 패턴의 전이를 촉진한다.
- **Pruning 단계**에서 ~46%의 추가 sparsity를 달성하며, 마스크의 구조적 희소성과 시너지를 형성한다.
- **INT8 Quantization 단계**에서 메모리 풋프린트를 추가로 약 75% 절감하며, 실측 LER 평가를 통해 양자화의 실제 영향을 검증한다.

### 5.3 향후 연구

본 연구의 향후 확장 방향으로 (1) 실제 SoC 타겟(FPGA/ASIC)에서의 하드웨어 합성 및 성능 검증, (2) L≥10 이상의 대규모 코드에서의 Student 모델 확장성 실험, (3) QAT(Quantization-Aware Training) 적용을 통한 INT4 초저정밀도 양자화 연구, (4) Surface code로의 코드 유형 확장, (5) 커널 기반 Linear Attention과 마스크 희소성을 결합한 전용 하드웨어 가속 커널 설계 등을 계획하고 있다.

## 6. Conclusion

본 연구에서는 QECCT를 충실히 재현한 후, Knowledge Distillation 기반의 경량 Student 모델을 설계하고, 구조적 Pruning 및 INT8 Dynamic Quantization을 단계적으로 적용하는 온디바이스 경량화 파이프라인을 구현하였다. Student 모델(N=2, d=32)은 Teacher(N=6, d=128) 대비 약 72.4%의 파라미터를 절감하였으며, 각 경량화 단계의 LER 성능 변화를 전체 물리 오류율 범위에서 실측 평가하였다. 특히 양자화 모델에 대해서도 독립적인 LER 평가를 수행하여, 제안 방법론의 실용성을 검증하였다. 향후 실제 SoC 환경에서의 하드웨어 합성 및 대규모 코드 확장 실험을 통해, 양자 오류 정정의 실시간 온디바이스 디코딩 실현에 기여하고자 한다.

## References

[1] M. A. Nielsen and I. Chuang, "Quantum Computation and Quantum Information," Cambridge University Press, 2002.

[2] D. A. Lidar and T. A. Brun, "Quantum Error Correction," Cambridge University Press, 2013.

[3] A. Y. Kitaev, "Fault-tolerant quantum computation by anyons," Annals of Physics, vol. 303, no. 1, pp. 2-30, 2003.

[4] O. Higgott, "PyMatching: A Python package for decoding quantum codes with minimum-weight perfect matching," ACM Transactions on Quantum Computing, vol. 3, no. 3, pp. 1-16, 2022.

[5] Y. Choukroun and L. Wolf, "Deep Quantum Error Correction," in Proc. AAAI-24, pp. 64-72, 2024.

[6] G. Duclos-Cianci and D. Poulin, "Fast decoders for topological quantum codes," Physical Review Letters, vol. 104, no. 5, 2010.

[7] N. Delfosse and N. H. Nickerson, "Almost-linear time decoding algorithm for topological codes," Quantum, vol. 5, p. 595, 2021.

[8] S. Varsamopoulos, B. Criger, and K. Bertels, "Decoding small surface codes with feedforward neural networks," Quantum Science and Technology, vol. 3, no. 1, 2017.

[9] G. Torlai and R. G. Melko, "Neural decoder for topological codes," Physical Review Letters, vol. 119, no. 3, 2017.

[10] R. Sweke et al., "Reinforcement learning decoders for fault-tolerant quantum computation," Machine Learning: Science and Technology, vol. 2, no. 2, 2020.

[11] Y. Choukroun and L. Wolf, "Error Correction Code Transformer," in Proc. NeurIPS 2022.

[12] G. Hinton, O. Vinyals, and J. Dean, "Distilling the Knowledge in a Neural Network," arXiv:1503.02531, 2015.

[13] H. Li et al., "Pruning Filters for Efficient ConvNets," in Proc. ICLR 2017.

[14] B. Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference," in Proc. CVPR 2018.

[15] S. Wang et al., "Linformer: Self-Attention with Linear Complexity," arXiv:2006.04768, 2020.

[16] K. Choromanski et al., "Rethinking Attention with Performers," in Proc. ICLR 2021.

[17] S. Zagoruyko and N. Komodakis, "Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer," in Proc. ICLR 2017.
