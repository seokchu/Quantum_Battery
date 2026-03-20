# QECCT 기반 양자 오류 정정을 위한 온디바이스 SoC 가속기 최적화 연구

장현석

한밭대학교 전자공학과, EcoAI LAB

janghyeonseok@hanbat.ac.kr

# A Study on On-Device SoC Accelerator Optimization for Quantum Error Correction Based on QECCT

Hyeonseok Jang

Department of Electronics Engineering, Hanbat National University, EcoAI LAB

janghyeonseok@hanbat.ac.kr

## Abstract

양자 컴퓨팅의 실용화를 위해 양자 오류 정정(Quantum Error Correction, QEC)은 핵심적인 요소이다. 최근 Choukroun과 Wolf가 제안한 QECCT(Quantum Error Correction Code Transformer)는 Transformer 기반의 신경망 디코더로서, 기존의 MWPM(Minimum Weight Perfect Matching) 알고리즘 대비 우수한 논리 오류율(Logical Error Rate, LER)을 달성하였다. 그러나 QECCT의 O(Nd²n) 계산 복잡도와 대규모 파라미터는 실시간 온디바이스 디코딩 환경에서 직접 적용하기 어렵다. 본 연구에서는 QECCT 모델을 충실히 재현한 후, 이를 온디바이스 SoC 가속기 환경으로 전환할 때 발생하는 핵심 문제점을 분석하고, Knowledge Distillation, 구조적 Pruning, INT8/INT4 양자화, Linear Attention 등의 경량화 및 최적화 기법을 제안한다.

**Keywords**: Quantum Error Correction, Transformer, On-Device AI, SoC Accelerator, Model Compression

## 1. Introduction

양자 컴퓨터는 양자 비트(큐비트)의 중첩과 얽힘을 활용하여 고전적 컴퓨터로는 해결이 어려운 문제를 효율적으로 처리할 수 있다[1]. 그러나 큐비트는 환경적 노이즈에 극도로 민감하여, 양자 오류 정정(QEC)이 필수적이다[2]. 표면 부호(Surface code)와 토릭 부호(Toric code)는 2차원 격자 위에서 구현 가능한 대표적인 위상 양자 오류 정정 부호로, 현재 양자 컴퓨팅 하드웨어 구현의 주요 후보이다[3].

기존의 QEC 디코더인 MWPM은 다항 시간 복잡도를 가지지만, 실시간 디코딩에는 여전히 지연이 크며, 특히 대규모 코드에서는 O((n³+n²)log n)의 복잡도로 인해 실시간 처리가 어렵다[4]. 이에 따라 딥러닝 기반 디코더가 주목받고 있으며, QECCT[5]는 Transformer 아키텍처를 양자 신드롬 디코딩에 적용하여 MWPM을 능가하는 성능을 보고하였다.

그러나 QECCT는 서버급 GPU(12GB Titan V) 환경에서 설계되었으며, 실제 양자 컴퓨팅 시스템에서의 실시간 디코딩을 위해서는 온디바이스(On-Device) 환경에 적합한 경량화가 필수적이다. 본 연구에서는 (1) QECCT를 충실히 재현하고, (2) 온디바이스 SoC 가속기 환경으로의 도메인 전환 시 발생하는 문제점을 식별하며, (3) 이를 해결하기 위한 최적화 전략을 제안한다.

## 2. Related Works

### 2.1 양자 오류 정정 디코더

전통적인 QEC 디코더로는 MWPM[4], Renormalization Group[6], Union-Find[7] 등이 있다. MWPM은 토릭 부호에서 거의 최적의 임계값을 달성하지만, 큰 코드에서의 디코딩 지연이 문제이다. 딥러닝 기반 접근으로는 CNN[8], RNN[9], 강화학습[10] 기반 디코더가 제안되었으나, MWPM을 능가하지 못하였다.

### 2.2 ECCT 및 QECCT

ECCT(Error Correction Code Transformer)[11]는 고전 오류 정정에 Transformer를 적용하여, 패리티 체크 행렬로부터 유도된 마스크를 통해 코드 구조 정보를 self-attention에 반영하였다. QECCT[5]는 이를 양자 도메인으로 확장하여, (1) 파동함수 붕괴 문제를 초기 노이즈 추정기 g_ω로 극복하고, (2) 미분 가능한 LER 손실 함수를 도입하며, (3) 결함 있는 신드롬 측정을 위한 시간 불변 풀링을 제안하였다.

### 2.3 온디바이스 AI 가속기

온디바이스 AI 추론을 위해 Knowledge Distillation[12], Pruning[13], 양자화(Quantization)[14], 그리고 Linear Attention[15] 등의 모델 경량화 기법이 활발히 연구되고 있다. 특히 Transformer 모델의 self-attention 복잡도를 O(n²)에서 O(n)으로 줄이는 Linformer[15], Performer[16] 등의 효율적 어텐션 메커니즘이 주목받고 있다.

## 3. Method

### 3.1 QECCT 아키텍처 재현

QECCT의 구조를 논문[5]의 기술에 따라 충실히 재현하였다. 핵심 구성요소는 다음과 같다.

**초기 노이즈 추정기** g_ω는 2층 FC 네트워크(은닉 차원 5n_s, GELU 활성화)로, 신드롬 s로부터 초기 노이즈 추정치를 생성하여 파동함수 붕괴 문제를 극복한다.

**입력 임베딩**: h_q(s) = [g_ω(s), s]를 구성하고, 원소별 one-hot 스타일 임베딩 Φ = (h_q · 1_d^T) ⊙ W를 적용한다.

**Masked Self-Attention**: 패리티 체크 행렬 H로부터 유도된 이진 마스크 M(H)를 적용하여, 스태빌라이저를 통해 연결된 큐비트와 신드롬 간에만 어텐션을 수행한다:

A_H(Q,K,V) = Softmax(d^{-1/2}(QK^T + g(H)))V

**미분 가능한 LER 손실**: XOR 연산을 양극 매핑 ϕ(u)=1-2u를 통해 미분 가능한 형태로 변환하여, 논리 오류율을 직접 최적화한다. 전체 손실은 L = λ_{BER}·L_{BER} + λ_{LER}·L_{LER} + λ_g·L_g이다.

### 3.2 온디바이스 도메인 적응을 위한 최적화 기법

QECCT를 온디바이스 SoC 환경에 배포하기 위해 다음의 경량화 파이프라인을 제안한다.

**(1) Knowledge Distillation**: 전체 QECCT(Teacher)로부터 축소된 아키텍처(Student, N=2, d=32)로 지식을 전달한다. 특히 intermediate layer의 어텐션 패턴을 전달하는 Attention Transfer를 활용하여 코드 구조 인식 능력을 보존한다.

**(2) 구조적 Pruning**: L1-norm 기반의 필터 수준 프루닝을 적용하여 50-70%의 채널을 제거한다. 마스크 기반 어텐션의 구조적 희소성을 활용하여, 이미 비활성화된 어텐션 경로를 우선적으로 제거한다.

**(3) INT8/INT4 양자화**: Post-Training Quantization(PTQ)과 Quantization-Aware Training(QAT)을 적용한다. 특히 bipolar mapping 함수 ϕ의 수치 안정성을 위해 Mixed-Precision 전략을 사용하여 LER 손실 계산 경로는 FP16을 유지한다.

**(4) Linear Attention 대체**: O(n²) 복잡도의 self-attention을 Linformer[15] 또는 커널 기반 선형 어텐션으로 대체하여 O(n)으로 감소시킨다. 토릭 부호의 고 희소성(sparsity) 마스크 특성상, 이미 대부분의 어텐션 가중치가 마스킹되어 있어 선형 근사의 정보 손실이 최소화될 것으로 기대된다.

**(5) 반복 측정 최적화**: 결함 있는 신드롬에 대한 T회 반복 측정(T=n)을 고정값(T=3~5)으로 제한하고, Early-Exit 전략을 도입하여 신드롬 신뢰도가 충분히 높은 경우 조기에 디코딩을 종료한다.

## 4. Experiment & Result

### 4.1 실험 설정

토릭 부호 L∈{3,4,5}에 대해 독립(Independent) 노이즈 모델과 탈분극(Depolarization) 노이즈 모델에서 실험을 수행하였다. 모델 하이퍼파라미터는 논문[5]의 기본 설정(N=6, d=128, n_heads=8)을 따르며, Adam 옵티마이저(lr=5×10⁻⁴, cosine decay→5×10⁻⁷), 배치 크기 512를 사용하였다. 평가 지표로 BER(Bit Error Rate)과 LER(Logical Error Rate)을 사용하며, MWPM(PyMatching[4])을 베이스라인으로 비교한다.

### 4.2 QECCT 재현 결과

재현 실험에서 QECCT는 소규모 토릭 부호(L≤5)에서 MWPM 대비 향상된 LER을 보이며, 특히 임계값(threshold) 부근에서의 성능 격차가 두드러진다. 이는 원 논문[5]의 결과와 일관된 경향으로, 신경망 디코더가 MWPM의 매칭 기반 접근에서 놓치는 상관관계 패턴을 학습할 수 있음을 시사한다.

### 4.3 경량화 예비 분석

경량화 적용 시 예상되는 성능-효율성 트레이드오프를 Table 1에 정리하였다.

**Table 1**: 경량화 기법별 예상 성능 영향

| 기법 | 파라미터 감소 | 추론 속도 향상 | LER 영향 |
|------|:---:|:---:|:---:|
| KD (N=6→2, d=128→32) | ~90% | ~8× | Δ < 5% |
| Pruning (50%) | ~50% | ~2× | Δ < 3% |
| INT8 Quantization | ~75% (메모리) | ~2-4× | Δ < 1% |
| Linear Attention | - | ~2-3× (대규모) | Δ < 2% |
| 반복 측정 제한 (T=5) | - | T/5× | 코드 의존적 |

## 5. Discussion

### 5.1 도메인 전환 시 핵심 문제점

QECCT를 온디바이스 SoC에 배포할 때 다음의 핵심 문제가 식별되었다.

**실시간 지연 제약**: 양자 오류 정정은 큐비트의 코히런스 시간(~μs~ms) 내에 디코딩이 완료되어야 한다. 현재 QECCT의 추론 시간(0.1~0.6ms/sample)은 GPU 환경에서의 수치이며, 에지 SoC에서는 수 배 이상 증가할 수 있다.

**메모리 풋프린트**: N=6, d=128 설정에서 수백만 개의 파라미터가 필요하며, 이는 SRAM 용량이 제한된 SoC에서 직접 배포하기 어렵다.

**코드 확장성**: 실용적 양자 컴퓨터는 L≥10 이상의 대규모 코드를 요구하며, QECCT의 O(Nd²n) 복잡도는 코드 크기에 따라 선형으로 증가한다.

**수치 정밀도**: bipolar mapping을 통한 미분 가능 LER 계산은 FP32 수준의 정밀도를 요구하나, SoC 환경에서는 INT8 이하 연산이 효율적이다.

### 5.2 해결 전략

상기 문제를 해결하기 위해 다음의 단계적 전략을 제안한다. 먼저 Knowledge Distillation을 통해 모델 크기를 ~90% 축소한 후, 구조적 Pruning으로 잔여 불필요 파라미터를 제거한다. 다음으로 QAT 기반 INT8 양자화를 적용하되, LER 계산 경로는 FP16 Mixed-Precision으로 유지한다. 마지막으로 마스크의 구조적 희소성을 활용한 하드웨어 최적화 커널을 설계하여, SoC의 전용 가속 유닛에서 효율적으로 실행되도록 한다.

### 5.3 향후 연구

본 연구의 향후 확장 방향으로 (1) 실제 SoC 타겟(FPGA/ASIC)에서의 하드웨어 합성 및 성능 검증, (2) Surface code로의 확장 실험, (3) 실시간 적응형 디코더 설계 등을 계획하고 있다.

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
