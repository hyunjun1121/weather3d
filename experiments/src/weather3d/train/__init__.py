"""C3 fine-tuning (실험맥락 §8, 2026-08-23 사용자 확정 설계).

구성:
- data.py:     7-Scenes TrainSplit + NRGBD 비평가 장면 윈도우 데이터셋.
               Track A v2(synth/) 재사용, 즉석 (clean, degraded) 쌍 생성.
- trainer.py:  StreamVGGT(student) + VGGT(teacher) 증류 트레이너.
               teacher는 clean 입력, student는 clean/열화 혼합 입력.

13/13b 서베이로 확정된 사실(2026-08-26):
- ckpt(1,797키 flat float32)는 StreamVGGT/VGGT 양쪽 strict 로드 호환.
- loss_of_one_batch의 autocast는 capability<8에서 fp16을 이미 선택.
  남은 bf16 하드코딩은 Accelerator(mixed_precision=...) 한 공뿐이라
  여기서 "fp16"으로 지정한다.
- upstream 데이터셋 레지스트리에 7-Scenes/NRGBD 어댑터가 없어 자체 구현.
"""
