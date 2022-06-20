We currently support four SOT methods for inference: [DiMP](https://github.com/visionml/pytracking), [SiamRPN++](https://github.com/STVIR/pysot), [Stark](https://github.com/researchmm/Stark), and our [UTT](https://arxiv.org/abs/2203.15175). More methods could be easily reproduced with our codebase.

Data and Checkpoint structures
```
data
   |——————coco
   |        └——————train2017
   |        └——————annotations
   └——————GOT10k
   |        └——————train
   |        └——————val
   |        └——————test
   └——————LaSOTBenchmark
   |        └——————airplane
   |        |      ...
   |        └——————volleyball
   └——————OTB
   |        └——————Basketball
   |        |      ...
   |        └——————Woman
   └——————TrackingNet
   |        └——————TRAIN_0
   |        |      ...
   |        └——————TRAIN_11
   |        └——————TEST
checkpoints
   |——————dimp
   |        └——————dimp50.pth
   |        └——————super_dimp50.pth
   |——————siamrpn
   |        └—————siamrpn50.pth
   |——————stark
   |        └——————stark50.pth
   |——————utt
   |        └——————utt_sot50.pth
```

Test SiamRPN
```
 torchrun --nproc_per_node 8 --master_port 9999 tools/train_dist.py --config-file configs/siamrpn/siamrpn++.yaml --config-func siamrpn --eval-only 
```

Test DiMP
```
 torchrun --nproc_per_node 8 --master_port 9999 tools/train_dist.py --config-file configs/dimp/dimp50.yaml --config-func dimp --eval-only 
 ```

Test Stark
```
 torchrun --nproc_per_node 8 --master_port 9999 tools/train_dist.py --config-file configs/stark/stark50.yaml --config-func stark --eval-only
```

Test UTT 
```
 torchrun --nproc_per_node 8 --master_port 9999 tools/train_dist.py --config-file configs/utt/utt.yaml --config-func utt --eval-only
```