We currently support fou MOT methods: [TransTrack](https://github.com/PeizeSun/TransTrack), [FairMOT](https://github.com/ifzhang/FairMOT), [ByteTrack](https://github.com/ifzhang/ByteTrack), and our [UTT](https://arxiv.org/abs/2203.15175). More methods could be easily reproduced with our codebase.

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
   └——————crowdhuman
   |         └——————Crowdhuman_train
   |         └——————Crowdhuman_val
   |         └——————annotation_train.odgt
   |         └——————annotation_val.odgt
   └——————MOT
   |        └——————train
   |        └——————test
checkpoints
   |——————fairmot
   |        └——————dla34.pth
   |——————transtrack
   |        └—————transtrack50.pth
   |——————yolox
   |        └——————yolox.pth
   |——————utt
   |        └——————utt_mot.pth
```

Test FairMOT
```
 torchrun --nproc_per_node 8 --master_port 9999 tools/train_dist.py --config-file configs/fairmot/fairmot.yaml --config-func fairmot --mode mot --eval-only
 ```

 Test TransTrack
 ```
 torchrun --nproc_per_node 8 --master_port 9999 tools/train_dist.py --config-file configs/transtrack/transtrack.yaml --config-func transtrack --mode mot --eval-only
 ```

 Test ByteTrack
 ```
 torchrun --nproc_per_node 8 --master_port 9999 tools/train_dist.py --config-file configs/bytetrack/bytetrack.yaml --config-func bytetrack --mode mot --eval-only
 ```

 Test UTT
 ```
 torchrun --nproc_per_node 8 --master_port 9999 tools/train_dist.py --config-file configs/utt/utt.yaml --config-func utt --mode mot --eval-only
 ```


For training, preprocessing MOT datasets to generate annotations
```
python trackron/data/datasets/data_specs/mot_to_coco.py --data-root $MOT_ROOT
```

preprocessing CrowdHuman datasets to generate annotations
```
python trackron/data/datasets/data_specs/crowdhuman_to_coco.py --data-root $CRODHUMAN_ROOT
```
