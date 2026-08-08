Method: WaveRep

Implementation: pretrained inference

Dataset: GenVidBench

Subset: common-1000

Samples: 500 real + 500 fake

Valid samples: 1000

Invalid samples: 0



Weights:

D:\\HaitongNi\\MEngProject\\repos\\WaveRep-SyntheticVideoDetection\\demo\\weights\\weights\_dinov2\_G4.ckpt



Results:

ROC-AUC\_fake = 0.9953

AP\_fake = 0.9956

Accuracy = 0.9600

Precision\_fake = 0.9694

Recall\_fake = 0.9500

F1\_fake = 0.9596

MCC = 0.9202



Confusion Matrix \[\[TN, FP], \[FN, TP]]:

\[\[485, 15],

&#x20;\[25, 475]]



Prediction CSV:

D:\\HaitongNi\\MEngProject\_SecondBench\\01\_WaveRep\\results\\waverep\_genvidbench\_common1000\_predictions.csv



Note:

The printed dataset name still says AIGVDBench because of a leftover display string in the script. The actual input CSV was GenVidBench common-1000.



Method: D3

Implementation: pretrained / feature-based inference

Dataset: GenVidBench

Subset: common-1000

Samples: 500 real + 500 fake

Total samples: 1000



Encoder: XCLIP-16

Loss Type: L2



Input format:

16 uniformly sampled frames per video

Real CSV:

D:\\HaitongNi\\MEngProject\_SecondBench\\02\_D3\\metadata\\genvidbench\_common1000\_real\_frames.csv

Fake CSV:

D:\\HaitongNi\\MEngProject\_SecondBench\\02\_D3\\metadata\\genvidbench\_common1000\_fake\_frames.csv



Threshold-free metrics:

AP\_fake = 0.7510

AP\_real = 0.6468

ROC-AUC\_fake = 0.7438

ROC-AUC\_real = 0.7438



Best-threshold diagnostic metrics:

Best Threshold on fake\_score = -4.303232

Best Accuracy = 0.6960

Best Precision\_fake = 0.6750

Best Recall\_fake = 0.7560

Best F1\_fake = 0.7132



Confusion Matrix \[\[TN, FP], \[FN, TP]]:

\[\[318, 182],

&#x20;\[122, 378]]



Prediction CSV:

D:\\HaitongNi\\MEngProject\\repos\\D3\\results\\genvidbench\_common1000\\predictions\_20260518\_180250.csv



Note:

The best-threshold metrics are diagnostic only because the threshold was selected on the same test subset. ROC-AUC and AP are more appropriate for threshold-free comparison.



Per-generator breakdown:

Note: D3 was evaluated on GenVidBench common-1000 using 16 uniformly sampled frames per video. The recall values below use the diagnostic best threshold selected on the same test subset: fake\_score >= -4.303232.



Metric definitions:

fake\_recall\_at\_best\_threshold = proportion of fake videos from this generator classified as fake using the diagnostic best threshold

mean\_fake\_score = average D3 fake score for this generator

auc\_vs\_all\_real = AUC using this generator's fake samples vs all 500 real samples

ap\_vs\_all\_real = AP using this generator's fake samples vs all 500 real samples



Harder generators for D3:

t2vz      n=62  recall=0.0161  mean\_score=-11.6096  AUC\_vs\_real=0.0970  AP\_vs\_real=0.0606

musev     n=63  recall=0.5714  mean\_score=-4.1239   AUC\_vs\_real=0.6518  AP\_vs\_real=0.1657

svd       n=62  recall=0.6613  mean\_score=-3.8483   AUC\_vs\_real=0.6860  AP\_vs\_real=0.1607



Generators with relatively high fake recall:

mora      n=63  recall=0.9365  mean\_score=-2.2407   AUC\_vs\_real=0.8740  AP\_vs\_real=0.4235

ms        n=63  recall=0.9365  mean\_score=-2.3172   AUC\_vs\_real=0.8734  AP\_vs\_real=0.3919

cogvideo  n=63  recall=0.9524  mean\_score=-2.3134   AUC\_vs\_real=0.8757  AP\_vs\_real=0.3846

pika      n=62  recall=0.9839  mean\_score=-1.5624   AUC\_vs\_real=0.9339  AP\_vs\_real=0.5831

vc2       n=62  recall=0.9839  mean\_score=-1.3258   AUC\_vs\_real=0.9538  AP\_vs\_real=0.6324



Observation:

D3 shows strong generator-dependent behavior on GenVidBench. It separates several fake generators reasonably well, especially vc2, pika, cogvideo, ms, and mora. However, it almost completely fails on t2vz, with only 1.6% fake recall at the diagnostic threshold and an AUC below 0.5 against real videos. This suggests that D3's representation is highly sensitive to generator-specific artifacts and does not provide uniformly robust fake-video detection across all GenVidBench generators.





Method: NSG-VD

Implementation: pretrained inference / standard-Pika-d checkpoint

Dataset: GenVidBench

Subset: common-1000

Samples: 500 real + 500 fake



Checkpoint:

ckpts\\standard-Pika-d.pth



Feature setting:

NSG-VD velocity features

diffuse\_steps = 5

num\_frames = 8

ref\_load\_len = 500

test\_load\_len = 500



Results:

AUROC = 0.6886

Accuracy = 0.5510

Precision\_fake = 0.7339

Recall\_fake = 0.1600

F1\_fake = 0.2627



Observation:

NSG-VD shows weak-to-moderate ranking ability on GenVidBench, with AUROC = 0.6886. Its fake precision is relatively high, but fake recall is very low, meaning it only detects a small portion of fake videos while missing most fake samples. This is similar to its behavior on AIGVDBench, where the default threshold was also conservative and poorly calibrated for the target benchmark.



Per-generator breakdown:

Note: NSG-VD was evaluated on GenVidBench common-1000 using pretrained standard-Pika-d checkpoint. The default NSG-VD decision threshold is raw\_score > 1.



Metric definitions:

fake\_recall\_at\_1 = proportion of fake videos from this generator classified as fake using raw\_score > 1

mean\_score\_fake = average NSG-VD raw fake score for this generator

auc\_vs\_all\_real = AUC using this generator's fake samples vs all 500 real samples

ap\_vs\_all\_real = AP using this generator's fake samples vs all 500 real samples



Per-generator results:

pika      n=62  recall=0.0968  mean\_score=0.5191  AUC\_vs\_real=0.7188  AP\_vs\_real=0.1991

t2vz      n=62  recall=0.1129  mean\_score=0.4871  AUC\_vs\_real=0.6668  AP\_vs\_real=0.1827

mora      n=63  recall=0.1429  mean\_score=0.5143  AUC\_vs\_real=0.6458  AP\_vs\_real=0.1862

svd       n=62  recall=0.1613  mean\_score=0.5668  AUC\_vs\_real=0.7132  AP\_vs\_real=0.2092

ms        n=63  recall=0.1746  mean\_score=0.5936  AUC\_vs\_real=0.7171  AP\_vs\_real=0.2187

vc2       n=62  recall=0.1774  mean\_score=0.5709  AUC\_vs\_real=0.7015  AP\_vs\_real=0.2111

musev     n=63  recall=0.1905  mean\_score=0.5837  AUC\_vs\_real=0.6610  AP\_vs\_real=0.2119

cogvideo  n=63  recall=0.2222  mean\_score=0.5720  AUC\_vs\_real=0.6855  AP\_vs\_real=0.2086



Observation:

NSG-VD shows consistently low fake recall across all GenVidBench fake generators at its default threshold raw\_score > 1. No generator reaches 25% fake recall, which explains the low overall recall of 0.1600. However, the AUC values are mostly around 0.65–0.72, suggesting that NSG-VD has some ranking ability but its default threshold is poorly calibrated for this benchmark.



Compared with WaveRep and D3, NSG-VD is much more conservative. WaveRep detects most fake generators with high recall, D3 has strong generator-dependent variation, while NSG-VD produces uniformly low recall across all generator categories.





Method: VidGuard-R1-style / Qwen2.5-VL

Implementation: zero-shot VLM multiple-choice inference

Dataset: GenVidBench

Subset: common-1000

Samples: 500 real + 500 fake



Model:

Qwen/Qwen2.5-VL-7B-Instruct



Prompt setting:

A = AI-generated/fake

B = real



Valid samples: 997

Invalid predictions: 3



Overall results:

Accuracy = 0.6229

Precision\_fake = 0.6879

Recall\_fake = 0.4540

F1\_fake = 0.5470

MCC = 0.2622



Confusion Matrix \[\[TN, FP], \[FN, TP]]:

\[\[394, 103],

&#x20;\[273, 227]]



Prediction JSON:

D:\\HaitongNi\\MEngProject\\repos\\VidGuard-R1\\results\\genvidbench\_common1000\\vidguard\_genvidbench\_common1000\_qwen25vl7b.json



Per-generator breakdown:

Note: VidGuard-R1-style evaluation was performed using Qwen2.5-VL-7B-Instruct with multiple-choice prompting. The model outputs A for fake and B for real. Among 1000 samples, 997 predictions were valid and 3 were invalid. The per-generator results below use valid predictions only.



Metric definitions:

fake\_recall = proportion of fake videos from this generator classified as fake

accuracy\_vs\_all\_real = accuracy using this generator's fake samples vs all valid real samples

precision\_fake\_vs\_all\_real = fake precision using this generator's fake samples vs all valid real samples

recall\_fake\_vs\_all\_real = same as fake\_recall

f1\_fake\_vs\_all\_real = fake F1 using this generator's fake samples vs all valid real samples

mcc\_vs\_all\_real = MCC using this generator's fake samples vs all valid real samples



Harder generators for VidGuard:

svd       n=62  recall=0.0968  acc\_vs\_real=0.7156  precision=0.0550  F1=0.0702  MCC=-0.0876  CM=\[\[394,103],\[56,6]]

musev     n=63  recall=0.1111  acc\_vs\_real=0.7161  precision=0.0636  F1=0.0809  MCC=-0.0765  CM=\[\[394,103],\[56,7]]

mora      n=63  recall=0.3810  acc\_vs\_real=0.7464  precision=0.1890  F1=0.2526  MCC=0.1311  CM=\[\[394,103],\[39,24]]



Moderate generators:

ms        n=63  recall=0.5079  acc\_vs\_real=0.7607  precision=0.2370  F1=0.3232  MCC=0.2221  CM=\[\[394,103],\[31,32]]

t2vz      n=62  recall=0.5645  acc\_vs\_real=0.7674  precision=0.2536  F1=0.3500  MCC=0.2602  CM=\[\[394,103],\[27,35]]

cogvideo  n=63  recall=0.5714  acc\_vs\_real=0.7679  precision=0.2590  F1=0.3564  MCC=0.2664  CM=\[\[394,103],\[27,36]]



Generators with relatively high fake recall:

vc2       n=62  recall=0.6935  acc\_vs\_real=0.7818  precision=0.2945  F1=0.4135  MCC=0.3476  CM=\[\[394,103],\[19,43]]

pika      n=62  recall=0.7097  acc\_vs\_real=0.7835  precision=0.2993  F1=0.4211  MCC=0.3584  CM=\[\[394,103],\[18,44]]



Observation:

VidGuard-R1-style zero-shot evaluation shows strong generator-dependent behavior on GenVidBench. It performs poorly on svd and musev, with fake recall close to 10%, and performs relatively better on pika and vc2, with fake recall around 70%. However, even for the easier generators, fake precision remains low when compared against all real samples. This suggests that the VLM-based multiple-choice prompt can detect some generator-specific fake cues, but it is not a reliable standalone fake-video detector.



Compared with NSG-VD, VidGuard achieves higher overall fake recall, but it also produces more real false positives. Compared with WaveRep, VidGuard is much weaker and less stable across generators.





Method: ReStraV

Implementation: DINOv2 temporal geometry feature extraction + MLP training

Dataset: GenVidBench

Subset: common-1000

Evaluation setting: trained-on-subset

Valid samples: 1000

Train samples: 500

Test samples: 500

Positive class: fake = 1, real = 0



Input CSV:

D:\\HaitongNi\\MEngProject\_SecondBench\\metadata\\genvidbench\_common1000.csv



Feature:

21-D temporal geometry vector

NUM\_FRAMES = 24

WINDOW\_SEC = 2.0



Training:

MLP hidden sizes: 64, 32

Epochs = 20

Learning rate = 0.001

Train/test split = stratified 50/50

Normalization = train-set mean/std only

Best threshold selected on train set = 0.6450



Threshold-free metrics on test set:

ROC-AUC\_fake = 0.9876

AP\_fake = 0.9897



Threshold-based metrics on test set:

Accuracy = 0.9500

Precision\_fake = 0.9668

Recall\_fake = 0.9320

F1\_fake = 0.9491

MCC = 0.9006



Confusion Matrix \[\[TN, FP], \[FN, TP]]:

\[\[242, 8],

&#x20;\[17, 233]]



Prediction CSV:

D:\\HaitongNi\\MEngProject\\repos\\ReStraV\\results\\genvidbench\_common1000\\restrav\_genvidbench\_common1000\_predictions.csv



Important note:

ReStraV requires training a lightweight classifier on extracted temporal geometry features. Therefore, this result is a trained-on-subset evaluation rather than pure pretrained inference.





Method: MM-Det

Implementation: official customized-dataset pipeline with saved sample-level predictions

Dataset: GenVidBench

Subset: common-1000

Frame setting: cap64

Samples: 500 real + 500 fake

Checkpoint: weights\\MM-Det\\current\_model.pth

Cache MM: True



Important preprocessing note:

Because GenVidBench real videos were much longer than fake videos, full-frame reconstruction caused severe frame-count imbalance. I therefore used a capped-frame setting: up to 64 uniformly sampled frames per video. Videos shorter than the model window size of 10 frames were padded by repeating the last available frame.



Score note:

Used original score as fake score.



Valid samples: 1000

Threshold: 0.5



Threshold-free metrics:

ROC-AUC\_fake = 0.582280

AP\_fake = 0.548602



Threshold-based metrics @ score\_fake >= 0.5:

Accuracy = 0.523000

Precision\_fake = 0.512118

Recall\_fake = 0.972000

F1\_fake = 0.670807

MCC = 0.104547



Confusion Matrix \[\[TN, FP], \[FN, TP]]:

\[\[37, 463],

&#x20;\[14, 486]]



Prediction CSV:

D:\\HaitongNi\\MEngProject\\repos\\MM-Det\\results\\genvidbench\_common1000\\mmdet\_genvidbench\_common1000\_cap64\_predictions.csv



Generator breakdown CSV:

D:\\HaitongNi\\MEngProject\\repos\\MM-Det\\results\\genvidbench\_common1000\\mmdet\_genvidbench\_common1000\_cap64\_generator\_breakdown.csv



Per-generator breakdown:

Note: MM-Det was evaluated on GenVidBench common-1000 using the cap64 setting. The generator metadata was re-merged with GenVidBench metadata after prediction, because the original saved prediction script retained AIGVDBench metadata fields.



Preprocessing note:

Up to 64 uniformly sampled frames per video were used. Videos shorter than the model window size of 10 frames were padded by repeating the last available frame.



Metric definitions:

fake\_recall\_at\_0.5 = proportion of fake videos from this generator classified as fake at threshold 0.5

mean\_score\_fake = average MM-Det fake score for this generator

auc\_vs\_all\_real = AUC using this generator's fake samples vs all 500 real samples

ap\_vs\_all\_real = AP using this generator's fake samples vs all 500 real samples



Per-generator results:

t2vz      n=62  recall=0.9032  mean\_score=0.9062  AUC\_vs\_real=0.4260  AP\_vs\_real=0.0937

mora      n=63  recall=0.9524  mean\_score=0.9579  AUC\_vs\_real=0.5573  AP\_vs\_real=0.1253

cogvideo  n=63  recall=0.9683  mean\_score=0.9681  AUC\_vs\_real=0.5624  AP\_vs\_real=0.1280

musev     n=63  recall=0.9683  mean\_score=0.9659  AUC\_vs\_real=0.6021  AP\_vs\_real=0.1393

svd       n=62  recall=0.9839  mean\_score=0.9851  AUC\_vs\_real=0.6187  AP\_vs\_real=0.1420

ms        n=63  recall=1.0000  mean\_score=0.9921  AUC\_vs\_real=0.6111  AP\_vs\_real=0.1417

pika      n=62  recall=1.0000  mean\_score=0.9999  AUC\_vs\_real=0.6511  AP\_vs\_real=0.1518

vc2       n=62  recall=1.0000  mean\_score=0.9999  AUC\_vs\_real=0.6295  AP\_vs\_real=0.1447



Observation:

MM-Det shows very high fake recall across all GenVidBench fake generators, with every generator above 90% recall and several reaching 100%. However, this does not mean strong detection performance, because the overall confusion matrix shows that MM-Det also misclassifies most real samples as fake. The low AUC/AP values indicate weak ranking ability despite high fake recall. In other words, MM-Det is strongly biased toward predicting fake on this benchmark.





