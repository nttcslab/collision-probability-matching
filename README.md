# Description

Note that this code only performs training under the proposed CPM loss (in addition to the cross-entropy loss) and excludes the calculation of the true collision probability, i.e., the test-retest reliability, and the final epsilon value as the true collision probability minus the resulting CPM-bound predictive collision probability.
This is a sample implementation of the collision probability matching (CPM), including:

- Training code 
- Sample dataset for running the code


## Paper
Our AISTATS 2023 paper can be viewed from http://aistats.org/aistats2023/  
The url will be updated after the paper is published on AISTATS 2023 proceedings.


To cite this paper:
```BibTex
@inproceedings{narimatsu2023collision,
  title={Collision Probability Matching Loss for Disentangling Epistemic Uncertainty from Aleatoric Uncertainty},
  author={Hiromi Narimatsu and Mayuko Ozawa and Shiro Kumano},
  booktitle={The 26th International Conference on Artificial Intelligence and Statistics (AISTATS2023) (to appear)},
  year={2023}.
}
```

## How to run
### Model training
Use `train.py` to run the code.
`python train.py 
To change the following parameters, you can use these arguments:
* `--inputs data/labels.csv`
* `--features data/features.csv` (It is used as features.)
* `--weight 100000`
* `--seed 123`
* `--output output_dir`
* `--num_epochs 100`

### Output results
This code output the measures as the following format:
`train | count:1 | epoch:001/100 | lr:2.000000e-04 | accuracy:0.0925 | ccp:0.1967 | cp:0.2006 | diff:1.964208e-05 | conf:0.2164 | cost:3.5913 | cost1:1.9642 | cost2:1.6271`  

* lr: learning rate
* accuracy: machine accuracy
* ccp: cross collision probability
* cp: predictive collision probability
* diff: squared error of cp - ccp 
* conf: machine confidence 
* cost: final loss 
* cost1: cpm loss
* cost2: cross entropy loss

If ccp=cp is not satisfied with at least two significant digits, please increase the weight (e.g., by 10 times). Their discrepancy tends to lead to a large error in the estimated epsilon.


## Dataset examples
Sample data is included in the data directory. (We will also upload the sample data soon.)
* `labels.csv: This file includes a set of synthesized labels of 50 virtual respondents to 50 images on a 5-point scale. Note that this is different from the dataset used in our AISTATS 2023 paper.`  
  Data format:
  `[image_id, feat1, feat2, feat3, ..., featN]`

* `features.csv: This file includes a set of features of the 50 images. They are the action units obtained using OpenFace [1] for each image.`  
  Data format:  
 `[subject_id, image_id, V1, V2, V3, V4, V5, response]`



## Verified software versions?
This code was tested on the following libraries.

* `Python`: `3.9.12`
* `torch`: `1.12.1`


## Files
* `README.md`: This file. 
* `LICENSE.md`: Document of agreement for using this sample code. Read this carefully before using the code. 
* `train.py`: Sample code of collision probability matching


## History
- Release Code (24th Feb, 2023) 


## Licenses

See [LICENSE](LICENSE) for details.


## Reference
[1] Baltrusaitis, T., Zadeh, A., Lim, Y. C., and Morency, L. (2018). Openface 2.0: Facial behavior analysis toolkit. In 2018 13th IEEE International Conference on Automatic Face Gesture Recognition (FG 2018), pages 59–66.


