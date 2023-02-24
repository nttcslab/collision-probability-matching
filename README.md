# cpm_test

This is a sample implementation of the collision probability matching (CPM), includes:

- Training code 
- Sample dataset for run the code


## Paper
Paper is here: http://aistats.org/aistats2023/  
We will update the document url after publish the following AISTATS 2023 proceedings.

Cite the following article to refer to this work.
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
* `--inputs data/sampledata_sample50img.csv`
* `--action_units data/stim_list_withAUs.csv` (It is used as features.)
* `--weight 100000`
* `--seed 123`
* `--output output_dir`
* `--num_epochs 100`


## Dataset Examples
Sample data is included in the data directory. (We will also upload the sample data soon.)
* `stim_list_withAUs.csv`: This file is used as a feature. This sample contains the Action Unit information obtained using OpenFace [1] by inputting each image.  
  Data format:
  `[feature_id, feat1, feat2, feat3, ..., featN]`
* `sampledata_sample50img.csv`: This is a sample of the input data created to run this sample code. It simulates the score that a respondent gives to a target image on a 5-point scale. 50 images with 50 resopondents are included in the dataset. (Note that this data is different from the one used in the experiment of our paper.)  
  Data format:  
 `[subject_id, feature_id, V1, V2, V3, V4, V5, response]`

 The two files can be mapped by feature_id. 


## Software version
Codes are confirmed to run with the following libraries. Likely to be compatible with newer versions. 

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


