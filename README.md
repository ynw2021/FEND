# FEND
Code for CVPR 2023 'FEND: A Future Enhanced Distribution-Aware Contrastive Learning
Framework for Long-tail Trajectory Prediction'

### Requirements
We use the same requirements as the Trajectron++, see:
https://github.com/StanfordASL/Trajectron-plus-plus


### Data

#### nuScenes (Bird's-eye view):
For the processed files, you can run the processing script of nuScenes at:

```
python Trajectron_plus_plus/experiments/nuScenes/process_data.py
```

### Pre-trained Models
Pretrained models are provided under ```models/```. 

### Testing
Example to test our model on nuScenes dataset:

```
python test_nuscenes_fend.py --model models/nuScenes_model/trajectron_map_int_fend_ewta_withoutfrequency --checkpoint 25 --data data/nuScenes_test_full.pkl --kalman kalman/nuScenes_VEHICLE_test_ewta_baseline_withoutfrequency.pkl --node_type VEHICLE
```


Example to test the baseline traj++ewta model on nuScenes dataset:

```
python test_nuscenes_fend.py --model models/nuScenes_model/trajectron_map_int_fend_ewta_withoutfrequency --checkpoint 25 --data data/nuScenes_test_full.pkl --kalman kalman/nuScenes_VEHICLE_test_ewta_baseline_withoutfrequency.pkl --node_type VEHICLE
```


We use the Traj++EWTA without resampling as the baseline model to select the hard samples, as our method is implemented on it. Testing with the Traj++EWTA without resampling as the baseline model to select hard samples can be done as follows:

```
python test_nuscenes_fend.py --model models/nuScenes_model/trajectron_map_int_fend_ewta_withoutfrequency --checkpoint 25 --data data/nuScenes_test_full.pkl --kalman kalman/nuScenes_VEHICLE_test_ewta_baseline_withfrequency.pkl --node_type VEHICLE
```


### Training
Example to train our model on nuScenes dataset:

```
python Trajectron_plus_plus/trajectron/train_nuscenes_fend.py
```

### Citation
If you use our repository or find it useful in your research, please cite the following paper:


<pre class='bibtex'>
@inproceedings{wang2023fend,
  title={Fend: A future enhanced distribution-aware contrastive learning framework for long-tail trajectory prediction},
  author={Wang, Yuning and Zhang, Pu and Bai, Lei and Xue, Jianru},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={1400--1409},
  year={2023}
}
</pre>
