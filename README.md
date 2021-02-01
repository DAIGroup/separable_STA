# Separable STA network implementation
This is an **implementation** of the **separable spatio-temporal attention** (STA) network described in
(Das et al. 2019).

This network consists of a spatio-temporal attention block trained using an 3-layer LSTM network that learnt from
skeletal information. This attention block modulates the convolutional _feature_ output _g_ of an I3D
network (see [`DAIGroup/i3d`](github.com/DAIGroup/i3d) repository), both spatially and temporally. This modulated
features _gt_ and _gs_ are then concatenated and feed to a classifier for the final one-hot output vector. Please see
the original paper (Das et al. 2019) for further details. The two required branches are provided in separate 
repositories:

* `DAIGroup/i3d`: for the I3D branch, [here](github.com/DAIGroup/i3d).
* `DAIGroup/LSTM_action_recognition`: for the LSTM branch, [here](github.com/DAIGroup/LSTM_action_recognition).

Please **checkout** these first, and then change the `code_path` variable of `sta_config.py` to the common directory
where **all three** projects will reside. 

Following their instructions, we provide the implementation we used to replicate their experiments, as well as for our
modified version with alternative data preprocessing.

## Description of files

* `sta_config.py` is a file that can be modified to change training behaviours, run cross-subject (CS) or cross-view
  (CV2) experiments. As well as to tune other parameters (GPUs used, etc.).
* `toyota_generator.py` is the data generator for each epoch. The provided class contains a flag that can be modified to
determine whether it is a training or a test data generator (`is_test` flag).
* `separable_sta.py` contains the implementation of the additional layers of the attention block.
* `sta_train.py` is the main file to run for training.
* `sta_evaluate.py` is the main file to run for testing/evaluation.

## References

* **(Das et al. 2019)** Das, S., Dai, R., Koperski, M., Minciullo, L., Garattoni, L., Bremond, F., & Francesca, G. (2019). Toyota smarthome: Real-world activities of daily living. In Proceedings of the IEEE International Conference on Computer Vision (pp. 833-842).
* **(Climent-Pérez et al. 2021, _accepted_)** Climent-Pérez, P., Florez-Revuelta, F. (2021). Improved action recognition with Separable spatio-temporalattention using alternative Skeletal and Video pre-processing, Sensors, _accepted_.
