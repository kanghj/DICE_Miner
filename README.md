# DICE_Miner
This is the DICE specification miner


In ~/\<user\>/DICE_files, git clone the following repositories:

```
git clone https://github.com/kanghj/deep_spec_learning_ws
git clone https://github.com/kanghj/DICE_dummy_code
git clone https://github.com/kanghj/DICE_Tester
git clone https://github.com/kanghj/DICE_Miner
git clone https://github.com/kanghj/fsa_model_ground_truths

```
Then run a docker container (running the openjdk8 image):
```
docker run -v ${PWD}:/workspace  -it --name DICE adoptopenjdk/openjdk8 /bin/bash
```


From the docker container, run the following commands:
```
cd /workspace
cp DICE_Miner/run.sh run.sh

sh run.sh
```
