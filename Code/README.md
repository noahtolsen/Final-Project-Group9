# Instruction to Run Our Project

Because of the extremely large size of our data (1.1gb zipped) we could not include the actual data in the git repository. As such we have written a bash script to retrieve and format the data for you.
The script acceses the Kaggle API downloads the data, and formats so it can be used with our code. As such please make sure you can access the Kaggle API from the command line. Once you have this part
set up, please run:

```bash
./data_download.sh

```

After letting this run you should have a Data folder with everything properly formatted. Next you should be able to run the three models. They are titled:

- our_model.py
- resnet.py
- vgg.py


All three files are written in python 3 and can be run with python3 <model>.py.


It is important to note than given the size of the data and the complexity of the task, even on a gpu these model take several hours to run.
