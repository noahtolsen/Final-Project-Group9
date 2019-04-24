#!/bin/bash

#Make sure you have a working version of the Kaggle API on your computer
#if needed you can install it with pip install kaggle and then make a folder titled .kaggle in your home directory containing your API key
kaggle datasets download -d grassknoted/asl-alphabet
unzip asl-alphabet.zip

unzip asl_alphabet_test.zip
unzip asl_alphabet_train.zip
mkdir Data
mv asl_alphabet_train Data
mv asl_alphabet_test Data

for i in {a..z} 
do
	mkdir Data/asl_alphabet_test/$i
	mv "Data/asl_alphabet_test/${i}_test.jpg" Data/asl_alphabet_test/$i
done


mkdir Data/asl_alphabet_test/nothing
mv Data/asl_alphabet_test/nothing_test.jpg Data/asl_alphabet_test/nothing

mkdir Data/asl_alphabet_test/space
mv Data/asl_alphabet_test/space_test.jpg Data/asl_alphabet_test/space
