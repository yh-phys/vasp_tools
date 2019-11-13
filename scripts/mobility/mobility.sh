#!/bin/bash
#13 November, 2019
# Firstly, you should prepare the input files for vasp calculation, 
# the details can view my blog: https://yh-phys.github.io/

#To use it: bash mobility.sh
mkdir mobility-x
cd mobility-x
x=4.083622259999999                           #"x" stands for the lattice constant in x direction
for i in $(seq 0.98 0.005 1.02)                #"i" defines the range of strain
do
cp -r ../IS-x ./$i                               #"IS-x" stands for the origin file 
sed -i "3s/$x/$(echo "$x*$i"|bc)/g" $i/POSCAR
cd $i
qsub ./pbs
cd $OLDPWD
done
cd ../
mkdir mobility-y
cd mobility-y
y=7.073041233239241                            #"y" stands for the lattice constant in y direction
for j in $(seq 0.98 0.005 1.02)                #"j" defines the range of strain
do
cp -r ../IS-y ./$j                               #"IS-y" stands for the origin file 
sed -i "4s/$y/$(echo "$y*$j"|bc)/g" $j/POSCAR
cd $j
qsub ./pbs
cd $OLDPWD
done
