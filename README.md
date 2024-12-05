Create the correct folder file:
name folder "include" add the following:
    data.hpp
    data_handler.hpp
    common.hpp
    neuron.hpp
    layer.hpp
    network.hpp
name folder "src" add the following:
    data.cc
    data_handler.cc
    common.cc
    neuron.cc
    layer.cc
    network.cc
name folder "data" add the following:
    output_binary_file.data
    VOOD_labels.csv

compile command:

g++ -std=c++11 -g -O0 .\src\data.cc .\src\data_handler.cc .\src\common.cc .\src\layer.cc .\src\neuron.cc .\src\network.cc -I./include -o test.exe

Ensure that you change the paths of the files to your current directory
