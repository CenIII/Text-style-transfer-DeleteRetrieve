# Text style transfer - DeleteRetrieve
The task is sentiment transfer of text using non-parallel data. Our model first delete the phrases in a sentence with the most of style information, and use the broken sentence together with trained target style embedding to reconstruct the transferred sentence.

## Getting Started

These instructions will provide information about how to use our code.

### Prerequisites

Several packages are needed to run the code. To install them, run
```
pip install -r requirement.txt
```
Or install whatever packages that are missing manually, for example
```
pip install torch
```

### Installing

The checkpoint of trained model are needed for evaluating and online transferring. 
You can download [checkpoint](https://drive.google.com/file/d/1dCYjVoylK4BOgiKlGDolpJEU2TQq3LtK/view?usp=sharing) here. After downloading,

```
unzip exp.zip
mv exp path_of_this_git/RNNS
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

