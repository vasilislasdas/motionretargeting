# Motionretargeting

Performs motion retargeting between mocap characters. The characters can have the same number of joints(intra-structure or intra-class retargeting) or different number of joints(cross-structure retargeting). The deployed networks are based on the wasserstein-gan architecture, where both the encoder and the discriminator are built using transformers. The code has been tested on Ubuntu 20.04 LTS using python 3.8 and pytorch 1.9.0 with cuda 10.2.


## Getting Started

In the next sections you can find instructions on how to:
- Download the training and the test set
- Preprocess the training set as a preparation step for training 
- Training the retargeter network
- Retarget between various characters


### Prerequisites

Download the training and the test and unzip them in the dataset folder.
- [Training set](https://drive.google.com/file/d/1UhXkFfOVaDUdpCmIGkScOoi1IG7c3olH/view?usp=sharing)
- [Test set](https://drive.google.com/file/d/1nc4MRO-QTjqqr96vuhAr4P6HdZdBM3W6/view?usp=sharing)

### Installing

A step by step series of examples that tell you how to get a development
environment running

Say what the step will be

    Give the example

And repeat

    until finished

End with an example of getting some data out of the system or using it
for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Sample Tests

Explain what these tests test and why

    Give an example

### Style test

Checks if the best practices and the right coding style has been used.

    Give an example

## Deployment

Add additional notes to deploy this on a live system

## Built With

  - [Contributor Covenant](https://www.contributor-covenant.org/) - Used
    for the Code of Conduct
  - [Creative Commons](https://creativecommons.org/) - Used to choose
    the license

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code
of conduct, and the process for submitting pull requests to us.

## Versioning

We use [Semantic Versioning](http://semver.org/) for versioning. For the versions
available, see the [tags on this
repository](https://github.com/PurpleBooth/a-good-readme-template/tags).

## Authors

  - **Billie Thompson** - *Provided README Template* -
    [PurpleBooth](https://github.com/PurpleBooth)

See also the list of
[contributors](https://github.com/PurpleBooth/a-good-readme-template/contributors)
who participated in this project.

## License

This project is licensed under the [CC0 1.0 Universal](LICENSE.md)
Creative Commons License - see the [LICENSE.md](LICENSE.md) file for
details

## Acknowledgments

  - Hat tip to anyone whose code is used
  - Inspiration
  - etc

