
! This is a personal project and not meant to be used in any actual capacity !
I don't mind anyone using it, I would just advise against it

# Compiling

Currently Requires `nvcc` even when only using CPU networks

```
cmake -B build .

cmake --build build --target=install
```

Connections can be used by adding the following to your project's `CMakeLists.txt`

```CMake
find_package(cntns REQUIRED)

# if the project should use the GPU
cntns_use_cuda()

target_link_libraries(
  project_name
  cntns
)
```

# Creating a Neural Network

## create_network

```C++
auto network = cntns::create_network<*ARENA*>(
  *Cost Function*,
  *Layers*
);
```

- *Arena* : either ArenaType::CPU or ArenaType::GPU
- *Cost Function* : currently `MSE`, or `CrossEntropy`
- *Layers* : any number of cntns::Layer<>

## Layers

```C++
auto layer = cntns::Layer<*input size*, *output size*, *activation function*, *arena*>{};
```

- *Input size* : The number of nodes in the previous layer
- *Output size* : The number of nodes in the next layer
- *Activation Func* : currently `Sigmoid`, `ReLu`, or `SoftMax`



### Example

```C++
auto cpuNetwork = cntns::create_network<ArenaType::CPU>(
    CrossEntropy{},
    Layer<784, 40, Sigmoid, ArenaType::CPU>{},
    Layer<40, 10, SoftMax, ArenaType::CPU>{}
  );
```

# Example, Classifying MNIST

```C++
#include "cntns.hpp"

int main() {

  using namespace cntns;

  // Create the network
  auto network = create_network<ArenaType::CPU>(
    CrossEntropy{},
    Layer<784, 40, Sigmoid, ArenaType::CPU>{},
    Layer<40, 10, SoftMax, ArenaType::CPU>{}
  );

  auto trainingConfig = TrainingConfig{
    .epochs = 10,
    .batchSize = 100,
    .learningRate = 0.1
  };

  auto trainingData = TrainingData{
    load_mnist_images("trainimages"),  // loads mnist data into a vector of Vecs784>, method not provided
    load_mnist_labels("trainlabels")   // loads mnist data into a vector of Vec<10>, method not provided
  };

  // Train network, in this case on the CPU
  train_network<ArenaType::CPU>(network, traningConfig, trainingData);

  auto testingData = TestingData{
    load_mnist_images("testimages"),  // loads mnist data into a vector of Vec<784>, method not provided
    load_mnist_labels("testlabels")   // loads mnist data into a vector of Vec<10>, method not provided
  };

  double accuracy = test_network<ArenaType::CPU>(network, testingData);
  std::cout << "Accuracy: " << std::to_string(accuracy) << '\n';
}
```
