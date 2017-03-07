/* raz_nn -- razbit's neural network */
/* network.h -- a class describing a neural network */

#ifndef NETWORK_H
#define NETWORK_H


#include <vector>

#include "neuron.h"

class NNetwork
{
private:
	std::vector<Neuron*> inputs;
	std::vector<Neuron*> hiddens;
	std::vector<Neuron*> outputs;

	double rnd_weight();
	
public:
	NNetwork(int n_inputs, int n_hiddens, int n_outputs);
	~NNetwork();

	void feed_forward(double* pattern, int* result = NULL);
};


#endif /* NETWORK_H */
