/* raz_nn -- razbit's neural network */
/* network.h -- a class describing a neural network */

#ifndef NETWORK_H
#define NETWORK_H


#include <vector>

#include "neuron.h"
#include "nntrainer.h"

#define NN_RND_W_RANGE 0.5

class NNetwork
{
	friend class NNTrainer;

private:
	std::vector<Neuron*> inputs;
	std::vector<Neuron*> hiddens;
	std::vector<Neuron*> outputs;

	/* in range [-NN_RND_W_RANGE, NN_RND_W_RANGE] */
	double rnd_weight();
	
public:
	NNetwork(int n_inputs, int n_hiddens, int n_outputs);
	~NNetwork();

	void feed_forward(double* pattern, int* result = NULL);
};


#endif /* NETWORK_H */
