/* raz_nn -- razbit's neural network */
/* nntrainer.h -- a class for training the network */

#ifndef NNTRAINER_H
#define NNTRAINER_H

#include <vector.h>

#include "neuron.h"
#include "network.h"

class NNTrainer
{
private:
	std::vector<double> o_e_grads;
	std::vector<double> h_e_grads;
	std::vector<std::vector<double> > i_h_dweights;
	std::vector<std::vector<double> > h_o_dweights;

	NNetwork* network;

	std::vector<double*> trainingset;
	size_t set_index;
	size_t start_outvals;

	double l_rate;

	void calc_e_grads();
	void calc_deltas();

public:
	NNTrainer(NNetwork* netw, std::vector<double*> t_set, double rate);
	~NNTrainer();

	
};


#endif /* NNTRAINER_H */
