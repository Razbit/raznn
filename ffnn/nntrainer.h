/* raz_nn -- razbit's neural network */
/* nntrainer.h -- a class for training the network */

#ifndef NNTRAINER_H
#define NNTRAINER_H

#include <vector>
#include <stdlib.h>

#include "neuron.h"
#include "network.h"

#include "logger.h"

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
	size_t g_set_start; // start of generalization set

	double l_rate;
	double mse;
	double accuracy;
	long epoch;

	Logger* log;

	void calc_e_grads();
	void calc_deltas();
	void run_epoch(size_t size, bool test);
	void update_weights();

public:
	NNTrainer(NNetwork* netw, std::vector<double*> t_set, size_t g_set, Logger* log);
	~NNTrainer();

	void train(double rate, double epochs, double accuracy);

};


#endif /* NNTRAINER_H */
