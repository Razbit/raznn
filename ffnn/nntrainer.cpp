/* raz_nn -- razbit's neural network */
/* nntrainer.h -- a class for training the network */

#include <vector.h>

#include "neuron.h"
#include "network.h"
#include "nntrainer.h"

NNTrainer::NNTrainer(NNetwork* netw, std::vector<double*> t_set, double rate);
{
	network = netw;
	trainingset = t_set;
	start_outvals = network->inputs.size() - 1;
	set_index = 0;
	l_rate = rate;
}

NNTrainer::~NNTrainer()
{
	
}

void NNTrainer::calc_e_grads()
{
	/* desired outputs are at the end of each 'row' of the vector */

	/* error gradients of output neurons */
	o_e_grads.resize(network->outputs.size());
	for (int i = 0; i < network->outputs.size(); i++)
	{
		double val = network->outputs.at(i)->get_value();
		double desired = trainingset.at(set_index)[i+start_outvals];
		o_e_grads.at(i) = val * (1 - val) * (desired - val);
	}

	/* error gradients of hidden neurons */
	h_e_grads.resize(network->hiddens.size());
	for (int i = 0; i < network->hiddens.size(); i++)
	{
		double val = network->hiddens.at(i)->get_value();
		double sum = 0.0;
		for (int j = 0; j < network->outputs.size(); j++)
		{
			double weight = network->outputs.at(j)->inputs.at(i).second;
			sum += o_e_grads.at(j) * weight;
		}
		h_e_grads.at(i) = val * (1 - val) * sum;
	}
}

void NNTrainer::calc_deltas()
{
	/* delta weights of input->hidden */
	i_h_dweights.resize(network->hiddens.size());
	for(int i = 0; i < network->hiddens.size(); i++)
	{
		double e_grad = o_e_grads.at(i);
		i_h_dweights.at(i).resize(network->inputs.size());
		for (int j = 0; j < network->inputs.size(); j++)
		{
			i_h_dweights.at(i).at(j) = l_rate * network->inputs.at(j)->get_value() * h_e_grads.at(i);
		}
	}
}
