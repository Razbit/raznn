/* raz_nn -- razbit's neural network */
/* nntrainer.h -- a class for training the network */

#include <vector>
#include <math.h>
#include <stdlib.h>

#include "neuron.h"
#include "network.h"
#include "nntrainer.h"

#include "logger.h"

NNTrainer::NNTrainer(NNetwork* netw, std::vector<double*> t_set, size_t g_set, Logger* log)
{
	network = netw;
	trainingset = t_set;
	g_set_start = g_set;
	start_outvals = network->inputs.size() - 1;
	set_index = 0;
	l_rate = 0.001;
	this->log = log;
}

NNTrainer::~NNTrainer()
{
	
}

void NNTrainer::calc_e_grads()
{
	/* desired outputs are at the end of each 'row' of the vector */

	log->write(2, "Calculating error gradients\n");
	
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
	log->write(2, "Calculating delta weights\n");

	/* delta weights of input->hidden */
	i_h_dweights.resize(network->hiddens.size());
	for(int i = 0; i < network->hiddens.size(); i++)
	{
		i_h_dweights.at(i).resize(network->inputs.size());
		for (int j = 0; j < network->inputs.size(); j++)
		{
			i_h_dweights.at(i).at(j) = l_rate * network->inputs.at(j)->get_value() * h_e_grads.at(i);
		}
	}

	/* delta weights of hidden->output */
	h_o_dweights.resize(network->outputs.size());
	for (int i = 0; i < network->outputs.size(); i++)
	{
		h_o_dweights.at(i).resize(network->hiddens.size());

		for (int j = 0; j < network->hiddens.size(); j++)
		{
			h_o_dweights.at(i).at(j) = l_rate * network->hiddens.at(j)->get_value() * o_e_grads.at(i);
		}
	}
}

void NNTrainer::update_weights()
{
	log->write(2, "Updating weights\n");

	for (int i = 0; i < network->outputs.size(); i++)
	{
		for (int j = 0; j < network->outputs.at(i)->inputs.size(); j++)
		{
			network->outputs.at(i)->inputs.at(j).second += \
				h_o_dweights.at(i).at(j);
		}
	}

	for (int i = 0; i < network->hiddens.size(); i++)
	{
		for (int j = 0; j < network->hiddens.at(i)->inputs.size(); j++)
		{
			network->hiddens.at(i)->inputs.at(j).second += \
				i_h_dweights.at(i).at(j);
		}
	}
}

int clamp(double n)
{
	if (n < 0.1)
		return 0;
	else if (n > 0.9)
		return 1;
	else
		return -1;
}

void NNTrainer::run_epoch(size_t size, bool test)
{
	/* feed, calculate errors and deltas, update weights */

	double incorrect = 0.0;
	double mse = 0.0;
	int* res = new int[network->outputs.size()];
	
	for (int i = 0; i < size; i++)
	{
		log->write(2, "Feeding set %d\n", i);
		network->feed_forward(trainingset.at(set_index), res);

		if (!test)
		{
			calc_e_grads();
			calc_deltas();
			update_weights();	
		}

		bool correct = true;
		for (int j = 0; j < network->outputs.size(); j++)
		{
			double target = trainingset.at(set_index)[start_outvals+j];
			if (res[j] != clamp(target))
				correct = false;

			mse += pow(network->outputs.at(j)->get_value() - trainingset.at(set_index)[start_outvals+j], 2);
		}

		if (!correct)
			incorrect++;

		set_index++;
	}

	accuracy = 100 - (incorrect / size * 100);
	mse = mse / (network->outputs.size() * size);
	epoch++;
}

void NNTrainer::train(double rate, double epochs, double accuracy)
{
	l_rate = rate;

	size_t epoch_size = g_set_start/epochs;

	/* train */
	while(this->accuracy < accuracy || this->epoch == epochs)
	{
		log->write(1, "Epoch %d of %d..\n", epoch, epochs);
		run_epoch(epoch_size, false);
		log->write(1, "Epoch over, MSE: %f Acc%%: %f\n", this->mse, this->accuracy);
	}

	/* test using the generalization set */

	log->write(1, "Running generalizaton set..\n");
	run_epoch(trainingset.size()-set_index, true);
	log->write(1, "Generalization set done, MSE: %f Acc%%: %f\n", this->mse, this->accuracy);
}
