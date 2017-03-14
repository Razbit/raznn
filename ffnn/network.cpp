/* raz_nn -- razbit's neural network */
/* network.cpp -- a class describing a neural network */

#include <vector>
#include <stdlib.h>

#include "neuron.h"
#include "network.h"

#include "logger.h"

NNetwork::NNetwork(int n_inputs, int n_hiddens, int n_outputs, Logger* log)
{
	this->log = log;

	log->write(1, "Setting up the network..\n");
	
	/* generate and set up the neural network */
	/* input layer */
	for (int i = 0; i <= n_inputs; i++)
	{
		inputs.push_back(new Neuron(log));
	}

	/* input bias */
	inputs.back()->set_value(-1.0);

	log->write(2, "Input layer created with %d neurons\n", n_inputs);

	/* hidden layer */
	for (int i = 0; i < n_hiddens; i++)
	{
		hiddens.push_back(new Neuron(log));

		/* connect to input layer */
		for (int j = 0; j <= n_inputs; j++)
			hiddens.back()->connect(inputs.at(j), rnd_weight());
	}

	/* hidden bias */
	hiddens.push_back(new Neuron(log));
	hiddens.back()->set_value(-1.0);

	log->write(2, "Hidden layer created with %d neurons\n", n_hiddens);

	/* output layer */
	for (int i = 0; i < n_outputs; i++)
	{
		Neuron* tmp = new Neuron(log);
		outputs.push_back(tmp);

		/* connect to hidden layer */
		for (int j = 0; j <= n_hiddens; j++)
			outputs.back()->connect(hiddens.at(j), rnd_weight());
	}

	log->write(2, "Output layer created with %d neurons\n", n_outputs);
}

NNetwork::~NNetwork()
{
	
}

/* Feed a pattern through the network */
void NNetwork::feed_forward(double* pattern, int* result /* = NULL */)
{
	log->write(2, "Feeding pattern ");
	/* set input values */
	for (int i = 0; i < inputs.size() - 1; i++)
	{
		inputs.at(i)->set_value(pattern[i]);
		log->write(2, "%d ", pattern[i]);
	}

	log->write(3, "\n Calculating hidden layer\n");

	/* calculate hidden layer */
	for (int i = 0; i < hiddens.size() - 1; i++)
	{
		hiddens.at(i)->activation();
	}

	log->write(3, "Calculating output layer\n");

	/* calculate output layer */
	for (int i = 0; i < outputs.size(); i++)
	{
		hiddens.at(i)->activation();
	}

	if (result)
	{
		for (int i = 0; i < outputs.size(); i++)
			result[i] = outputs.at(i)->clamp_neuron();
	}

	log->write(2, "Done.\n");
}

double NNetwork::rnd_weight()
{
	double tmp = (double)(rand() - RAND_MAX / 2);
	return tmp * (NN_RND_W_RANGE / RAND_MAX);
}
