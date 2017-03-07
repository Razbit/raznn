/* raz_nn -- razbit's neural network */
/* network.cpp -- a class describing a neural network */

#include <vector>

#include "neuron.h"
#include "network.h"

NNetwork::NNetwork(int n_inputs, int n_hiddens, int n_outputs)
{
	/* generate and set up the neural network */
	/* input layer */
	for (int i = 0; i <= n_inputs; i++)
	{
		inputs.push_back(new Neuron());
	}

	/* input bias */
	inputs.back()->set_value(-1.0);

	/* hidden layer */
	for (int i = 0; i < n_hiddens; i++)
	{
		hiddens.push_back(new Neuron());

		/* connect to input layer */
		for (int j = 0; j <= n_inputs; j++)
			hiddens.back()->connect(inputs.at(j), rnd_weight());
	}

	/* hidden bias */
	hiddens.push_back(new Neuron())
	hiddens.back()->set_value(-1.0);

	/* output layer */
	for (int i = 0; i < n_outputs; i++)
	{
		Neuron* tmp = new Neuron();
		outputs.push_back(tmp);

		/* connect to hidden layer */
		for (int j = 0; j <= n_hiddens; j++)
			outputs.back()->connect(hiddens.at(j), rnd_weight());
	}
}

NNetwork::~NNetwork()
{
	
}

/* Feed a pattern through the network */
void NNetwork::feed_forward(double* pattern, int* result = NULL)
{
	/* set input values */
	for (int i = 0; i < inputs.size() - 1; i++)
	{
		inputs.at(i)->set_value(pattern[i]);
	}

	/* calculate hidden layer */
	for (int i = 0; i < hiddens.size() - 1; i++)
	{
		hiddens.at(i)->activation();
	}

	/* calculate output layer */
	for (int i = 0; i < outputs.size(); i++)
	{
		hiddens.at(i)->activation();
	}

	if (result)
	{
		for (int i = 0; i < outputs.size(); i++)
			result[i] = outputs.at(i)->clamp();
	}
}
