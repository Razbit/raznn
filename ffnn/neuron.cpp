/* raz_nn -- razbit's neural network */
/* neuron.cpp -- a class describing a single neuron */

#include <vector>
#include <utility>
#include <math.h>

/* constructor */
Neuron::Neuron()
{
	value = 0.0;
}

/* destructor */
Neuron::~Neuron()
{
	
}

void Neuron::connect(Neuron* neuron, double weight)
{
	inputs.push_back(std::make_pair(neuron, weight));
}

void Neuron::disconnect(Neuron* neuron)
{
	/* find the pair */
	for (int i = 0; i < n_inputs; i++)
	{
		if (inputs.at(i).first == neuron)
		{
			inputs.erase(inputs.begin()+i);
			break;
		}
	}
}

void Neuron::set_value(double val)
{
	value = val;
}

void Neuron::set_weight(Neuron* neuron, double weight)
{
	for (int i = 0; i < inputs.size(); i++)
	{
		if (inputs.at(i).first == neuron)
		{
			inputs.at(i).second = weight;
			break;
		}
	}
}

void Neuron::set_weight_i(int index, double weight)
{
	inputs.at(index).second = weight;
}

double Neuron::get_value()
{
	return value;
}

double Neuron::get_weight(Neuron* neuron)
{
	for (int i = 0; i < inputs.size(); i++)
	{
		if (inputs.at(i).first == neuron)
		{
			return inputs.at(i).second;
		}
	}
}

double* Neuron::get_weight_i(int index)
{
	return inputs.at(index).second;
}

void Neuron::activation()
{
	/* calculate weighted sum of the inputs */
	double sum = 0.0;
	
	for (int i = 0; i < inputs.size(); i++)
		sum += inputs.at(i).second * inputs.at(i).first->get_value();

	/* activate using sigmoid function */
	value = 1/(1+exp(-sum));
}

inline int Neuron::clamp()
{
	if (value > 0.9)
		return 1;
	else if (value < 0.1)
		return 0;
	else
		return -1;
}
