/* raz_nn -- razbit's neural network */
/* neuron.cpp -- a class describing a single neuron */

#include <vector>
#include <utility>

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
	n_inputs++;
}

void Neuron::disconnect(Neuron* neuron)
{
	/* find the pair */
	for (int i = 0; i < n_inputs; i++)
	{
		if (inputs.at(i).first == neuron)
		{
			inputs.erase(inputs.begin()+i);
			n_inputs--;
			break;
		}
	}
}

void Neuron::set_value(double val)
{
	value = val;
}

void Neuron::set_weight(int index, double weight)
{
	weights[index] = weight;
}

void Neuron::set_inputs(int n)
{
	n_inputs = n;
}

double Neuron::get_value()
{
	return value;
}

double Neuron::get_weight(int index)
{
	return weights[index];
}

double* Neuron::get_weights()
{
	return weights;
}
