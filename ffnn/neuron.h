/* raz_nn -- razbit's neural network */
/* neuron.h -- a class describing a single neuron */

#ifndef NEURON_H
#define NEURON_H


#include <vector>
#include <utility>

class Neuron
{
private:
	double value;
	/* Input neuron's addr and its weight */
	std::vector<std::pair<Neuron*, double> > inputs;

public:
	Neuron();
	~Neuron();

	void connect(Neuron* neuron, double weight);
	void disconnect(Neuron* neuron);

	void set_value(double val);
	void set_weight(Neuron* neuron, double weight);
	void set_weight_i(int index, double weight);
	double get_value();
	double get_weight(Neuron* neuron);
	double get_weight_i(int index);

	void activation();
	inline int clamp()
};


#endif /* NEURON_H */
