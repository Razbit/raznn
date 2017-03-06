/* raz_nn -- razbit's neural network */
/* neuron.h -- a class describing a single neuron */

#include <vector>
#include <utility>

class Neuron
{
private:
	double value;
	int n_inputs;
	std::vector<std::pair<Neuron*, double> > inputs;

public:
	Neuron();
	~Neuron();

	void connect(Neuron* neuron, double weight);
	void disconnect(Neuron* neuron);

	void set_value(double val);
	void set_weight(Neuron* neuron, double weight);
	double get_value();
	double get_weight(Neuron* neuron);
	double* get_weights();

	inline double activation();
	inline int clamp()
};
