/* raz_nn -- razbit's neural network */

#include "network.h"
#include "nntrainer.h"
#include "logger.h"
#include "loggerfile.h"
#include "loggerconsole.h"
#include "csv_parser.h"

#include <vector>

int main()
{
	Logger* logger = new LoggerConsole();
	logger->set_level(5);
	NNetwork* network = new NNetwork(3, 2, 1, logger);

	std::vector<double* > set;
	logger->write(2, "Parsing training set CSV\n");
	set = parse_csv("test");
	
	NNTrainer* trainer = new NNTrainer(network, set, 12000, logger);
	logger->set_level(3);
	trainer->train(0.1, 100, 0.75);
}
