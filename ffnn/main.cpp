/* raz_nn -- razbit's neural network */

#include "network.h"
#include "nntrainer.h"
#include "logger.h"
#include "loggerfile.h"
#include "loggerconsole.h"

#include <vector>

int main()
{
	//std::vector<double*> set;
	
	Logger* logger = new LoggerConsole();
	logger->set_level(5);
	NNetwork* network = new NNetwork(3, 2, 1, logger);
//	NNTrainer* tranger = new NNTrainer(network, set, 100, logger);
	
}
