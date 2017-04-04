#include <vector>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "csv_parser.h"

std::vector<double* > parse_csv(const char* name)
{
	std::vector<double* > ret;
	
	FILE * fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;

    fp = fopen(name, "r");
    if (fp == NULL)
        return ret;

    while ((read = getline(&line, &len, fp)) != -1) {
	    char* token = strtok(line, ",");
	    int n = read/2;
	    ret.push_back(new double[n]);
	    int i = 0;
	    
	    while (token != NULL)
	    {
		    ret.back()[i++] = atof(token);
		    token = strtok(NULL, ",");
	    }
    }

    fclose(fp);
    if (line)
        free(line);
}
