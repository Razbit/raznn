#ifndef LOGGER_H
#define LOGGER_H

/**
 * @file
 * @brief An abstract class for logging.
 */

#define SEVERE 0
#define OTHER 1

/**
 * @brief An abstract class for logging things.
 * @author Eetu "Razbit" Pesonen
 * @version 0.01
 * @date 2015
 * @copyright Any kind of redistribution without explicit permission is prohibited.
 */

class Logger
{
protected:
	int level;

public:
    Logger(){}
    virtual ~Logger(){}

    void set_level(int _level){level = _level;}

    /**
     * @brief Write a log message.
     * @details Uses a function from the printf() -family internally, so is used similarily.
     * @param level Logging level.
     * @param fmt A C-string containing the formatting used.
     */
    virtual void write(int level, const char* fmt, ...) = 0;

    /**
     * @brief Write an error message.
     * @details Uses a function from the printf() -family internally, so is used similarily.
     * @param fmt A C-string containing the formatting used.
     * @param severity An integer for severity of the error (0 == terminate)
     */
    virtual void error(int severity, const char* fmt, ...) = 0;
};

#endif // LOGGER_H
