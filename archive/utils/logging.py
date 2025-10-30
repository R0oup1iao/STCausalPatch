def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)