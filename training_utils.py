def print_logs_to_file(log_line, file_name="train_logs.txt"):
    with open(file_name, "a") as tl:
        tl.write(log_line + "\n")