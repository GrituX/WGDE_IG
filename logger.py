import os


class Logger:
    out_dir = ''
    do_print = True

    @staticmethod
    def log(s):
        file_out = os.path.join(Logger.out_dir, 'print.log')
        with open(file_out, 'a') as f:
            f.write(s+'\n')
        if Logger.do_print:
            print(s)
