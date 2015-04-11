#!/usr/bin/python3


if __name__ == '__main__':
    for fname in ['abs_esp.tex', 'abs_en.tex']:
        with open(fname, 'r') as f:
            lines = f.readlines()
            abs_line = ''

            for line in lines:
                if line.startswith('\\noindent '):
                    abs_line = line

            abs_line = abs_line[10:]

            print(
                "La cantidad de palabras del abstract {f} es".format(f=fname),
                len(abs_line.split())
            )
