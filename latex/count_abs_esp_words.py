#!/usr/bin/python3


if __name__ == '__main__':
    with open('abs_esp.tex', 'r') as f:
        lines = f.readlines()
        abs_line = ''

        for line in lines:
            if line.startswith('\\noindent '):
                abs_line = line

        abs_line = abs_line[10:]

        print(
            "La cantidad de palabras del abstract es",
            len(abs_line.split())
        )
