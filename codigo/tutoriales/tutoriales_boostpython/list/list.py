#  Copyright Joel de Guzman 2002-2007. Distributed under the Boost
#  Software License, Version 1.0. (See accompanying file LICENSE_1_0.txt
#  or copy at http://www.boost.org/LICENSE_1_0.txt)
#  Hello World Example from the tutorial

import list



if __name__ == '__main__':
    l = list.Lista()
    
    l.add(4)
    l.add(3)
    l.add(2)
    l.add(1)
    
    print l.sub(0), "== 1"
    print l.sub(1), "== 2"
    print l.sub(2), "== 3"
    print l.sub(3), "== 4"
