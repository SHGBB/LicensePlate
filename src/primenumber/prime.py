import numpy as np
import pandas as pd
import math


class UsualMath:

    def judgment_prim(self, num_list):
        num = int("".join(map(str, num_list)))

        if num % 2 == 0:
            print('This is Not Prime Number.')

        prime_ = []
        for i in range(1, num + 1):
            if num % (i) == 0:
                prime_.append(i)
            else:
                continue

        if len(prime_)==2:
            prime_message1 = 'Prime Number. '.format(num)
            prime_message2 = '{}'.format(prime_)

        else:
            prime_message1 = 'There are Divisor. '.format(num)
            prime_message2 = '{}'.format(prime_)

        return prime_message1, prime_message2

###############################################################################
# m = UsualMath()
# m.judgment_prim([6, 5])

