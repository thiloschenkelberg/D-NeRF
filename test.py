from numpy import float64


def done():
    chance = float64(1.)
    for i in range(0,16000):
        chance *= float64((160_000-i)/16_000_000)
    return chance
    
    
if __name__=='__main__':
    chance = done()
    print(chance)