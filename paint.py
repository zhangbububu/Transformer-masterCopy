

impo

warmup = 4000

def f(x):
    return 512**(-0.5) * min( x**(-0.5),x * (warmup**(-1.5)))

x = [ i for i in range(1,1000)]

y = [f(i) for i in x]
print(y)

plt.plot(x,y)


