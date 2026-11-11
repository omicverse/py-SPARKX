"""Central-chi-square subset of AS 155, translated from CompQuadForm qfc.cpp.

Robert B. Davies (1980), Applied Statistics 29, 323-333. GPL-3.
Only positive coefficients and one degree of freedom per component are needed
by SPARK-X. Preserve its default integration accuracy, not a different null CDF.
"""
import math
import numpy as np

PI = 3.14159265358979


def _exp(x):
    return 0. if x < -50. else math.exp(x)


class Davies:
    def __init__(self, coefficients, point, limit=10000):
        self.lb = np.asarray(coefficients, dtype=float)
        self.c = float(point)
        self.mean = float(self.lb.sum())
        self.sigsq = 0.
        self.intl = 0.
        self.limit = limit
        self.count = 0
        self.fail = False

    def counter(self):
        self.count += 1
        if self.count > self.limit:
            raise RuntimeError('Davies could not locate integration parameters.')

    def errbd(self, u):
        self.counter()
        xc = u*self.sigsq
        total = u*xc
        for lb in self.lb[::-1]:
            x = 2*u*lb; y = 1-x
            xc += lb/y
            total += x*x/y + math.log1p(-x) + x
        return _exp(-.5*total), xc

    def ctff(self, accuracy, up):
        u2=up; u1=0.; c1=self.mean
        rb=2*self.lb.max() if u2>0 else 0.
        while True:
            bound,c2=self.errbd(u2/(1+u2*rb))
            if bound<=accuracy: break
            u1=u2;c1=c2;u2*=2
        while (c1-self.mean)/(c2-self.mean)<.9:
            u=(u1+u2)/2
            bound,xc=self.errbd(u/(1+u*rb))
            if bound>accuracy:u1=u;c1=xc
            else:u2=u;c2=xc
        return c2,u2

    def truncation(self, u, tau):
        self.counter()
        sum2=(self.sigsq+tau)*u*u
        prod1=2*sum2;prod2=0.;prod3=0.;s=0
        for lb in self.lb:
            x=(2*u*lb)**2
            if x>1:
                prod2+=math.log(x);prod3+=math.log1p(x);s+=1
            else:prod1+=math.log1p(x)
        prod2+=prod1;prod3+=prod1
        x=_exp(-.25*prod2)/PI;y=_exp(-.25*prod3)/PI
        err1=1. if s==0 else x*2/s
        err1=min(err1,2.5*y if prod3>1 else 1.)
        err2=1. if .5*sum2<=y else y/(.5*sum2)
        return min(err1,err2)

    def findu(self, ut, accuracy):
        u=ut/4
        if self.truncation(u,0)>accuracy:
            while self.truncation(ut,0)>accuracy:ut*=4
        else:
            ut=u;u/=4
            while self.truncation(u,0)<=accuracy:ut=u;u/=4
        for divisor in (2.,1.4,1.2,1.1):
            u=ut/divisor
            if self.truncation(u,0)<=accuracy:ut=u
        return ut

    def cfe(self, point):
        self.counter()
        ax=abs(point);sign=1 if point>0 else -1;total=0.
        ordered=sorted(self.lb,reverse=True)
        for j in range(len(ordered)-1,-1,-1):
            lb=ordered[j]
            if lb*sign>0:
                ax1=ax-lb;ax2=lb/.0866
                if ax1>ax2:ax=ax1
                else:
                    ax=min(ax,ax2)
                    total=(ax-ax1)/lb+j
                    break
        if total>100:
            self.fail=True
            return 1.
        return 2**(total/4)/(PI*ax*ax)

    def integrate(self, terms, interval, tau, main):
        # Vectorize the integrand; preserve reverse summation order from qfc.
        u=(np.arange(terms,-1,-1)+.5)*interval
        phase=-2*u*self.c
        exponent=-.5*self.sigsq*u*u
        for lb in self.lb[::-1]:
            x=2*lb*u
            exponent-=.25*np.log1p(x*x)
            phase+=np.arctan(x)
        weight=interval/PI*np.where(exponent < -50,0.,np.exp(exponent))/u
        if not main:
            exponent=-.5*tau*u*u
            weight*=1-np.where(exponent < -50,0.,np.exp(exponent))
        self.intl+=float(np.sum(np.sin(.5*phase)*weight))

    def sf(self, accuracy=1e-4):
        sd=math.sqrt(float(2*(self.lb**2).sum()))
        if not sd:return float(self.c<=0)
        ut=16/sd;up=4.5/sd;un=-up;xlim=float(self.limit)
        ut=self.findu(ut,.5*accuracy)
        if self.c!=0 and self.lb.max()>.07*sd:
            tau=.25*accuracy/self.cfe(self.c)
            if self.fail:self.fail=False
            elif self.truncation(ut,tau)<.2*accuracy:
                self.sigsq+=tau;ut=self.findu(ut,.25*accuracy)
        acc=.5*accuracy
        while True:
            upper,up=self.ctff(acc,up);d1=upper-self.c
            if d1<0:return 0.
            lower,un=self.ctff(acc,un);d2=self.c-lower
            if d2<0:return 1.
            interval=2*PI/max(d1,d2)
            terms=ut/interval;aux=3/math.sqrt(acc)
            if terms>aux*1.5:
                if aux>xlim:raise RuntimeError('Davies term limit exceeded.')
                naux=int(math.floor(aux+.5));interval1=ut/naux;x=2*PI/interval1
                if x>abs(self.c):
                    tau=.33*acc/(1.1*(self.cfe(self.c-x)+self.cfe(self.c+x)))
                    if not self.fail:
                        acc*=.67;self.integrate(naux,interval1,tau,False)
                        xlim-=aux;self.sigsq+=tau
                        ut=self.findu(ut,.25*acc);acc*=.75
                        continue
            if terms>xlim:raise RuntimeError('Davies term limit exceeded.')
            self.integrate(int(math.floor(terms+.5)),interval,0.,True)
            return .5+self.intl
