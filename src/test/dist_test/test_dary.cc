#include <iostream>
#include "darray/dary.h"
#include "utils/timer.h"


int main() {
  lapis::DAry x({1000000});
  lapis::DAry y({1000000});
  x.Random();
  y.Random();
  lapis::Timer t;
  for(int i=0;i<100;i++){
    float *dptrx=x.dptr();
    float *dptry=y.dptr();
    for(int k=0;k<10000;k++)
      dptrx[k]*=dptry[k];
  }
  std::cout<<"arymath: "<<t.elapsed()/10<<std::endl;
  lapis::DAry m({1000000});
  lapis::DAry n({1000000});
  m.Random();
  n.Random();
  t.Reset();
  for(int i=0;i<100;i++)
    m.Mult(m,n);
  std::cout<<"arymath: "<<t.elapsed()/10<<std::endl;


  lapis::DAry a({2,2});
  lapis::DAry b,c;
  b.InitLike(a);
  c.InitLike(a);
  a.Random();
  b.Random();
  std::cout<<a.ToString()<<std::endl;
  std::cout<<b.ToString()<<std::endl;
  c.Dot(a,b);
  std::cout<<"c=a.b"<<c.ToString()<<std::endl;
  a.Add(b);
  std::cout<<"a=a+b"<<a.ToString()<<std::endl;
  a.Mult(a,b);
  std::cout<<"a=a*b"<<a.ToString()<<std::endl;
  a.Minus(a,b);
  std::cout<<"a=a-b"<<a.ToString()<<std::endl;

  c.Random();
  std::cout<<"random c "<<c.ToString()<<std::endl;
  a.Threshold(c, 0.3);
  std::cout<<"a=threshold(c,0.3) "<<a.ToString()<<std::endl;

  a.Pow(c, 0.4);
  std::cout<<"a=Pow(c,0.4) "<<a.ToString()<<std::endl;

  c.Set(0.5);
  std::cout<<"c=set(0.5) "<<c.ToString()<<std::endl;
  a.Square(c);
  std::cout<<"a=square(c) "<<a.ToString()<<std::endl;

  c.Copy(a);
  std::cout<<"c=Copy(a) "<<c.ToString()<<std::endl;

  lapis::DAry d({2});
  d.SumRow(b);
  std::cout<<"d=SumRow(b) "<<d.ToString()<<std::endl;
  d.SumCol(b);
  std::cout<<"d=SumCol(b) "<<d.ToString()<<std::endl;
  b.AddRow(d);
  std::cout<<"b=AddRow(d) "<<b.ToString()<<std::endl;
  b.AddCol(d);
  std::cout<<"b=AddCol(d) "<<b.ToString()<<std::endl;

  std::cout<<"max(b) "<<b.Max()<<std::endl;
  std::cout<<"Sum(b) "<<b.Sum()<<std::endl;

  lapis::DAry e({3,3,3});
  e.SampleGaussian(0.0f,1.0f);
  std::cout<<"Gaussain e "<<e.ToString()<<std::endl;

  lapis::DAry f({9});
  f.Sum(e, 0, {0,2});
  std::cout<<"f.sum  "<<f.ToString()<<std::endl;

  return 0;
}

