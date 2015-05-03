#include <glog/logging.h>
#include <mpi.h>
#include <utility>
#include <vector>

#include "da/gary.h"
#include "da/dary.h"
#include "da/ary.h"


using std::make_pair;
using std::vector;
void Debug() {
  int i = 0;
  char hostname[256];
  gethostname(hostname, sizeof(hostname));
  printf("PID %d on %s ready for attach\n", getpid(), hostname);
  fflush(stdout);
  while (0 == i)
    sleep(5);
}



void TestPar(int pdim, int rank){
  lapis::DAry a1, a2;
  lapis::DAry a3, a4;
  vector<lapis::Range> slice{make_pair(0,4), make_pair(0,8)};
  a1.SetShape({4,8});
  a2.SetShape({4,8});
  a1.Setup(pdim);
  a2.Setup(pdim);
  a1.Random();
  a2.Random();
  ARMCI_Barrier();


  if(rank==0){
    //Debug();
    LOG(ERROR)<<"test simple partition along "<< pdim<<" dim";
    a3=a1.Fetch(slice);
    a4=a2.Fetch(slice);
    LOG(ERROR)<<"fetch a";
    LOG(ERROR)<<a3.ToString();
    LOG(ERROR)<<"fetch b";
    LOG(ERROR)<<a4.ToString();
    a3.Add(a4);
    LOG(ERROR)<<"a<- a+b";
    LOG(ERROR)<<a3.ToString();
  }
  ARMCI_Barrier();
  a1.Add(a2);
  ARMCI_Barrier();
  if(rank==0){
    lapis::DAry a5;
    a5=a1.Fetch(slice);
    LOG(ERROR)<<"add then fetch";
    LOG(ERROR)<<a5.ToString();
  }
}



void TestMixedParElt(int pa, int pb, int pc, int rank){
  LOG(ERROR)<<" p dim for a,b,c is "<<pa<<" "<<pb<<" "<<pc;
  vector<lapis::Range> slice{make_pair(0,3),make_pair(0,6), make_pair(0,2)};
  lapis::DAry a1, a2, a3;
  a1.SetShape({3,6,2});
  a2.SetShape({3,6,2});
  a3.SetShape({3,6,2});
  a1.Setup(pa);
  a2.Setup(pb);
  a3.Setup(pc);
  a1.Random();
  a2.Random();
  a3.Random();

  ARMCI_Barrier();
  if(rank==0){
    LOG(ERROR)<<"test elementwise ops with mixed partition";
    lapis::DAry a5, a4;
//    Debug();
    a5=a1.Fetch(slice);
    a4=a2.Fetch(slice);
    LOG(ERROR)<<"fetch a";
    LOG(ERROR)<<a5.ToString();
    LOG(ERROR)<<"fetch b";
    LOG(ERROR)<<a4.ToString();
    a5.Copy(a4);
    LOG(ERROR)<<"fetch op a.Copy(b)";
    LOG(ERROR)<<a5.ToString();
  }
  ARMCI_Barrier();
  a1.Copy(a2);
  ARMCI_Barrier();
  if(rank==0){
    lapis::DAry a5;
    a5=a1.Fetch(slice);
    LOG(ERROR)<<"op fetch a.Copy(b)";
    LOG(ERROR)<<a5.ToString();
  }

//////////////////////////////////////////////////
  ARMCI_Barrier();
  if(rank==0){
    lapis::DAry a8, a4, a5({3,6,2});
    //Debug();
    a8=a1.Fetch(slice);
    a4=a2.Fetch(slice);
    LOG(ERROR)<<"fetch a";
    LOG(ERROR)<<a8.ToString();
    LOG(ERROR)<<"fetch b";
    LOG(ERROR)<<a4.ToString();
    a5.Mult(a8,a4);
    LOG(ERROR)<<"fetch op c.mult(a,b)";
    LOG(ERROR)<<a5.ToString();
  }
  ARMCI_Barrier();
  a3.Mult(a1,a2);
  ARMCI_Barrier();
  if(rank==0){
    lapis::DAry a5;
    a5=a3.Fetch(slice);
    LOG(ERROR)<<"op fetch a.Mult(b,c)";
    LOG(ERROR)<<a5.ToString();
  }
//////////////////////////////////////////////////
  ARMCI_Barrier();
  if(rank==0){
    lapis::DAry a8, a4, a5({3,6,2});
    //Debug();
    a8=a1.Fetch(slice);
    a4=a2.Fetch(slice);
    LOG(ERROR)<<"fetch a";
    LOG(ERROR)<<a8.ToString();
    LOG(ERROR)<<"fetch b";
    LOG(ERROR)<<a4.ToString();
    a5.Div(a8,a4);
    LOG(ERROR)<<"fetch op c.div(a,b)";
    LOG(ERROR)<<a5.ToString();
  }
  ARMCI_Barrier();
  a3.Div(a1,a2);
  ARMCI_Barrier();
  if(rank==0){
    lapis::DAry a5;
    a5=a3.Fetch(slice);
    LOG(ERROR)<<"op fetch a.div(b,c)";
    LOG(ERROR)<<a5.ToString();
  }
//////////////////////////////////////////////////
  ARMCI_Barrier();
  if(rank==0){
    lapis::DAry a8, a4, a5({3,6,2});
    //Debug();
    a8=a1.Fetch(slice);
    LOG(ERROR)<<"fetch a";
    LOG(ERROR)<<a8.ToString();
    a5.Mult(a8, 3.0);
    LOG(ERROR)<<"fetch op c.mult(a,3)";
    LOG(ERROR)<<a5.ToString();
  }
  ARMCI_Barrier();
  a3.Mult(a1,3.0);
  ARMCI_Barrier();
  if(rank==0){
    lapis::DAry a5;
    a5=a3.Fetch(slice);
    LOG(ERROR)<<"op fetch a.mult(b,3)";
    LOG(ERROR)<<a5.ToString();
  }

//////////////////////////////////////////////////
  ARMCI_Barrier();
  if(rank==0){
    lapis::DAry a8, a4, a5({3,6,2});
    //Debug();
    a8=a1.Fetch(slice);
    LOG(ERROR)<<"fetch a";
    LOG(ERROR)<<a8.ToString();
    a5.Square(a8);
    LOG(ERROR)<<"fetch op c.square(a)";
    LOG(ERROR)<<a5.ToString();
  }
  ARMCI_Barrier();
  a3.Square(a1);
  ARMCI_Barrier();
  if(rank==0){
    lapis::DAry a5;
    a5=a3.Fetch(slice);
    LOG(ERROR)<<"op fetch a.sqaure(b)";
    LOG(ERROR)<<a5.ToString();
  }


//////////////////////////////////////////////////
  ARMCI_Barrier();
  if(rank==0){
    lapis::DAry a8, a4, a5({3,6,2});
    //Debug();
    a8=a1.Fetch(slice);
    LOG(ERROR)<<"fetch a";
    LOG(ERROR)<<a8.ToString();
    a5.Pow(a8,3.0);
    LOG(ERROR)<<"fetch op c.pow(a, 3)";
    LOG(ERROR)<<a5.ToString();
  }
  ARMCI_Barrier();
  a3.Pow(a1,3.0);
  ARMCI_Barrier();
  if(rank==0){
    lapis::DAry a5;
    a5=a3.Fetch(slice);
    LOG(ERROR)<<"op fetch a.pow(b,3)";
    LOG(ERROR)<<a5.ToString();
  }


//////////////////////////////////////////////////
  ARMCI_Barrier();
  a3.SampleUniform(0.0,3.0);
  ARMCI_Barrier();
  if(rank==0){
    lapis::DAry a5;
    a5=a3.Fetch(slice);
    LOG(ERROR)<<"op fetch a.uniform(0,3)";
    LOG(ERROR)<<a5.ToString();
  }
//////////////////////////////////////////////////
  ARMCI_Barrier();
  a3.SampleGaussian(0.0,1.0);
  ARMCI_Barrier();
  if(rank==0){
    lapis::DAry a5;
    a5=a3.Fetch(slice);
    LOG(ERROR)<<"op fetch a.norm(0,1)";
    LOG(ERROR)<<a5.ToString();
  }

//////////////////////////////////////////////////
  ARMCI_Barrier();
  a3.Fill(1.43);
  ARMCI_Barrier();
  if(rank==0){
    lapis::DAry a5;
    a5=a3.Fetch(slice);
    LOG(ERROR)<<"op fetch a.fill(1.43)";
    LOG(ERROR)<<a5.ToString();
  }


//////////////////////////////////////////////////
  ARMCI_Barrier();
  a1.Random();
  ARMCI_Barrier();
  if(rank==0){
    lapis::DAry a8, a4, a5({3,6,2});
    a4=a1.Fetch(slice);
    a5.Threshold(a4,0.3);
    LOG(ERROR)<<"fetch op b=threshold(a,0.3)";
    LOG(ERROR)<<a4.ToString();
    LOG(ERROR)<<a5.ToString();
  }

  ARMCI_Barrier();
  a3.Threshold(a1, .30f);
  ARMCI_Barrier();
  if(rank==0){
    lapis::DAry a5;
    a5=a3.Fetch(slice);
    LOG(ERROR)<<"op fetch b=threshold(a,0.3)";
    LOG(ERROR)<<a5.ToString();
  }

//////////////////////////////////////////////////
  ARMCI_Barrier();
  a1.Random();
  ARMCI_Barrier();
  if(rank==0){
    lapis::DAry a8, a4, a5({3,6,2});
    a4=a1.Fetch(slice);
    a5.Max(a4,0.3);
    LOG(ERROR)<<"fetch op b=max(a,0.3)";
    LOG(ERROR)<<a4.ToString();
    LOG(ERROR)<<a5.ToString();
  }

  ARMCI_Barrier();
  a3.Max(a1, .30f);
  ARMCI_Barrier();
  if(rank==0){
    lapis::DAry a5;
    a5=a3.Fetch(slice);
    LOG(ERROR)<<"op fetch b=max(a,0.3)";
    LOG(ERROR)<<a5.ToString();
  }


//////////////////////////////////////////////////
  ARMCI_Barrier();
  if(rank==0){
    lapis::DAry a6, a4, a5({3,6,2});
    a6=a1.Fetch(slice);
    a4=a2.Fetch(slice);
    a5.Map([](float a, float b) {return a+2*b;}, a6,a4);
    LOG(ERROR)<<"fetch op b=map(a+2b)";
    LOG(ERROR)<<a6.ToString();
    LOG(ERROR)<<a4.ToString();
    LOG(ERROR)<<a5.ToString();
  }
  ARMCI_Barrier();
  a3.Map([](float a, float b) {return a+2*b;}, a1,a2);
  if(rank==0){
    lapis::DAry a5;
    a5=a3.Fetch(slice);
    LOG(ERROR)<<"op fetch b=map(a+2b)";
    LOG(ERROR)<<a5.ToString();
  }
  LOG(ERROR)<<"finish elementwise ops";
}


void TestLargeDot(int pa, int pb, int pc, int rank){
  if(rank==0){
    LOG(ERROR)<<"test Dot, partition for a, b, c : "
      << pa<<" "<<pb<<" "<<pc<<" dim";
  }

  double t1, t2, t3;
  t1=MPI_Wtime();
  lapis::DAry a,b,c;
  a.SetShape({256,9216});
  b.SetShape({9216,4096});
  c.SetShape({256,4096});
  a.Setup(pa);
  b.Setup(pb);
  c.Setup(pc);
  a.Random();
  b.Random();
  c.Random();
  ARMCI_Barrier();
  t2=MPI_Wtime();
  c.Dot(a,b);
  t3=MPI_Wtime();
  ARMCI_Barrier();
  LOG(ERROR)<<"setup time: "<<t2-t1<<" dot time: "
    <<t3-t2<<" wait time:"<<MPI_Wtime()-t3;
}

void TestDot(int pa, int pb, int pc, int rank){
  vector<lapis::Range> slicea{make_pair(0,4), make_pair(0,8)};
  vector<lapis::Range> sliceb{make_pair(0,8), make_pair(0,4)};
  vector<lapis::Range> slicec{make_pair(0,4), make_pair(0,4)};
  lapis::DAry a,b,c;
  a.SetShape({4,8});
  b.SetShape({8,4});
  c.SetShape({4,4});
  a.Setup(pa);
  b.Setup(pb);
  c.Setup(pc);
  a.Random();
  b.Random();
  c.Random();
  //////////////////////
  ARMCI_Barrier();
  if(rank==0){
    LOG(ERROR)<<"test Dot, partition for a, b, c : "
      << pa<<" "<<pb<<" "<<pc<<" dim";
    LOG(ERROR)<<"c=a*b";
    lapis::DAry x,y,z;
    x=a.Fetch(slicea);
    y=b.Fetch(sliceb);
    z=c.Fetch(slicec);
    z.Dot(x,y);
    LOG(ERROR)<<"fetch dot ";
    LOG(ERROR)<<z.ToString();
  }
  ARMCI_Barrier();
  //Debug();
  c.Dot(a,b);
  ARMCI_Barrier();
  if(rank==0){
    lapis::DAry z;
    z=c.Fetch(slicec);
    LOG(ERROR)<<"dot fetch";
    LOG(ERROR)<<z.ToString();
  }
  /////////////////////////////
  ARMCI_Barrier();

  if(rank==0){
    LOG(ERROR)<<"a=c*b^T";
    lapis::DAry x,y,z;
    x=a.Fetch(slicea);
    y=b.Fetch(sliceb);
    z=c.Fetch(slicec);
    x.Dot(z,y, false, true);
    LOG(ERROR)<<"fetch dot ";
    LOG(ERROR)<<x.ToString();
  }
  ARMCI_Barrier();
  //Debug();
  a.Dot(c,b, false, true);
  ARMCI_Barrier();
  if(rank==0){
    lapis::DAry z;
    z=a.Fetch(slicea);
    LOG(ERROR)<<"dot fetch";
    LOG(ERROR)<<z.ToString();
  }

  /////////////////////////////
  ARMCI_Barrier();
  if(rank==0){
    LOG(ERROR)<<"b=a^T*c";
    lapis::DAry x,y,z;
    x=a.Fetch(slicea);
    y=b.Fetch(sliceb);
    z=c.Fetch(slicec);
    y.Dot(x,z, true, false);
    LOG(ERROR)<<"fetch dot ";
    LOG(ERROR)<<y.ToString();
  }
  ARMCI_Barrier();
  //Debug();
  b.Dot(a,c, true, false);
  ARMCI_Barrier();
  if(rank==0){
    lapis::DAry z;
    z=b.Fetch(sliceb);
    LOG(ERROR)<<"dot fetch";
    LOG(ERROR)<<z.ToString();
  }
  ARMCI_Barrier();
  /////////////////////////////
  ARMCI_Barrier();
  if(rank==0){
    LOG(ERROR)<<"b=a^T*c^T";
    lapis::DAry x,y,z;
    x=a.Fetch(slicea);
    y=b.Fetch(sliceb);
    z=c.Fetch(slicec);
    y.Dot(x,z, true, true);
    LOG(ERROR)<<"fetch dot ";
    LOG(ERROR)<<y.ToString();
  }
  ARMCI_Barrier();
  //Debug();
  b.Dot(a,c, true, true);
  ARMCI_Barrier();
  if(rank==0){
    lapis::DAry z;
    z=b.Fetch(sliceb);
    LOG(ERROR)<<"dot fetch";
    LOG(ERROR)<<z.ToString();
  }
  ARMCI_Barrier();
}


void TestSubarray(int pa, int pb, int pc, int rank){
  vector<lapis::Range> slicea{make_pair(0,4), make_pair(0,8)};
  vector<lapis::Range> sliceb{make_pair(0,8), make_pair(0,4)};
  vector<lapis::Range> slicec{make_pair(0,4), make_pair(0,4)};
  vector<lapis::Range> slice{make_pair(0,4)};
  lapis::DAry a,b,c;
  a.SetShape({4});
  b.SetShape({8,4});
  c.SetShape({4,4});
  a.Setup(pa);
  b.Setup(pb);
  c.Setup(pc);
  b.Random();
  c.Random();

  //Debug();
  lapis::DAry sb=b[2];
  lapis::DAry sc=c[3];

  ARMCI_Barrier();
  if(rank==0){
    LOG(ERROR)<<"test subary, partition for a, b, c : "
      << pa<<" "<<pb<<" "<<pc<<" dim";
    lapis::DAry y,z, x({4});
    LOG(ERROR)<<"fetch full b, c";
    y=b.Fetch(sliceb);
    z=c.Fetch(slicec);
    LOG(ERROR)<<y.ToString();
    LOG(ERROR)<<z.ToString();
    LOG(ERROR)<<"fetch sub, sb[2], sc[3]";
    y=sb.Fetch(slice);
    z=sc.Fetch(slice);
    LOG(ERROR)<<y.ToString();
    LOG(ERROR)<<z.ToString();
  }
  ARMCI_Barrier();
  a.Add(sb,sc);
  ARMCI_Barrier();
  //Debug();
  if(rank==0){
    lapis::DAry z;
    z=a.Fetch(slice);
    LOG(ERROR)<<"sub add fetch, sb[2]+sc[3]";
    LOG(ERROR)<<z.ToString();
  }
}

void TestReshape(int pa, int pb, int pc, int rank){
  vector<lapis::Range> sliceb3{make_pair(0,2),make_pair(0,4), make_pair(0,4)};
  vector<lapis::Range> sliceb{make_pair(0,8), make_pair(0,4)};
  vector<lapis::Range> slicec{make_pair(0,4), make_pair(0,4)};
  vector<lapis::Range> slicea{make_pair(0,4)};
  lapis::DAry a,b,c,b3,b2,b1;
  a.SetShape({4});
  b.SetShape({8,4});
  c.SetShape({4,4});
  a.Setup(pa);
  b.Setup(pb);
  c.Setup(pc);
  b.Random();
  c.Random();

  b3=b.Reshape({2,4,4});
  //Debug() ;
  b2=b3[1];
  if(rank==0){
    LOG(ERROR)<<"test reshape+subary, partition for a, b, c : "
      << pa<<" "<<pb<<" "<<pc<<" dim";
    lapis::DAry y,z,x;
    LOG(ERROR)<<"fetch b, b2, c";
    y=b.Fetch(sliceb);
    z=b2.Fetch(slicec);
    x=c.Fetch(slicec);
    LOG(ERROR)<<y.ToString();
    LOG(ERROR)<<z.ToString();
    LOG(ERROR)<<x.ToString();
    LOG(ERROR)<<"fetch sub, b2+c";
    z.Add(x);
    LOG(ERROR)<<z.ToString();
  }

  ARMCI_Barrier();
  c.Add(b2);
  ARMCI_Barrier();
  if(rank==0){
    lapis::DAry y,z,x;
    x=c.Fetch(slicec);
    LOG(ERROR)<<"sub add,fetch c+b2";
    LOG(ERROR)<<x.ToString();
  }
  ARMCI_Barrier();
  b2.Add(c);
  ARMCI_Barrier();
  if(rank==0){
    lapis::DAry y,z,x;
    x=b2.Fetch(slicec);
    LOG(ERROR)<<"sub add,fetch b2+c";
    LOG(ERROR)<<x.ToString();
  }
  ARMCI_Barrier();
  b1=b2[2];
  if(rank==0){
    lapis::DAry y,z,x;
    x=b1.Fetch(slicea);
    LOG(ERROR)<<"fetch b1";
    LOG(ERROR)<<x.ToString();
  }

  a.Add(b1);
  ARMCI_Barrier();
  if(rank==0){
    lapis::DAry y,z,x;
    x=a.Fetch(slicea);
    LOG(ERROR)<<"add fetch a+b1";
    LOG(ERROR)<<x.ToString();
  }
  ARMCI_Barrier();
  b1.Add(a);
  ARMCI_Barrier();
  if(rank==0){
    lapis::DAry y,z,x;
    x=b1.Fetch(slicea);
    LOG(ERROR)<<"add fetch b1+a";
    LOG(ERROR)<<x.ToString();
  }

  ARMCI_Barrier();
  {
    lapis::DAry b3=b.Reshape({4,2,4});
    lapis::DAry a;
    a.SetShape({2,4});
    a.Setup(pa);
    a.Random();
    lapis::DAry b1=b3[1];
    lapis::DAry b2=b3[3];
    lapis::DAry c;
    c.SetShape({2,2});
    c.Setup(pc);
    ARMCI_Barrier();
    c.Dot(a,b2,false, true);
    ARMCI_Barrier();
    if(rank==0){
      lapis::DAry x,y,z,zz({2,2});
      y=b3.Fetch({make_pair(0,4), make_pair(0,2), make_pair(0,4)});
      x=a.Fetch({make_pair(0,2), make_pair(0,4)});
      LOG(ERROR)<<"fetch b,a";
      LOG(ERROR)<<y.ToString();
      LOG(ERROR)<<x.ToString();
      z=y[3];
      zz.Dot(x,z,false, true);
      LOG(ERROR)<<"fetch dot c=a*b[3]^T";
      LOG(ERROR)<<zz.ToString();

      x=a.Fetch({make_pair(0,2), make_pair(0,4)});
      y=b2.Fetch({make_pair(0,2), make_pair(0,4)});
      z=c.Fetch({make_pair(0,2), make_pair(0,2)});
      LOG(ERROR)<<"op fetch c=a*b[3]^T";
      LOG(ERROR)<<x.ToString();
      LOG(ERROR)<<y.ToString();
      LOG(ERROR)<<z.ToString();

    }
    ARMCI_Barrier();
  }
}



int main(int argc, char**argv){
 // MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  MPI_Init(&argc, &argv);
  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  vector<int> procs;
  for (int i = 0; i < nprocs; i++) {
    procs.push_back(i);
  }
  //Debug();
  lapis::GAry::Init(rank,procs);
  google::InitGoogleLogging(argv[0]);
  /*
  if(nprocs%3==0){
    TestMixedParElt(0,0,0,rank);
    TestMixedParElt(0,0,1,rank);
    TestMixedParElt(0,1,0,rank);
    TestMixedParElt(1,0,0,rank);
    TestMixedParElt(1,1,0,rank);
    TestMixedParElt(1,1,1,rank);
    TestMixedParElt(0,1,1,rank);
  }
  if(nprocs%2==0){
    TestMixedParElt(1,1,1,rank);
    TestMixedParElt(1,2,1,rank);
    TestMixedParElt(2,1,1,rank);
    TestMixedParElt(1,1,2,rank);
    TestMixedParElt(2,2,2,rank);
  }
  TestDot(0,0,0,rank);
  TestDot(0,0,1,rank);
  TestDot(0,1,0,rank);
  TestDot(0,1,1,rank);
  TestDot(1,0,0,rank);
  TestDot(1,0,1,rank);
  TestDot(1,1,0,rank);
  TestDot(1,1,1,rank);

  TestPar(0, rank);
  TestPar(1, rank);
  */
  double start, end;
  start=MPI_Wtime();
  TestLargeDot(0,0,0,rank);
  TestLargeDot(0,0,1,rank);
  TestLargeDot(0,1,0,rank);
  TestLargeDot(0,1,1,rank);
  TestLargeDot(1,0,0,rank);
  TestLargeDot(1,0,1,rank);
  TestLargeDot(1,1,0,rank);
  TestLargeDot(1,1,1,rank);
  end=MPI_Wtime();
  if(rank==0)
    LOG(ERROR)<<"dot time for 256*4k 4k*4k matrix, "<<end-start;
  /*
  TestSubarray(0,0,0,rank);
  TestSubarray(0,0,1,rank);
  TestSubarray(0,1,0,rank);
  TestSubarray(0,1,1,rank);
  TestReshape(0,0,0,rank);
  TestReshape(0,0,1,rank);
  TestReshape(0,1,0,rank);
  TestReshape(0,1,1,rank);
  */

  LOG(ERROR)<<"finish";
  lapis::GAry::Finalize();
  MPI_Finalize();
  return 0;
}

