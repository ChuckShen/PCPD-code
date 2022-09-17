// compile with: g++ -std=c++11 -O3 pcpd1dml.cpp -o pcpd1dml
// usage: ./pcpd1dml D p L T poccup RUN 
///Parameters:
// D: diffusion rate; p: contral parameter; L: system size; 
// T: simulation time steps; poccup: initial occupation probability; 
// RUN: number of runs
#include <iostream>
#include <fstream>
#include <string>
#include <numeric>
#include <cmath>
#include <sys/time.h>
#include <ctime>
#include <climits>
#include <cstdlib>

using namespace std;

// time seed
struct timeval time_now{};
time_t msecs_time;

// lateral size, simulation time steps, number of runs, number of thread for each job
int L, T, TT, RUN;

// initial occupation probability
float poccup;

/* rates */
float d, p;

/* propensities */
float dl, dr, ann, cl /*,cr*/;


///////////////////-xoshiro256**-///////////////////////
static inline uint64_t rotl(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
}

uint64_t next(uint64_t (&st)[4]) {
        const uint64_t result_starstar = rotl(st[1] * 5, 7) * 9;

        const uint64_t t = st[1] << 17;

        st[2] ^= st[0];
        st[3] ^= st[1];
        st[1] ^= st[2];
        st[0] ^= st[3];

        st[2] ^= t;

        st[3] = rotl(st[3], 45);

        return result_starstar;
}

double get_rand(uint64_t (&st)[4]) {
        return static_cast<double>(next(st)) / UINT64_MAX;
}
/////////////////////////////////////////////////


void Move(bool& sitei, bool& sitej) { 
	if (sitei && !sitej) { sitei = 0; sitej = 1;}
}

void Annihilate(bool& sitei, bool& sitej) {
        if (sitei && sitej) { sitei = 0; sitej = 0;}
}

void Create(bool& sitei, bool& sitej, bool& sitek) {
        if (sitei && sitej && !sitek) { sitek = 1; }
}

void Update(bool* s, uint64_t (&st)[4], int i, int L) 
{
	int h, /*i*/ j, k, l;
	h = (i - 1 + L) % L;
	j = (i + 1) % L;
        k = (i + 2) % L;
	l = (i + 3) % L;

	double r = get_rand(st);
	if (r<dl) Move(s[j], s[k]); // diff to right
	else if (r<dr) Move(s[j], s[i]); //diff to left
	else if (r<ann) Annihilate(s[j], s[k]);
	else if (r<cl) Create(s[j], s[k], s[l]); //create at right
	else Create(s[i], s[j], s[h]); // create at left
}

void doPCPD(bool* s, uint64_t (&st)[4], int L) {
	for (int i = 0; i < L; i++) Update(s, st, next(st)%L, L);
	//for (int i = 1; i < L; i+=2) Update(i);
	//for (int i = 0; i < L-1; i+=2) Update(i);
}

void do_it(int L, int RUN, float poccup, double* rho, int* surv, uint64_t (&st)[4])
{
   bool s[L];
   for (int run = 0; run < RUN; run++) {
     if (poccup == 1) fill_n(s, L, 1);
     else {
         for (int i = 0; i < L; i++) {
             if (get_rand(st) < poccup) s[i] = 1;
             else s[i] = 0;
         }
     }
     for (int y = 0; y < T; y++) {
       if (y%5==0) {
           int NA = accumulate(s, s+L, 0);
           rho[y/5] += (double)(NA)/L;
           if (NA > 1) surv[y/5] += 1;
       }
       doPCPD(s, st, L);
       }
  }
}


int main(int argc, char *argv[]){
   gettimeofday(&time_now, nullptr);
   time_t msecs_time = (time_now.tv_sec * 10000) + (time_now.tv_usec / 100); // time since epoch in unit of 10^-4 sec
   
   static uint64_t st[4] = { 0x180ec6d33cfd0aba*(uint64_t)(msecs_time), 0xd5a61266f0c9392c*(uint64_t)(msecs_time), 0xa9582618e03fc9aa*(uint64_t)(msecs_time), 0x39abdc4529b1661c*(uint64_t)(msecs_time) };   
   
   //////////////////-rates-//////////////
   d = strtod(argv[1], NULL);
   p = strtod(argv[2], NULL);
  
   dl = d/2; dr = d;
   ann = dr + p*(1-dr);
   cl = ann + (1-p)*(1-dr)/2;
   //////////////////////////////////////
   
   L = stoi(argv[3]);                // lateral size
   T = stoi(argv[4]);                // simulation time steps
   TT = T/5;
   poccup = strtod(argv[5], NULL);   // initial occupation probability
   RUN = stoi(argv[6]);              // number of runs
   
   double rho[TT] = {0};
   int surv[TT] = {0};
   
   string filename, D, Pval, Lval, Tval, poccupval, RUNval, time_stamp;
   D = argv[1]; Pval = argv[2]; Lval = argv[3]; Tval = argv[4];
   poccupval = argv[5]; RUNval = argv[6];
   time_stamp = to_string(msecs_time);
   filename = "PCPD1d_"+D+"_"+Pval+"_"+Lval+"_"+Tval+"_"+poccupval+"_"+RUNval+"_"+time_stamp+".dat";

   ofstream outfile;
   outfile.open(filename.c_str());
   
   do_it(L, RUN, poccup, rho, surv, st);

   for (int time=0; time<T/5; time++) {
       outfile << 5*time << "\t" << rho[time]/RUN << "\t" << (double)(surv[time])/RUN << endl;
   }
   outfile.close();

   return 0;
}