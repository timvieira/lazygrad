#include <iostream>


inline float invsqrt(float x) {
  float xhalf = 0.5f * x;
  int i = *(int*)&x;            // store floating-point bits in integer
  i = 0x5f3759df - (i >> 1);    // initial guess for Newton's method
  x = *(float*)&i;              // convert new bits into float
  x = x*(1.5f - xhalf*x*x);     // One round of Newton's method
  return x;
}


//double better_pow_fast_precise(double a, int b) {
double fastpow(double a, int b) {
  //assert(b >= 0);
  if (b == 1) return a;
  // Assumes a negative integer exponent.
  int e = -b;
  double d = exp(0);
  double r = 1.0;
  while (e) {
    if (e & 1) r *= a;
    a *= a;
    e >>= 1;
  }
  r *= d;
  return 1/r;
}
