/* This file was automatically generated by CasADi 3.6.4.
 *  It consists of: 
 *   1) content generated by CasADi runtime: not copyrighted
 *   2) template code copied from CasADi source: permissively licensed (MIT-0)
 *   3) user code: owned by the user
 *
 */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) Dynamic_Pacejka_model_impl_dae_fun_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

static const casadi_int casadi_s0[14] = {10, 1, 0, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
static const casadi_int casadi_s1[7] = {3, 1, 0, 3, 0, 1, 2};
static const casadi_int casadi_s2[3] = {0, 0, 0};

/* Dynamic_Pacejka_model_impl_dae_fun:(i0[10],i1[10],i2[3],i3[],i4[],i5[3])->(o0[10]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a2, a3, a4, a5, a6, a7, a8, a9;
  a0=arg[1]? arg[1][0] : 0;
  a1=arg[0]? arg[0][3] : 0;
  a2=arg[0]? arg[0][2] : 0;
  a3=cos(a2);
  a3=(a1*a3);
  a4=arg[0]? arg[0][4] : 0;
  a5=sin(a2);
  a5=(a4*a5);
  a3=(a3-a5);
  a0=(a0-a3);
  if (res[0]!=0) res[0][0]=a0;
  a0=arg[1]? arg[1][1] : 0;
  a3=sin(a2);
  a3=(a1*a3);
  a2=cos(a2);
  a2=(a4*a2);
  a3=(a3+a2);
  a0=(a0-a3);
  if (res[0]!=0) res[0][1]=a0;
  a0=arg[1]? arg[1][2] : 0;
  a3=arg[0]? arg[0][5] : 0;
  a0=(a0-a3);
  if (res[0]!=0) res[0][2]=a0;
  a0=arg[1]? arg[1][3] : 0;
  a2=2.8818443804034583e-01;
  a5=1.0000000000000001e-01;
  a6=arg[0]? arg[0][8] : 0;
  a6=(a6-a1);
  a5=(a5*a6);
  a6=1.8999999999999999e+00;
  a7=10.;
  a8=1.5875000000000000e-01;
  a9=(a8*a3);
  a9=(a9+a4);
  a10=9.9999999999999998e-13;
  a11=(a1+a10);
  a9=(a9/a11);
  a9=atan(a9);
  a9=(a7*a9);
  a9=(-a9);
  a9=atan(a9);
  a9=(a6*a9);
  a9=sin(a9);
  a11=arg[0]? arg[0][7] : 0;
  a12=sin(a11);
  a12=(a9*a12);
  a5=(a5-a12);
  a12=3.4700000000000002e+00;
  a13=(a12*a4);
  a13=(a13*a3);
  a5=(a5+a13);
  a5=(a2*a5);
  a0=(a0-a5);
  if (res[0]!=0) res[0][3]=a0;
  a0=arg[1]? arg[1][4] : 0;
  a5=1.7144999999999999e-01;
  a13=(a5*a3);
  a13=(a13-a4);
  a1=(a1+a10);
  a13=(a13/a1);
  a13=atan(a13);
  a13=(a13+a11);
  a7=(a7*a13);
  a7=atan(a7);
  a6=(a6*a7);
  a6=sin(a6);
  a7=cos(a11);
  a7=(a9*a7);
  a7=(a6-a7);
  a12=(a12*a4);
  a12=(a12*a3);
  a7=(a7-a12);
  a2=(a2*a7);
  a0=(a0-a2);
  if (res[0]!=0) res[0][4]=a0;
  a0=arg[1]? arg[1][5] : 0;
  a2=2.1222410865874362e+01;
  a8=(a8*a9);
  a11=cos(a11);
  a8=(a8*a11);
  a5=(a5*a6);
  a8=(a8-a5);
  a2=(a2*a8);
  a0=(a0-a2);
  if (res[0]!=0) res[0][5]=a0;
  a0=arg[1]? arg[1][6] : 0;
  a2=arg[0]? arg[0][9] : 0;
  a0=(a0-a2);
  if (res[0]!=0) res[0][6]=a0;
  a0=arg[1]? arg[1][7] : 0;
  a2=arg[2]? arg[2][0] : 0;
  a0=(a0-a2);
  if (res[0]!=0) res[0][7]=a0;
  a0=arg[1]? arg[1][8] : 0;
  a2=arg[2]? arg[2][1] : 0;
  a0=(a0-a2);
  if (res[0]!=0) res[0][8]=a0;
  a0=arg[1]? arg[1][9] : 0;
  a2=arg[2]? arg[2][2] : 0;
  a0=(a0-a2);
  if (res[0]!=0) res[0][9]=a0;
  return 0;
}

CASADI_SYMBOL_EXPORT int Dynamic_Pacejka_model_impl_dae_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int Dynamic_Pacejka_model_impl_dae_fun_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int Dynamic_Pacejka_model_impl_dae_fun_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Dynamic_Pacejka_model_impl_dae_fun_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int Dynamic_Pacejka_model_impl_dae_fun_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Dynamic_Pacejka_model_impl_dae_fun_release(int mem) {
}

CASADI_SYMBOL_EXPORT void Dynamic_Pacejka_model_impl_dae_fun_incref(void) {
}

CASADI_SYMBOL_EXPORT void Dynamic_Pacejka_model_impl_dae_fun_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int Dynamic_Pacejka_model_impl_dae_fun_n_in(void) { return 6;}

CASADI_SYMBOL_EXPORT casadi_int Dynamic_Pacejka_model_impl_dae_fun_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real Dynamic_Pacejka_model_impl_dae_fun_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Dynamic_Pacejka_model_impl_dae_fun_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    case 5: return "i5";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Dynamic_Pacejka_model_impl_dae_fun_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Dynamic_Pacejka_model_impl_dae_fun_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s1;
    case 3: return casadi_s2;
    case 4: return casadi_s2;
    case 5: return casadi_s1;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Dynamic_Pacejka_model_impl_dae_fun_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int Dynamic_Pacejka_model_impl_dae_fun_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 6;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
