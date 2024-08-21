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
  #define CASADI_PREFIX(ID) kinematic_model_cost_ext_cost_0_fun_jac_hess_ ## ID
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
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_s5 CASADI_PREFIX(s5)
#define casadi_s6 CASADI_PREFIX(s6)
#define casadi_sq CASADI_PREFIX(sq)

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

casadi_real casadi_sq(casadi_real x) { return x*x;}

static const casadi_int casadi_s0[11] = {7, 1, 0, 7, 0, 1, 2, 3, 4, 5, 6};
static const casadi_int casadi_s1[7] = {3, 1, 0, 3, 0, 1, 2};
static const casadi_int casadi_s2[3] = {0, 0, 0};
static const casadi_int casadi_s3[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s4[14] = {10, 1, 0, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
static const casadi_int casadi_s5[20] = {10, 10, 0, 1, 2, 3, 5, 7, 7, 7, 7, 7, 7, 0, 1, 2, 3, 4, 3, 4};
static const casadi_int casadi_s6[13] = {0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

/* kinematic_model_cost_ext_cost_0_fun_jac_hess:(i0[7],i1[3],i2[],i3[3])->(o0,o1[10],o2[10x10,7nz],o3[],o4[0x10]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a2, a3, a4, a5, a6, a7, a8, a9;
  a0=1.0000000000000000e-02;
  a1=arg[3]? arg[3][2] : 0;
  a2=sin(a1);
  a3=arg[0]? arg[0][0] : 0;
  a4=arg[3]? arg[3][0] : 0;
  a5=(a3-a4);
  a5=(a2*a5);
  a6=cos(a1);
  a7=arg[0]? arg[0][1] : 0;
  a8=arg[3]? arg[3][1] : 0;
  a9=(a7-a8);
  a9=(a6*a9);
  a5=(a5-a9);
  a9=casadi_sq(a5);
  a9=(a0*a9);
  a10=cos(a1);
  a3=(a3-a4);
  a3=(a10*a3);
  a1=sin(a1);
  a7=(a7-a8);
  a7=(a1*a7);
  a3=(a3+a7);
  a7=casadi_sq(a3);
  a9=(a9+a7);
  a7=1.0000000000000001e-01;
  a8=arg[1]? arg[1][0] : 0;
  a4=(a7*a8);
  a11=(a4*a8);
  a12=arg[1]? arg[1][1] : 0;
  a13=(a0*a12);
  a14=(a13*a12);
  a11=(a11+a14);
  a14=arg[1]? arg[1][2] : 0;
  a15=(a0*a14);
  a16=(a15*a14);
  a11=(a11+a16);
  a9=(a9+a11);
  a11=6.6666666666666666e-02;
  a16=arg[0]? arg[0][6] : 0;
  a11=(a11*a16);
  a9=(a9-a11);
  if (res[0]!=0) res[0][0]=a9;
  a7=(a7*a8);
  a4=(a4+a7);
  if (res[1]!=0) res[1][0]=a4;
  a12=(a0*a12);
  a13=(a13+a12);
  if (res[1]!=0) res[1][1]=a13;
  a14=(a0*a14);
  a15=(a15+a14);
  if (res[1]!=0) res[1][2]=a15;
  a3=(a3+a3);
  a15=(a10*a3);
  a5=(a5+a5);
  a5=(a0*a5);
  a14=(a2*a5);
  a15=(a15+a14);
  if (res[1]!=0) res[1][3]=a15;
  a3=(a1*a3);
  a5=(a6*a5);
  a3=(a3-a5);
  if (res[1]!=0) res[1][4]=a3;
  a3=0.;
  if (res[1]!=0) res[1][5]=a3;
  if (res[1]!=0) res[1][6]=a3;
  if (res[1]!=0) res[1][7]=a3;
  if (res[1]!=0) res[1][8]=a3;
  a3=-6.6666666666666666e-02;
  if (res[1]!=0) res[1][9]=a3;
  a3=2.0000000000000001e-01;
  if (res[2]!=0) res[2][0]=a3;
  a3=2.0000000000000000e-02;
  if (res[2]!=0) res[2][1]=a3;
  if (res[2]!=0) res[2][2]=a3;
  a3=(a10+a10);
  a3=(a10*a3);
  a5=(a2+a2);
  a5=(a0*a5);
  a5=(a2*a5);
  a3=(a3+a5);
  if (res[2]!=0) res[2][3]=a3;
  a3=(a1+a1);
  a10=(a10*a3);
  a5=(a6+a6);
  a0=(a0*a5);
  a2=(a2*a0);
  a10=(a10-a2);
  if (res[2]!=0) res[2][4]=a10;
  if (res[2]!=0) res[2][5]=a10;
  a1=(a1*a3);
  a6=(a6*a0);
  a1=(a1+a6);
  if (res[2]!=0) res[2][6]=a1;
  return 0;
}

CASADI_SYMBOL_EXPORT int kinematic_model_cost_ext_cost_0_fun_jac_hess(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int kinematic_model_cost_ext_cost_0_fun_jac_hess_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int kinematic_model_cost_ext_cost_0_fun_jac_hess_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void kinematic_model_cost_ext_cost_0_fun_jac_hess_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int kinematic_model_cost_ext_cost_0_fun_jac_hess_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void kinematic_model_cost_ext_cost_0_fun_jac_hess_release(int mem) {
}

CASADI_SYMBOL_EXPORT void kinematic_model_cost_ext_cost_0_fun_jac_hess_incref(void) {
}

CASADI_SYMBOL_EXPORT void kinematic_model_cost_ext_cost_0_fun_jac_hess_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int kinematic_model_cost_ext_cost_0_fun_jac_hess_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int kinematic_model_cost_ext_cost_0_fun_jac_hess_n_out(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_real kinematic_model_cost_ext_cost_0_fun_jac_hess_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* kinematic_model_cost_ext_cost_0_fun_jac_hess_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* kinematic_model_cost_ext_cost_0_fun_jac_hess_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    case 3: return "o3";
    case 4: return "o4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* kinematic_model_cost_ext_cost_0_fun_jac_hess_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    case 3: return casadi_s1;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* kinematic_model_cost_ext_cost_0_fun_jac_hess_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s3;
    case 1: return casadi_s4;
    case 2: return casadi_s5;
    case 3: return casadi_s2;
    case 4: return casadi_s6;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int kinematic_model_cost_ext_cost_0_fun_jac_hess_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 4;
  if (sz_res) *sz_res = 5;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
